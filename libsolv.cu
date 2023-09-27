/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#include "libsolv.h"
#include <stdio.h>
#include <stdlib.h>

int nextPowerOfTwoBCG(int v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

void cudaDeviceSpmv_2(ModelDataGPU* md, double* dx, double* db, double* dA, int* djA, int* diA){
  for(int row=0;row<md->nrows;row++){
    dx[row] = 0.0;
  }
  for(int row=0;row<md->nrows;row++){
    for (int j = diA[row]; j < diA[row + 1]; j++) {
      double mult = db[row] * dA[j];
      int i_dx=djA[j];
      dx[i_dx] += mult;
    }
  }
}

void cudaDevicedotxy_2(ModelDataGPU* md, double *g_idata1, double *g_idata2,
                       double *g_odata){
  *g_odata=0.;
  for(int i=0;i<md->nrows;i++){
    *g_odata+=g_idata1[i]*g_idata2[i];
  }
}

void solveGPU_block(ModelDataGPU* md){
  int len_cell = md->nrows / md->n_cells;
  int threads_block = len_cell;
  int n_shr_memory = nextPowerOfTwoBCG(len_cell);
  md->n_shr_empty = n_shr_memory - threads_block;
  double alpha,rho0,omega0,beta,rho1,temp1,temp2;
  alpha=rho0=omega0=beta=rho1=temp1=temp2=1.0;
  for(int i=0; i<md->nrows; i++){
    md->dn0[i]=0.0;
    md->dp0[i]=0.0;
  }
  cudaDeviceSpmv_2(md,md->dr0,md->dx,md->dA,md->djA,md->diA);
  for(int i=0;i<md->nrows;i++){
    md->dr0[i] = md->dtempv[i] - md->dr0[i];
    md->dr0h[i] = md->dr0[i];
  }
  int it=0;
  while(it<1000 && temp1>1.0E-30){
    cudaDevicedotxy_2(md,md->dr0, md->dr0h, &rho1);
    beta = (rho1 / rho0) * (alpha / omega0);
    for (int i = 0; i < md->nrows; i++) {
      md->dp0[i] =
          beta * md->dp0[i] + md->dr0[i] - md->dn0[i] * omega0 * beta;
      md->dy[i] = md->ddiag[i] * md->dp0[i];
    }
    cudaDeviceSpmv_2(md,md->dn0, md->dy, md->dA, md->djA, md->diA);
    cudaDevicedotxy_2(md,md->dr0h, md->dn0, &temp1);
    alpha = rho1 / temp1;
    for (int i = 0; i < md->nrows; i++){
      md->ds[i] = md->dr0[i] - alpha * md->dn0[i];
      md->dx[i] += alpha * md->dy[i];
      md->dy[i] = md->ddiag[i] * md->ds[i];
    }
    cudaDeviceSpmv_2(md,md->dt, md->dy, md->dA, md->djA, md->diA);
    for (int i = 0; i < md->nrows; i++) {
      md->dr0[i] = md->ddiag[i] * md->dt[i];
    }
    cudaDevicedotxy_2(md,md->dy, md->dr0, &temp1);
    cudaDevicedotxy_2(md,md->dr0, md->dr0, &temp2);
    omega0 = temp1 / temp2;
    for (int i = 0; i < md->nrows; i++) {
      md->dx[i] += omega0 * md->dy[i];
      md->dr0[i] = md->ds[i] - omega0 * md->dt[i];
      md->dt[i] = 0.0;
    }
    cudaDevicedotxy_2(md,md->dr0, md->dr0, &temp1);
    temp1 = sqrt(temp1);
    rho0 = rho1;
    it++;
  }
  if(it>=1000){
    printf("it>=BCG_MAXIT\n %d>%d",it,1000);
    exit(0);
  }
  TODO: CPU TO GPU
  cudaMemcpyAsync(x + offset_nrows, mGPU->dx, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost, 0);
}
