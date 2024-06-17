#include "libsolv.h"

__device__
void print_double(double *x, int len, const char *s){
  __syncthreads();
  if(threadIdx.x==0 && blockIdx.x==0){
    for (int i=0; i<len; i++){
      printf("%s[%d]=%.17le\n",s,i,x[i]);
    }
  }
  __syncthreads();
}

__device__
void solveBcgCudaDeviceCVODE(ModelDataGPU *md){
  double alpha,rho0,omega0,beta,rho1,temp1,temp2;
  alpha=rho0=omega0=beta=rho1=temp1=temp2=1.0;
  int k=threadIdx.x + blockDim.x*blockIdx.x;
  double sum = 0.0;
  int nnz=md->diA[md->n_specs];
  ///*
  for (int j=0;j<md->n_specs;j++) {
    int i = j + k * md->n_specs;
    md->dn0[i] = 0.0;
    md->dp0[i] = 0.0;
    sum = 0.0;
    for (int j2 = md->diA[j]; j2 < md->diA[j + 1]; j2++) {
      sum += md->dx[md->djA[j2] + k * md->n_specs] * md->dA[j2 + k * nnz];
    }
    md->dr0[i] = sum;
  }
  for (int j=0;j<md->n_specs;j++) {
    int i = j + k * md->n_specs;
    md->dr0[i] = md->dtempv[i] - md->dr0[i];
    md->dr0h[i] = md->dr0[i];
  }
  //print_double(md->dx,md->n_specs,"dx37");
  int it = 0;
  while(it<1000 && temp1>1.0E-30){
  //while(it<1){
    rho1 = 0;
    for (int j=0;j<md->n_specs;j++) {
      int i = j + k * md->n_specs;
      rho1 += md->dr0[i] * md->dr0h[i];
    }
    beta = (rho1 / rho0) * (alpha / omega0); 
    for (int j=0;j<md->n_specs;j++) {
      int i = j + k * md->n_specs;
      md->dp0[i] = beta * md->dp0[i] + md->dr0[i] - md->dn0[i] * omega0 * beta;
      md->dy[i] = md->ddiag[i] * md->dp0[i];
    }
    
    for (int j=0;j<md->n_specs;j++) {
      int i = j + k * md->n_specs;
      sum = 0.0;
      for (int j2 = md->diA[j]; j2 < md->diA[j + 1]; j2++) {
        sum += md->dy[md->djA[j2] + k * md->n_specs] * md->dA[j2 + k * nnz];
      }
      md->dn0[i] = sum;
    }
    //print_double(md->dn0,md->n_specs,"dn059");
    temp1 = 0;
    for (int j=0;j<md->n_specs;j++) {
      int i = j + k * md->n_specs;
      temp1 += md->dr0h[i] * md->dn0[i];
    }
    alpha = rho1 / temp1;
    for (int j=0;j<md->n_specs;j++) {
      int i = j + k * md->n_specs;
      md->ds[i] = md->dr0[i] - alpha * md->dn0[i];
      md->dx[i] += alpha * md->dy[i];
      md->dy[i] = md->ddiag[i] * md->ds[i];
    }
    for (int j=0;j<md->n_specs;j++) {
      int i = j + k * md->n_specs;
      sum = 0.0;
      for (int j2 = md->diA[j]; j2 < md->diA[j + 1]; j2++) {
        sum += md->dy[md->djA[j2] + k * md->n_specs] * md->dA[j2 + k * nnz];
      }
      md->dt[i] = sum;
    }
    for (int j=0;j<md->n_specs;j++) {
      int i = j + k * md->n_specs;
      md->dr0[i] = sum;
      md->dr0[i] = md->ddiag[i] * md->dt[i];
    }
    temp1 = 0;
    temp2 = 0;
    for (int j=0;j<md->n_specs;j++) {
      int i = j + k * md->n_specs;
      temp1 += md->dy[i] * md->dr0[i];
      temp2 += md->dr0[i] * md->dr0[i];
    }
    omega0 = temp1 / temp2;
    for (int j=0;j<md->n_specs;j++) {
      int i = j + k * md->n_specs;
      md->dx[i] += omega0 * md->dy[i];
      md->dr0[i] = md->ds[i] - omega0 * md->dt[i];
      md->dt[i] = 0.0;
    }
    temp1 = 0;
    for (int j=0;j<md->n_specs;j++) {
      int i = j + k * md->n_specs;
      temp1 += md->dr0[i] * md->dr0[i];
    }
    temp1 = sqrt(temp1);
    rho0 = rho1;
    //*/
    it++;
    //print_double(md->dx,md->n_specs,"dx105");
  }
}

__global__
void cudaGlobalCVode(ModelDataGPU md_object) {
  ModelDataGPU *md = &md_object;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<md->n_cells)
    solveBcgCudaDeviceCVODE(md);
}


void solveGPU_block(ModelDataGPU* mGPU){
  mGPU->n_specs = mGPU->nrows / mGPU->n_cells;
  int threads_block = 1024;
  int blocks = (mGPU->n_cells+threads_block-1)/threads_block;
  cudaGlobalCVode<<< blocks, threads_block >>>(*mGPU);
}
