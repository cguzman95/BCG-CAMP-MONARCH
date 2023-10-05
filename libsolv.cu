/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#include "libsolv.h"

__device__ void cudaDeviceSpmvCSR(double* dx, double* db, double* dA, int* djA, int* diA){
  int k=threadIdx.x + blockDim.x*blockIdx.x;
  for (int j=0;j<md->n_specs;j++){
    int i=j+k*md->n_specs;
    double sum = 0.0;
    int nnz=md->n_specs;


    for(int j=diA[threadIdx.x]; j<diA[threadIdx.x+1]; j++){
      sum+= db[djA[j]+blockDim.x*blockIdx.x]*dA[j+nnz*blockIdx.x];
    }
    dx[i]=sum;
  }


}

__device__ void cudaDeviceSpmvCSC(double* dx, double* db, double* dA, int* djA, int* diA){
  double mult;
  extern __shared__ double sdata[];
  int i= threadIdx.x + blockDim.x*blockIdx.x;
  unsigned int tid = threadIdx.x;
  dx[i]=0.0;
  int nnz=diA[blockDim.x];
  for(int j=diA[threadIdx.x]; j<diA[threadIdx.x+1]; j++){
    mult = db[i]*dA[j+nnz*blockIdx.x];
    atomicAdd_block(&(dx[djA[j]+blockDim.x*blockIdx.x]),mult);
  }
}

__device__ void cudaDeviceSpmv(double* dx, double* db, double* dA, int* djA, int* diA)
{
#ifdef CSC
  printf("NOT IMPLEMENTED cudaDeviceSpmvCSC\n");
  cudaDeviceSpmvCSC(dx,db,dA,djA,diA);
#else
  cudaDeviceSpmvCSR(dx,db,dA,djA,diA);
#endif
}

__device__
void solveBcgCudaDeviceCVODE(ModelDataGPU *md){
  double alpha,rho0,omega0,beta,rho1,temp1,temp2;
  alpha=rho0=omega0=beta=rho1=temp1=temp2=1.0;
  int k=threadIdx.x + blockDim.x*blockIdx.x;
  for (int j=0;j<md->n_specs;j++){
    int i=j+k*md->n_specs;
    md->dn0[i]=0.0;
    md->dp0[i]=0.0;
  }
  cudaDeviceSpmv(md->dr0,md->dx,md->dA,md->djA,md->diA, md->n_shr_empty);
  md->dr0[i]=md->dtempv[i]-md->dr0[i];
  md->dr0h[i]=md->dr0[i];
  int it=0;
  while(it<1000 && temp1>1.0E-30){
    rho1=0;
    rho1 += md->dr0[i]*md->dr0h[i];
    beta = (rho1 / rho0) * (alpha / omega0);
    md->dp0[i]=beta*md->dp0[i]+md->dr0[i]-md->dn0[i]*omega0*beta;
    md->dy[i]=md->ddiag[i]*md->dp0[i];
    cudaDeviceSpmv(md->dn0, md->dy, md->dA, md->djA, md->diA, md->n_shr_empty);
    temp1=0;
    temp1 += md->dr0h[i]*md->dn0[i];
    alpha = rho1 / temp1;
    md->ds[i]=md->dr0[i]-alpha*md->dn0[i];
    md->dx[i]+=alpha*md->dy[i];
    md->dy[i]=md->ddiag[i]*md->ds[i];
    cudaDeviceSpmv(md->dt, md->dy, md->dA, md->djA, md->diA, md->n_shr_empty);
    md->dr0[i]=md->ddiag[i]*md->dt[i];
    temp1=0;
    temp1 += md->dy[i]*md->dr0[i];
    temp2=0;
    temp2 += md->dr0[i]*md->dr0[i];
    omega0 = temp1 / temp2;
    md->dx[i]+=omega0*md->dy[i];
    md->dr0[i]=md->ds[i]-omega0*md->dt[i];
    md->dt[i]=0.0;
    temp1=0;
    temp1 += md->dr0[i]*md->dr0[i];
    cudaDevicedotxy(md->dr0, md->dr0, &temp1, md->n_shr_empty);
    temp1 = sqrt(temp1);
    rho0 = rho1;
    it++;
    __syncthreads();
  }
}

__global__
void cudaGlobalCVode(ModelDataGPU md_object) {
  ModelDataGPU *md = &md_object;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<md->nrows) {
    solveBcgCudaDeviceCVODE(md);
  }
}


void solveGPU_block(ModelDataGPU* mGPU){
  int len_cell = mGPU->nrows / mGPU->n_cells;
  int threads_block = 1024;
  int blocks = (mGPU->n_cells+threads_block-1)/threads_block;
  cudaGlobalCVode<<< blocks, threads_block >>>(*mGPU);
}
