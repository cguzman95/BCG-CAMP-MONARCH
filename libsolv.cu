/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
* Illinois at Urbana-Champaign
* SPDX-License-Identifier: MIT
*/

#include "libsolv.h"


using namespace std;

__global__ void cudamatScaleAddI(int nrows, double* dA, int* djA, int* diA, double alpha)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows)
  {
    int jstart = diA[row];
    int jend   = diA[row+1];
    for(int j=jstart; j<jend; j++)
    {
      if(djA[j]==row)
      {
        dA[j] = 1.0 + alpha*dA[j];
      }
      else{
        dA[j] = alpha*dA[j];
      }
    }
  }
}

// A = I - gamma*J
// dA  : Matrix values (nnz size)
// djA : Matrix columns (nnz size)
// diA : Matrix rows (nrows+1 size)
// alpha : Scale factor
void gpu_matScaleAddI(int nrows, double* dA, int* djA, int* diA, double alpha, int blocks, int threads)
{

  blocks = (nrows+threads-1)/threads;

  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);

  cudamatScaleAddI<<<dimGrid,dimBlock>>>(nrows, dA, djA, diA, alpha);
}

__global__
    void check_input_gpud(double *x, int len, int var_id)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  printf("%d[%d]=%-le\n",var_id,i,x[i]);

}

// Diagonal precond
__global__ void cudadiagprecond(int nrows, double* dA, int* djA, int* diA, double* ddiag)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows){
    int jstart=diA[row];
    int jend  =diA[row+1];
    for(int j=jstart;j<jend;j++){
      if(djA[j]==row){
        if(dA[j]!=0.0)
          ddiag[row]= 1.0/dA[j];
        else{
          ddiag[row]= 1.0;
        }
      }
    }
  }
}

void gpu_diagprecond(int nrows, double* dA, int* djA, int* diA, double* ddiag, int blocks, int threads)
{
  blocks = (nrows+threads-1)/threads;
  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);
  cudadiagprecond<<<dimGrid,dimBlock>>>(nrows, dA, djA, diA, ddiag);
}

__global__ void cudasetconst(double* dy,double constant,int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows){
    dy[row]=constant;
  }
}

__global__ void cudasetconst1(double* dy)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  dy[i]=0;
}

__global__ void cudasetconst2(double* dy)
{
  dy[0]=0;
}


void gpu_yequalsconst(double *dy, double constant, int nrows, int blocks, int threads)
{
  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);
  /*
  cudasetconst2<<<512,1024>>>(dy);
  cudasetconst1<<<256,1024>>>(dy);
  cudasetconst1<<<512,1024>>>(dy);
  cudasetconst1<<<1024,1024>>>(dy);
  cudasetconst1<<<2048,1024>>>(dy);
  cudasetconst1<<<4092,1024>>>(dy);
   */
}

__global__ void cudaSpmvCSC(double* dx, double* db, double* dA, int* djA, int* diA)
{
  double mult;
  int i= threadIdx.x + blockDim.x*blockIdx.x;
  __syncthreads();
  dx[i]=0.0;
  __syncthreads();
  int nnz=diA[blockDim.x];
  for(int j=diA[threadIdx.x]; j<diA[threadIdx.x+1]; j++){
    mult = db[i]*dA[j+nnz*blockIdx.x];
    atomicAdd_block(&(dx[djA[j]+blockDim.x*blockIdx.x]),mult);
  }
  __syncthreads();
}

__global__ void cudaSpmvCSR(double* dx, double* db, double* dA, int* djA, int* diA)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  double sum = 0.0;
  int nnz=diA[blockDim.x];
  for(int j=diA[threadIdx.x]; j<diA[threadIdx.x+1]; j++){
    sum+= db[djA[j]+blockDim.x*blockIdx.x]*dA[j+nnz*blockIdx.x];
  }
  __syncthreads();
  dx[i]=sum;
}

__global__ void cudaSpmvCSR4(double* dx, double* db, double* dA, int* djA, int* diA)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  double sum = 0.0;
  int nnz=diA[blockDim.x];
  for(int j=diA[threadIdx.x]; j<diA[threadIdx.x+1]; j++){
    sum+= dA[j+nnz*blockIdx.x];
  }
  __syncthreads();
  dx[i]=sum;
}

__global__ void cudaSpmvCSR3(double* dx, double* db, double* dA, int* djA, int* diA)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  double sum = 0.0;
  int nnz=diA[blockDim.x];
  for(int j=diA[threadIdx.x]; j<diA[threadIdx.x+1]; j++){
    sum+= dA[0+nnz*blockIdx.x];
    sum+= dA[1+nnz*blockIdx.x];
  }
  __syncthreads();
  dx[i]=sum;
}

__global__ void cudaSpmvCSR2(double* dx, double* db, double* dA, int* djA, int* diA)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  double sum = 0.0;
  int nnz=diA[blockDim.x];
  for(int j=diA[threadIdx.x]; j<diA[threadIdx.x+1]; j++){
    sum+= dA[0+nnz*blockIdx.x];
  }
  __syncthreads();
  dx[i]=sum;
}

__global__ void cudaSpmvCSR5(double* dx, double* db, double* dA, int* djA, int* diA)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  extern __shared__ double sdata[];
  if(i<1){
  unsigned int tid = threadIdx.x;
  double sum = 0.0;
  int nnz=diA[blockDim.x];
  for(int j=diA[threadIdx.x]; j<diA[threadIdx.x+1]; j++){
    sum+= dA[0+nnz*blockIdx.x];
    sum+= dA[1+nnz*blockIdx.x];
  }
  sum+= dA[2+nnz*blockIdx.x];
  sum+= dA[3+nnz*blockIdx.x];
  __syncthreads();
  dx[i]=sum;
  }
}

void gpu_spmv(double* dx ,double* db, double* dA, int *djA,int *diA,int blocks,int threads,int shr)
{
  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);
#ifdef CSC
  cudaSpmvCSC<<<blocks,threads>>>(dx, db, dA, djA, diA);
#else
  cudaSpmvCSR5<<<1,1>>>(dx, db, dA, djA, diA);
#endif
}

// y= a*x+ b*y
__global__ void cudaaxpby(double* dy,double* dx, double a, double b, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows){
    dy[row]= a*dx[row] + b*dy[row];
  }
}

void gpu_axpby(double* dy ,double* dx, double a, double b, int nrows, int blocks, int threads)
{

  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);

  cudaaxpby<<<dimGrid,dimBlock>>>(dy,dx,a,b,nrows);
}

// y = x
__global__ void cudayequalsx(double* dy,double* dx,int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows){
    dy[row]=dx[row];
  }
}

void gpu_yequalsx(double *dy, double* dx, int nrows, int blocks, int threads)
{
  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);

  cudayequalsx<<<dimGrid,dimBlock>>>(dy,dx,nrows);

}

__global__ void cudareducey(double *g_o, unsigned int n)
{
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;

  double mySum =  (tid < n) ? g_o[tid] : 0;

  sdata[tid] = mySum;
  __syncthreads();

  for (unsigned int s=blockDim.x/2; s>0; s>>=1)
  {
    if (tid < s)
      sdata[tid] = mySum = mySum + sdata[tid + s];

    __syncthreads();
  }

  if (tid == 0) g_o[blockIdx.x] = sdata[0];
}

__device__ void warpReduce_2(volatile double *sdata, unsigned int tid) {
  unsigned int blockSize = blockDim.x;
  if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

__global__ void cudadotxy(double *g_i1, double *g_i2, double *g_o, int n_shr_empty)
{
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  __syncthreads();
  if(tid<n_shr_empty)
    sdata[tid+blockDim.x]=0.;
  __syncthreads();
  //print_double(sdata,73,"sdata");
#ifdef DEV_cudaDevicedotxy_2
  //used for compare with cpu
  sdata[0]=0.;
  __syncthreads();
  if(tid==0){
    for(int j=0;j<blockDim.x;j++){
      sdata[0]+=g_i1[j+blockIdx.x*blockDim.x]*g_i2[j+blockIdx.x*blockDim.x];
    }
  }
#else
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<n_shr_empty)
    sdata[tid+blockDim.x]=0.;
  __syncthreads();
  sdata[tid] = g_i1[i]*g_i2[i];
  __syncthreads();
  unsigned int blockSize = blockDim.x+n_shr_empty;
  if ((blockSize >= 1024) && (tid < 512)) {
    sdata[tid] += sdata[tid + 512];
  }
  __syncthreads();
  if ((blockSize >= 512) && (tid < 256)) {
    sdata[tid] += sdata[tid + 256];
  }
  __syncthreads();
  if ((blockSize >= 256) && (tid < 128)) {
    sdata[tid] += sdata[tid + 128];
  }
  __syncthreads();
  if ((blockSize >= 128) && (tid < 64)) {
    sdata[tid] += sdata[tid + 64];
  }
  __syncthreads();
  if (tid < 32) warpReduce_2(sdata, tid);
#endif
  __syncthreads();
  *g_o = sdata[0];
  __syncthreads();
}

void gpu_dotxy(double* g_i1, double* g_i2, double* sum, double* g_o, int nshre, int blocks,int threads,int shr)
{
  cudadotxy<<<blocks,threads,shr>>>(g_i1,g_i2,g_o,nshre);
  cudaMemcpy(&sum, g_o, sizeof(double), cudaMemcpyDeviceToHost);
}

// z= a*z + x + b*y
__global__ void cudazaxpbypc(double* dz, double* dx,double* dy, double a, double b, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows){
    dz[row]=a*dz[row]  + dx[row] + b*dy[row];
  }
}

void gpu_zaxpbypc(double* dz, double* dx ,double* dy, double a, double b, int nrows, int blocks, int threads)
{

  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);

  cudazaxpbypc<<<dimGrid,dimBlock>>>(dz,dx,dy,a,b,nrows);
}

// z= x*y
__global__ void cudamultxy(double* dz, double* dx,double* dy, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows){
    dz[row]=dx[row]*dy[row];
  }
}

void gpu_multxy(double* dz, double* dx ,double* dy, int nrows, int blocks, int threads)
{

  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);

  cudamultxy<<<dimGrid,dimBlock>>>(dz,dx,dy,nrows);
}

// a*x + b*y = z
//__global__ void cudazaxpby(double* dz, double* dx,double* dy, double a, double b, int nrows)
__global__ void cudazaxpby(double a, double* dx, double b, double* dy, double* dz, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows){
    dz[row]=a*dx[row] + b*dy[row];
  }
}

void gpu_zaxpby(double a, double* dx, double b, double* dy, double* dz, int nrows, int blocks, int threads)
{

  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);

  cudazaxpby<<<dimGrid,dimBlock>>>(a,dx,b,dy,dz,nrows);
}

// y= a*x + y
__global__ void cudaaxpy(double* dy,double* dx, double a, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows){
    dy[row]=a*dx[row] + dy[row];
  }
}

void gpu_axpy(double* dy, double* dx ,double a, int nrows, int blocks, int threads)
{

  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);

  cudaaxpy<<<dimGrid,dimBlock>>>(dy,dx,a,nrows);
}

// sqrt(sum ( (x_i*y_i)^2)/n)
__global__ void cudaDVWRMS_Norm(double *g_i1, double *g_i2, double *g_o, unsigned int n)
{
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

  double mySum = (i < n) ? g_i1[i]*g_i1[i]*g_i2[i]*g_i2[i] : 0;

  if (i + blockDim.x < n)
    mySum += g_i1[i+blockDim.x]*g_i1[i+blockDim.x]*g_i2[i+blockDim.x]*g_i2[i+blockDim.x];

  sdata[tid] = mySum;
  __syncthreads();

  for (unsigned int s=blockDim.x/2; s>0; s>>=1)
  {
    if (tid < s)
      sdata[tid] = mySum = mySum + sdata[tid + s];

    __syncthreads();
  }

  if (tid == 0) g_o[blockIdx.x] = sdata[0];
}

double gpu_VWRMS_Norm(int n, double* vec1,double* vec2,double* h_temp,double* d_temp, int blocks,int threads)
{
  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);
  cudaDVWRMS_Norm<<<dimGrid,dimBlock,threads*sizeof(double)>>>(vec1,vec2,d_temp,n);
  int redsize= sqrt(blocks) +1;
  redsize=pow(2,redsize);
  dim3 dimGrid2(1,1,1);
  dim3 dimBlock2(redsize,1,1);
  cudareducey<<<dimGrid2,dimBlock2,redsize*sizeof(double)>>>(d_temp,blocks);
  double sum;
  cudaMemcpy(&sum, d_temp, sizeof(double), cudaMemcpyDeviceToHost);
  return sqrt(sum/n);
}

// y=alpha*y
__global__ void cudascaley(double* dy, double a, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows){
    dy[row]=a*dy[row];
  }
}

void gpu_scaley(double* dy, double a, int nrows, int blocks, int threads)
{
  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);

  cudascaley<<<dimGrid,dimBlock>>>(dy,a,nrows);
}


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

void solveGPU_block(ModelDataGPU* md){
  double *dA = md->dA;
  int *djA = md->djA;
  int *diA = md->diA;
  int nrows = md->nrows;
  int blocks = md->n_cells;
  int threads = md->nrows / md->n_cells;
  int n_shr_memory = nextPowerOfTwoBCG(threads);
  int shr = n_shr_memory * sizeof(double);
  int nshre=md->n_shr_empty;
  double *ddiag = md->ddiag;
  double *dr0 = md->dr0;
  double *dr0h = md->dr0h;
  double *dn0 = md->dn0;
  double *dp0 = md->dp0;
  double *dt = md->dt;
  double *ds = md->ds;
  double *dy = md->dy;
  double *dtempv = md->dtempv;
  double *dx = md->dx;
  double *dtemp = md->dtemp;

  printf("DEBUG\n");
  double alpha,rho0,omega0,beta,rho1,temp1,temp2;
  alpha=rho0=omega0=beta=rho1=temp1=temp2=1.0;
  gpu_spmv(dr0,dx,dA,djA,diA,1,threads,shr);
  /*
  gpu_spmv(dr0,dx,dA,djA,diA,2,threads,shr);
  gpu_spmv(dr0,dx,dA,djA,diA,4,threads,shr);
  gpu_spmv(dr0,dx,dA,djA,diA,512,threads,shr);
  gpu_spmv(dr0,dx,dA,djA,diA,1024,threads,shr);
*/
/*
  gpu_yequalsconst(dn0,0.0,nrows,blocks,threads);
  gpu_yequalsconst(dp0,0.0,nrows,blocks,threads);
  gpu_axpby(dr0,dtempv,1.0,-1.0,nrows,blocks,threads);
  gpu_yequalsx(dr0h,dr0,nrows,blocks,threads);
  int it=0;
  while(it<1 && temp1>1.0E-30){
    gpu_dotxy(dr0,dr0h,&rho1,dtemp,nshre,blocks,threads,shr);
    beta=(rho1/rho0)*(alpha/omega0);
    gpu_zaxpbypc(dp0,dr0,dn0,beta,-1.0*omega0*beta,nrows,blocks,threads);
    gpu_multxy(dy,ddiag,dp0,nrows,blocks,threads);
    gpu_spmv(dn0,dy,dA,djA,diA,blocks,threads,shr);
    gpu_dotxy(dr0,dr0h,&temp1,dtemp,nshre,blocks,threads,shr);
    alpha=rho1/temp1;
    gpu_zaxpby(1.0,dr0,-1.0*alpha,dn0,ds,nrows,blocks,threads);
    gpu_axpy(dx,dy,alpha,nrows,blocks,threads);
    gpu_multxy(dy,ddiag,ds,nrows,blocks,threads);
    gpu_spmv(dt,dy,dA,djA,diA,blocks,threads,shr);
    gpu_multxy(dr0,ddiag,dt,nrows,blocks,threads);
    gpu_dotxy(dy,dr0,&temp1,dtemp,nshre,blocks,threads,shr);
    gpu_dotxy(dr0,dr0,&temp2,dtemp,nshre,blocks,threads,shr);
    omega0= temp1/temp2;
    gpu_axpy(dx,dy,omega0,nrows,blocks,threads);
    gpu_zaxpby(1.0,ds,-1.0*omega0,dt,dr0,nrows,blocks,threads);
    gpu_dotxy(dr0,dr0,&temp1,dtemp,nshre,blocks,threads,shr);
    temp1=sqrt(temp1);
    rho0=rho1;
    it++;
  }
  */
}