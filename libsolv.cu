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
extern "C++" void gpu_matScaleAddI(int nrows, double* dA, int* djA, int* diA, double alpha, int blocks, int threads)
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
          //printf("cudadiagprecond else\n");
          ddiag[row]= 1.0;
        }
      }
    }
  }

}

extern "C++" void gpu_diagprecond(int nrows, double* dA, int* djA, int* diA, double* ddiag, int blocks, int threads)
{

  blocks = (nrows+threads-1)/threads;

  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);

  cudadiagprecond<<<dimGrid,dimBlock>>>(nrows, dA, djA, diA, ddiag);
  //check_input_gpud<< < 1, 5>> >(ddiag,nrows,0);
}

// y = constant
__global__ void cudasetconst(double* dy,double constant,int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows){
    dy[row]=constant;
  }
}

extern "C++" void gpu_yequalsconst(double *dy, double constant, int nrows, int blocks, int threads)
{
  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);

  cudasetconst<<<dimGrid,dimBlock>>>(dy,constant,nrows);

}


// x=A*b
__global__ void cudaSpmvCSR(double* dx, double* db, int nrows, double* dA, int* djA, int* diA)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows)
  {
    int jstart = diA[row];
    int jend   = diA[row+1];
    double sum = 0.0;
    for(int j=jstart; j<jend; j++)
    {
      sum+= db[djA[j]]*dA[j];
    }
    dx[row]=sum;
  }

}

__global__ void cudaSpmvCSC(double* dx, double* db, int nrows, double* dA, int* djA, int* diA)
{
  double mult;
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows)
  {
    int jstart = diA[row];
    int jend   = diA[row+1];
    for(int j=jstart; j<jend; j++)
    {
      mult = db[row]*dA[j];
      atomicAdd(&(dx[djA[j]]),mult);
    }
  }
}

extern "C++" void gpu_spmv(double* dx ,double* db, int nrows, double* dA, int *djA,int *diA,int blocks,int  threads)
{
  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);
#ifdef CSC
  cudasetconst<<<dimGrid,dimBlock>>>(dx, 0.0, nrows);
  cudaSpmvCSC<<<dimGrid,dimBlock>>>(dx, db, nrows, dA, djA, diA);
#else
  cudaSpmvCSR<<<dimGrid,dimBlock>>>(dx, db, nrows, dA, djA, diA);
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

extern "C++" void gpu_axpby(double* dy ,double* dx, double a, double b, int nrows, int blocks, int threads)
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

extern "C++" void gpu_yequalsx(double *dy, double* dx, int nrows, int blocks, int threads)
{
  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);

  cudayequalsx<<<dimGrid,dimBlock>>>(dy,dx,nrows);

}

__global__ void cudareducey(double *g_odata, unsigned int n)
{
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;

  double mySum =  (tid < n) ? g_odata[tid] : 0;

  sdata[tid] = mySum;
  __syncthreads();

  for (unsigned int s=blockDim.x/2; s>0; s>>=1)
  {
    if (tid < s)
      sdata[tid] = mySum = mySum + sdata[tid + s];

    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void cudadotxy(double *g_idata1, double *g_idata2, double *g_odata, unsigned int n)
{
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;//*2 because init blocks is half
  //unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;//*2 because init blocks is half

  double mySum = (i < n) ? g_idata1[i]*g_idata2[i] : 0;

  if (i + blockDim.x < n)
    mySum += g_idata1[i+blockDim.x]*g_idata2[i+blockDim.x];

  sdata[tid] = mySum;
  __syncthreads();

  //for (unsigned int s=(blockDim.x+1)/2; s>0; s>>=1)
  for (unsigned int s=blockDim.x/2; s>0; s>>=1)
  {
    if (tid < s)
      sdata[tid] = mySum = mySum + sdata[tid + s];

    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

//threads need to be pow of 2 //todo h_temp not needed
extern "C++" double gpu_dotxy(double* vec1, double* vec2, double* h_temp, double* d_temp, int nrows, int blocks,int threads)
{
  double sum;
  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);

  //threads*sizeof(double)
  cudadotxy<<<dimGrid,dimBlock,threads*sizeof(double)>>>(vec1,vec2,d_temp,nrows);
  cudaMemcpy(&sum, d_temp, sizeof(double), cudaMemcpyDeviceToHost);
  //printf("rho1 %f", sum);

  int redsize= sqrt(blocks) +1;
  redsize=pow(2,redsize);

  dim3 dimGrid2(1,1,1);
  dim3 dimBlock2(redsize,1,1);

  cudareducey<<<dimGrid2,dimBlock2,redsize*sizeof(double)>>>(d_temp,blocks);
  cudaMemcpy(&sum, d_temp, sizeof(double), cudaMemcpyDeviceToHost);

  return sum;

  /*
    cudaMemcpy(h_temp, d_temp, blocks * sizeof(double), cudaMemcpyDeviceToHost);
    double sum=0;
    for(int i=0;i<blocks;i++)
    {
      sum+=h_temp[i];
    }
    return sum;
  */
  /*dim3 dimGrid2(1,1,1);
  dim3 dimBlock2(blocks,1,1);

  //Cuda only sum kernel call
  //cudareducey<<<dimGrid2,dimBlock2,blocks*sizeof(double)>>>(d_temp,blocks); //Takes quasi WAY MORE than cpu calc

  cudaMemcpy(h_temp, d_temp, sizeof(double), cudaMemcpyDeviceToHost);
  return h_temp[0];*/
}

/*
extern "C++" double gpu_dotxy(double *dy, double* dx, int nrows)
{
   double dot=0.0;
   cublasHandle_t hl;
   cublasCreate(&hl);

   cublasDdot(hl,nrows,dy,1,dx,1,&dot);

   cublasDestroy(hl);
   return dot;
}
*/

// z= a*z + x + b*y
__global__ void cudazaxpbypc(double* dz, double* dx,double* dy, double a, double b, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows){
    dz[row]=a*dz[row]  + dx[row] + b*dy[row];
  }
}

extern "C++" void gpu_zaxpbypc(double* dz, double* dx ,double* dy, double a, double b, int nrows, int blocks, int threads)
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

extern "C++" void gpu_multxy(double* dz, double* dx ,double* dy, int nrows, int blocks, int threads)
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

extern "C++" void gpu_zaxpby(double a, double* dx, double b, double* dy, double* dz, int nrows, int blocks, int threads)
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

extern "C++" void gpu_axpy(double* dy, double* dx ,double a, int nrows, int blocks, int threads)
{

  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);

  cudaaxpy<<<dimGrid,dimBlock>>>(dy,dx,a,nrows);
}

// sqrt(sum ( (x_i*y_i)^2)/n)
__global__ void cudaDVWRMS_Norm(double *g_idata1, double *g_idata2, double *g_odata, unsigned int n)
{
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

  double mySum = (i < n) ? g_idata1[i]*g_idata1[i]*g_idata2[i]*g_idata2[i] : 0;

  if (i + blockDim.x < n)
    mySum += g_idata1[i+blockDim.x]*g_idata1[i+blockDim.x]*g_idata2[i+blockDim.x]*g_idata2[i+blockDim.x];

  sdata[tid] = mySum;
  __syncthreads();

  for (unsigned int s=blockDim.x/2; s>0; s>>=1)
  {
    if (tid < s)
      sdata[tid] = mySum = mySum + sdata[tid + s];

    __syncthreads();
  }

  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

extern "C++" double gpu_VWRMS_Norm(int n, double* vec1,double* vec2,double* h_temp,double* d_temp, int blocks,int threads)
{
  dim3 dimGrid(blocks,1,1);
  dim3 dimBlock(threads,1,1);

  cudaDVWRMS_Norm<<<dimGrid,dimBlock,threads*sizeof(double)>>>(vec1,vec2,d_temp,n);

  //cudaMemcpy(h_temp, d_temp, blocks * sizeof(double), cudaMemcpyDeviceToHost);

  int redsize= sqrt(blocks) +1;
  redsize=pow(2,redsize);

  dim3 dimGrid2(1,1,1);
  dim3 dimBlock2(redsize,1,1);

  cudareducey<<<dimGrid2,dimBlock2,redsize*sizeof(double)>>>(d_temp,blocks);

  double sum;
  cudaMemcpy(&sum, d_temp, sizeof(double), cudaMemcpyDeviceToHost);

  return sqrt(sum/n);

  /*
    double sum=0;
    for(int i=0;i<blocks;i++)
    {
      sum+=h_temp[i];
    }
    return sqrt(sum/n);
    */
}

// y=alpha*y
__global__ void cudascaley(double* dy, double a, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  if(row < nrows){
    dy[row]=a*dy[row];
  }
}

extern "C++" void gpu_scaley(double* dy, double a, int nrows, int blocks, int threads)
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

void solveGPU_block(ModelDataGPU* mGPU){
  double *dA = mGPU->dA;
  int *djA = mGPU->djA;
  int *diA = mGPU->diA;
  int nrows = mGPU->nrows;
  int blocks = mGPU->blocks;
  int threads = mGPU->threads;
  int maxIt = mGPU->maxIt;
  double tolmax = mGPU->tolmax;
  double *ddiag = mGPU->ddiag;
  double *dr0 = mGPU->dr0;
  double *dr0h = mGPU->dr0h;
  double *dn0 = mGPU->dn0;
  double *dp0 = mGPU->dp0;
  double *dt = mGPU->dt;
  double *ds = mGPU->ds;
  double *dAx2 = mGPU->dAx2;
  double *dy = mGPU->dy;
  double *dz = mGPU->dz;
  double *aux = mGPU->aux;
  double *dtempv = mGPU->dtempv;

  double alpha,rho0,omega0,beta,rho1,temp1,temp2;
  alpha=rho0=omega0=beta=rho1=temp1=temp2=1.0;
  gpu_spmv(dr0,dx,nrows,dA,djA,diA,blocks,threads);
  gpu_axpby(dr0,dtempv,1.0,-1.0,nrows,blocks,threads);
  gpu_yequalsx(dr0h,dr0,nrows,blocks,threads);
  gpu_yequalsconst(dn0,0.0,nrows,blocks,threads);
  gpu_yequalsconst(dp0,0.0,nrows,blocks,threads);
  int it=0;
  while(it<1000 && temp1>1.0E-30){
    rho1=gpu_dotxy(dr0, dr0h, aux, dtempv, nrows,(blocks + 1) / 2, threads);
    beta=(rho1/rho0)*(alpha/omega0);
    gpu_zaxpbypc(dp0,dr0,dn0,beta,-1.0*omega0*beta,nrows,blocks,threads);
    gpu_multxy(dy,ddiag,dp0,nrows,blocks,threads);
    gpu_spmv(dn0,dy,nrows,dA,djA,diA,blocks,threads);
    temp1=gpu_dotxy(dr0h, dn0, aux, dtempv, nrows,(blocks + 1) / 2, threads);
    alpha=rho1/temp1;
    gpu_zaxpby(1.0,dr0,-1.0*alpha,dn0,ds,nrows,blocks,threads);
    gpu_multxy(dz,ddiag,ds,nrows,blocks,threads);
    gpu_spmv(dt,dz,nrows,dA,djA,diA,blocks,threads);
    gpu_multxy(dAx2,ddiag,dt,nrows,blocks,threads);
    temp1=gpu_dotxy(dz, dAx2, aux, dtempv, nrows,(blocks + 1) / 2, threads);
    temp2=gpu_dotxy(dAx2, dAx2, aux, dtempv, nrows,(blocks + 1) / 2, threads);
    omega0= temp1/temp2;
    gpu_axpy(dx,dy,alpha,nrows,blocks,threads);
    gpu_axpy(dx,dz,omega0,nrows,blocks,threads);
    gpu_zaxpby(1.0,ds,-1.0*omega0,dt,dr0,nrows,blocks,threads);
    temp1=gpu_dotxy(dr0, dr0, aux, dtempv, nrows,(blocks + 1) / 2, threads);
    temp1=sqrt(temp1);
    rho0=rho1;
    it++;
  }
}