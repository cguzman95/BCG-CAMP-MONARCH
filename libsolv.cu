/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
 * Illinois at Urbana-Champaign
 * SPDX-License-Identifier: MIT
 */

#include "libsolv.h"

#ifndef DEV_OLD_FUNCTIONS


// Device functions (equivalent to global functions but in device to allow calls from gpu)
__device__ void cudaDevicematScaleAddI(int nrows, double* dA, int* djA, int* diA, double alpha)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
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

// Diagonal precond
__device__ void cudaDevicediagprecond(int nrows, double* dA, int* djA, int* diA, double* ddiag)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
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

// y = constant
__device__ void cudaDevicesetconst(double* dy,double constant,int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  dy[row]=constant;
}


// y= a*x+ b*y
__device__ void cudaDeviceaxpby(double* dy,double* dx, double a, double b, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  dy[row]= a*dx[row] + b*dy[row];
}

// y = x
__device__ void cudaDeviceyequalsx(double* dy,double* dx,int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  dy[row]=dx[row];
}

__device__ void cudaDevicemin(double *g_odata, double in, volatile double *sdata, int n_shr_empty)
{
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  __syncthreads();

  sdata[tid] = in;

  __syncthreads();
  //first threads update empty positions
  if(tid<n_shr_empty)
    sdata[tid+blockDim.x]=sdata[tid];
  __syncthreads(); //Not needed (should)

  for (unsigned int s=(blockDim.x+n_shr_empty)/2; s>0; s>>=1)
  {
    if (tid < s){
      if(sdata[tid + s] < sdata[tid] ) sdata[tid]=sdata[tid + s];
    }
    __syncthreads();
  }

  __syncthreads();
  *g_odata = sdata[0];
  __syncthreads();

}

__device__ void cudaDevicemaxI(int *g_odata, int in, volatile double *sdata, int n_shr_empty)
{
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  __syncthreads();

  sdata[tid] = in;

  __syncthreads();
  //first threads update empty positions
  if(tid<n_shr_empty)
    sdata[tid+blockDim.x]=sdata[tid];
  __syncthreads(); //Not needed (should)

  for (unsigned int s=(blockDim.x+n_shr_empty)/2; s>0; s>>=1)
  {
    if (tid < s){
      if(sdata[tid + s] > sdata[tid] ) sdata[tid]=sdata[tid + s];
    }
    __syncthreads();
  }

  __syncthreads();
  *g_odata = sdata[0];
  __syncthreads();

}

__device__ void cudaDeviceaddI(int *g_odata, int in, volatile double *sdata, int n_shr_empty)
{
  //extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  __syncthreads();

  sdata[tid] = in;

  __syncthreads();

  //first threads update empty positions
  if(tid<n_shr_empty)
    sdata[tid+blockDim.x]=sdata[tid];

  __syncthreads(); //Not needed (should)

  //if(blockIdx.x==0)printf("i %d in %le sdata[tid] %le\n",i,in,sdata[tid]);

  for (unsigned int s=(blockDim.x+n_shr_empty)/2; s>0; s>>=1)
  {
    if (tid < s){//&& sdata[tid + s]!=0.
      //if(sdata[tid + s] < sdata[tid] ) sdata[tid]=sdata[tid + s];
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  __syncthreads();
  *g_odata = sdata[0];
  __syncthreads();

}

__device__ void warpReduce(volatile double *sdata, unsigned int tid) {
  unsigned int blockSize = blockDim.x;
  if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}


__device__ void cudaDevicedotxy(double *g_idata1, double *g_idata2,
                                double *g_odata, int n_shr_empty)
{
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  __syncthreads();

  //Needed, when testing be careful with SRAM data remanesce https://stackoverflow.com/questions/22172881/why-does-my-kernels-shared-memory-seems-to-be-initialized-to-zero

  //first threads update empty positions
  if(tid<n_shr_empty)
    sdata[tid+blockDim.x]=0.;

  __syncthreads();
  sdata[tid] = g_idata1[i]*g_idata2[i];
  __syncthreads();

  /*
    for (unsigned int s=(blockDim.x+n_shr_empty)/2; s>0; s>>=1)
    {
      if (tid < s)
        sdata[tid] += sdata[tid + s];
      __syncthreads();
    }
    */

  //todo treat case deriv_length < 32
  //maybe https://github.com/cudpp/cudpp/blob/master/src/cudpp/kernel/reduce_kernel.cuh


  unsigned int blockSize = blockDim.x+n_shr_empty;

  // do reduction in shared mem
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

  if (tid < 32) warpReduce(sdata, tid);

  __syncthreads();//not needed?

  *g_odata = sdata[0];
  __syncthreads();


}

// z= a*z + x + b*y
__device__ void cudaDevicezaxpbypc(double* dz, double* dx,double* dy, double a, double b, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  dz[row]=a*dz[row]  + dx[row] + b*dy[row];
}

// z= x*y
__device__ void cudaDevicemultxy(double* dz, double* dx,double* dy, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  dz[row]=dx[row]*dy[row];
}

// z= a*x + b*y
__device__ void cudaDevicezaxpby(double a, double* dx, double b, double* dy, double* dz, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  dz[row]=a*dx[row] + b*dy[row];
}

// y= a*x + y
__device__ void cudaDeviceaxpy(double* dy,double* dx, double a, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  dy[row]=a*dx[row] + dy[row];
}

// sqrt(sum ( (x_i*y_i)^2)/n)
__device__ void cudaDeviceVWRMS_Norm(double *g_idata1, double *g_idata2, double *g_odata, int n, int n_shr_empty)
{
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  //unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  __syncthreads();

  //first threads update empty positions
  if(tid<n_shr_empty)
    sdata[tid+blockDim.x]=0.;

  __syncthreads(); //Not needed (should)

                    /*
                      double mySum = (i < n) ? g_idata1[i]*g_idata1[i]*g_idata2[i]*g_idata2[i] : 0;
                      if (i + blockDim.x < n)
                        mySum += g_idata1[i+blockDim.x]*g_idata1[i+blockDim.x]*g_idata2[i+blockDim.x]*g_idata2[i+blockDim.x];
                    */

  __syncthreads();
  sdata[tid] = g_idata1[i]*g_idata1[i]*g_idata2[i]*g_idata2[i];
  __syncthreads();

  for (unsigned int s=(blockDim.x+n_shr_empty)/2; s>0; s>>=1)
  {
    if (tid < s)
      sdata[tid] += sdata[tid + s];

    __syncthreads();
  }

  //if (tid == 0) g_odata[blockIdx.x] = sdata[0];
  g_odata[0] = sqrt(sdata[0]/n);
  //*g_odata = sqrt(sdata[0]/n);
  __syncthreads();
}

// y=alpha*y
__device__ void cudaDevicescaley(double* dy, double a, int nrows)
{
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  dy[row]=a*dy[row];
}



#endif

__device__ void cudaDeviceSpmvCSR(double* dx, double* db, double* dA, int* djA, int* diA)
{
  __syncthreads();
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  double sum = 0.0;
  int nnz=diA[blockDim.x];
  for(int j=diA[threadIdx.x]; j<diA[threadIdx.x+1]; j++){
    sum+= db[djA[j]+blockDim.x*blockIdx.x]*dA[j+nnz*blockIdx.x];
  }
  __syncthreads();
  dx[row]=sum;
  __syncthreads();
}

__device__ void cudaDeviceSpmvCSCAtomic(double* dx, double* db, double* dA, int* djA, int* diA, int n_shr_empty)
{
  double mult;
  extern __shared__ double sdata[];
  int row= threadIdx.x + blockDim.x*blockIdx.x;
  __syncthreads();
  dx[row]=0.0;
  __syncthreads();
  int nnz=diA[blockDim.x];
  for(int j=diA[threadIdx.x]; j<diA[threadIdx.x+1]; j++)
  {
#ifdef DEBUG_CUDADEVICESPMVCSC_BLOCK
    if(dA[j] != dA[j])
      printf("NAN dA[j]");
    if(djA[j] != djA[j])
      printf("NAN djA[j]]");
#endif
    mult = db[row]*dA[j+nnz*blockIdx.x];
    atomicAdd_block(&(dx[djA[j]+blockDim.x*blockIdx.x]),mult);
  }
  __syncthreads();
}

__device__ void cudaDeviceSpmvCSD(double* dx, double* db, double* dA, int* djA, int* diA, int n_shr_empty)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  __syncthreads();
  dx[tid]=0.0;
  __syncthreads();
  int nnz=1118;
  for(int iDiag=0; iDiag<blockDim.x; iDiag++) {
    if(threadIdx.x<diA[iDiag+1]-diA[iDiag]) {
      int dAi=diA[iDiag] + threadIdx.x  + nnz * blockIdx.x;
      int dbi=djA[diA[iDiag] + threadIdx.x] + blockDim.x*blockIdx.x;
      int dxi=((iDiag+djA[diA[iDiag] + threadIdx.x])%blockDim.x) + blockDim.x*blockIdx.x;
      dx[dxi] += db[dbi] * dA[dAi];
    }
    __syncthreads();
  }

}

__device__ void cudaDeviceSpmvBoolDet(double* dx, double* db, double* dA, int* diA)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  __syncthreads();
  dx[tid]=0.0;
  __syncthreads();
}

__device__ void cudaDeviceSpmvCUID(double* dx, double* db, double* dA, int* djA)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  __syncthreads();
  dx[tid]=0.0;
  __syncthreads();
  int nnz=1118;//todo
  int iRow=threadIdx.x;
  __syncthreads();
  for(int row=0; row<blockDim.x; row++) {
    if (djA[threadIdx.x + row * blockDim.x] >= 0) {
      dx[iRow + blockDim.x * blockIdx.x] +=
          db[tid] * dA[djA[threadIdx.x + row * blockDim.x] + nnz * blockIdx.x];
#ifdef DEBUG_CUID
      printf("dx db dA djA %lf %lf %d\n", dx[iRow], db[tid],dA[djA[tid + row * blockDim.x]], djA[tid + row * blockDim.x]);
#endif
    }
    iRow++;
    if (iRow >= blockDim.x) {
      iRow = 0;
    }
    __syncthreads();
  }
}

__device__ void cudaDeviceSpmvCSRVector(double* dx, double* db, double* dA, int* djA, int* diA, int n_shr_empty)
{
  int t = threadIdx.x;
  int warpSize = 32;
  int lane = t & (warpSize-1);
  int warpsPerBlock = blockDim.x / warpSize;
  int row = (blockIdx.x * warpsPerBlock) + (t / warpSize);
  extern __shared__ double vals[];
  unsigned int tid = threadIdx.x;
  if(tid<n_shr_empty)
    vals[tid+blockDim.x]=0.;
  int rowStart = diA[row];
  int rowEnd = diA[row+1];
  double sum = 0.;
  for (int j = rowStart + lane; j < rowEnd; j += warpSize)
  {
    int col = djA[j];
    sum += dA[j] * db[col];
  }
  vals[t] = sum;
  __syncthreads();
  if (lane < 16) vals[t] += vals[t + 16];
  if (lane <  8) vals[t] += vals[t + 8];
  if (lane <  4) vals[t] += vals[t + 4];
  if (lane <  2) vals[t] += vals[t + 2];
  if (lane <  1) vals[t] += vals[t + 1];
  __syncthreads();
  if (lane == 0)
  {
    dx[row] = vals[t];
  }
}

#ifdef CSR_ADAPTIVE

__device__ int devicenextPowerOfTwo(int v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

__device__ void cudaDevicedotxyCSRReduce(double *g_idata, double *g_idata2,
                                         double *g_odata, int n, int n_shr_empty, int n_shr_len)
{
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  __syncthreads();
  if(tid<n_shr_empty)
    sdata[tid+blockDim.x]=0.;

  __syncthreads();
  sdata[tid] = g_idata[i]*g_idata2[i];
  __syncthreads();

  for (unsigned int s=(n_shr_len)/2; s>0; s>>=1)
  {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  *g_odata = sdata[0];
  __syncthreads();
}

__device__ void cudaDeviceSpmvCSRReduce(double* dx, double* db, int nrows, double* dA, int* djA, int* diA)
{
  __syncthreads();
  int row = threadIdx.x + blockDim.x*blockIdx.x;
  double sum = 0.0;
  int nnz=diA[blockDim.x];
  int n_iters = nnz / blockDim.x; //todo /2?
  for(int i=0; i<n_iters; i++)
  {
    int offsetdA=diA[threadIdx.x+1]-diA[threadIdx.x];
    int n_shr_len=devicenextPowerOfTwo(offsetdA);
    int n_shr_empty=n_shr_len-(offsetdA);
    int j=row;
    dx[row] = db[djA[j]+blockDim.x*blockIdx.x]*dA[j+nnz*blockIdx.x];
    int idx=threadIdx.x/offsetdA;
    cudaDevicedotxyCSRReduce(&db[djA[j]+blockDim.x*blockIdx.x],
                             &dA[j+nnz*blockIdx.x],&dx[idx],n_shr_empty,n_shr_len);
  }
  __syncthreads();
  int residual=nnz-(blockDim.x*n_iters);
  if(threadIdx.x < residual){
  }
}

#endif

__device__ void cudaDeviceSpmv(double* dx, double* db, double* dA, int* djA, int* diA, int n_shr_empty)
{
#ifdef CSR
  cudaDeviceSpmvCSR(dx,db,dA,djA,diA);
#elif CSC_ATOMIC
  cudaDeviceSpmvCSCAtomic(dx,db,dA,djA,diA);
#elif CSD
  cudaDeviceSpmvCSD(dx,db,dA,djA,diA);
#elif CBD
  cudaDeviceSpmvBoolDet(dx,db,dA,djA);
#elif CUID
  cudaDeviceSpmvCUID(dx,db,dA,djA);
#elif CSR_VECTOR
  cudaDeviceSpmvCSRVector(dx,db,dA,djA,diA);
#elif CSR_ADAPTIVE
  cudaDeviceSpmvCSRReduce(dx,db,dA,djA,diA);
#else
  cudaDeviceSpmvCSR(dx,db,dA,djA,diA);
#endif

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

__device__ void cudaDevicedotxy_2(double *g_idata1, double *g_idata2,
                                  double *g_odata, int n_shr_empty){
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
      sdata[0]+=g_idata1[j+blockIdx.x*blockDim.x]*g_idata2[j+blockIdx.x*blockDim.x];
    }
  }
#else
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<n_shr_empty)
    sdata[tid+blockDim.x]=0.;
  __syncthreads();
  sdata[tid] = g_idata1[i]*g_idata2[i];
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
    //print_double(sdata,1,"sdata");
#endif
  __syncthreads();
  *g_odata = sdata[0];
  __syncthreads();
}

//Algorithm: Biconjugate gradient
__device__
    void solveBcgCuda(double* dA, int* djA, int* diA, double* dx, double* dtempv //Input data
        , int nrows, int blocks, int n_shr_empty, int maxIt, int mattype
        , int n_cells, double tolmax, double* ddiag //Init variables
        , double* dr0, double* dr0h, double* dn0, double* dp0
        , double* dt, double* ds, double* dAx2, double* dy, double* dz// Auxiliary vectors
    )
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int tid = threadIdx.x;
  int active_threads = nrows;

  //if(i<1){
  if (i < active_threads) {
#if defined(CSR_SHARED) || defined(CSR_SHARED_DB_JAC)
    extern __shared__ double sdata[];
    int nnz=diA[blockDim.x];
    for(int j=diA[threadIdx.x]; j<diA[threadIdx.x+1]; j++){
      sdata[j]=dA[j+nnz*blockIdx.x];
    }__syncthreads();
#endif
    double alpha, rho0, omega0, beta, rho1, temp1, temp2;
    alpha = rho0 = omega0 = beta = rho1 = temp1 = temp2 = 1.0;
    cudaDevicesetconst(dn0, 0.0, nrows);
    cudaDevicesetconst(dp0, 0.0, nrows);
    cudaDeviceSpmv(dr0, dx, dA, djA, diA, n_shr_empty); //y=A*x
    cudaDeviceaxpby(dr0, dtempv, 1.0, -1.0, nrows);
    __syncthreads();
    cudaDeviceyequalsx(dr0h, dr0, nrows);
    int it = 0;
    do
    {
      __syncthreads();
      cudaDevicedotxy(dr0, dr0h, &rho1, n_shr_empty);
#if defined(CSR_SHARED) || defined(CSR_SHARED_DB_JAC)
      for(int j=diA[threadIdx.x]; j<diA[threadIdx.x+1]; j++){
        sdata[j]=dA[j+nnz*blockIdx.x];
      }__syncthreads();
#endif
      __syncthreads();
      beta = (rho1 / rho0) * (alpha / omega0);
      __syncthreads();
      cudaDevicezaxpbypc(dp0, dr0, dn0, beta, -1.0 * omega0 * beta, nrows);   //z = ax + by + c
      __syncthreads();
      cudaDevicemultxy(dy, ddiag, dp0, nrows);
      __syncthreads();
      cudaDevicesetconst(dn0, 0.0, nrows);
      cudaDeviceSpmv(dn0, dy, dA, djA, diA,n_shr_empty);
      cudaDevicedotxy(dr0h, dn0, &temp1, n_shr_empty);
#if defined(CSR_SHARED) || defined(CSR_SHARED_DB_JAC)
      for(int j=diA[threadIdx.x]; j<diA[threadIdx.x+1]; j++){
        sdata[j]=dA[j+nnz*blockIdx.x];
      }__syncthreads();
#endif
      __syncthreads();
      alpha = rho1 / temp1;
      cudaDevicezaxpby(1.0, dr0, -1.0 * alpha, dn0, ds, nrows);
      __syncthreads();
      cudaDevicemultxy(dz, ddiag, ds, nrows); // precond z=diag*s
      cudaDeviceSpmv(dt, dz, dA, djA, diA, n_shr_empty);
      __syncthreads();
      cudaDevicemultxy(dAx2, ddiag, dt, nrows);
      __syncthreads();
      cudaDevicedotxy(dz, dAx2, &temp1, n_shr_empty);
      __syncthreads();
      cudaDevicedotxy(dAx2, dAx2, &temp2, n_shr_empty);
      __syncthreads();
      omega0 = temp1 / temp2;
      cudaDeviceaxpy(dx, dy, alpha, nrows); // x=alpha*y +x
      __syncthreads();
      cudaDeviceaxpy(dx, dz, omega0, nrows);
      __syncthreads();
      cudaDevicezaxpby(1.0, ds, -1.0 * omega0, dt, dr0, nrows);
      cudaDevicesetconst(dt, 0.0, nrows);
      __syncthreads();
      cudaDevicedotxy(dr0, dr0, &temp1, n_shr_empty);
      temp1 = sqrtf(temp1);
      rho0 = rho1;
      __syncthreads();
      it++;
    } while (it<maxIt && temp1>tolmax);//while(it<maxIt && temp1>tolmax);
  }
}

__global__
void cudaGlobalCVode(ModelDataGPU md_object,double* dA, int* djA, int* diA, double* dx, double* dtempv //Input data
                    , int nrows, int blocks, int n_shr_empty, int maxIt, int mattype
                    , int n_cells, double tolmax, double* ddiag //Init variables
                    , double* dr0, double* dr0h, double* dn0, double* dp0
                    , double* dt, double* ds, double* dAx2, double* dy, double* dz) {
  ModelDataGPU *md = &md_object;
  solveBcgCuda(dA, djA, diA, dx, dtempv, nrows, blocks, n_shr_empty, maxIt, mattype, n_cells,
tolmax, ddiag, dr0, dr0h, dn0, dp0, dt, ds, dAx2, dy, dz);
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

void solveGPU_block(ModelDataGPU* mGPU)
{
  //Init variables ("public")
  int nrows = mGPU->nrows;
  int nnz = mGPU->nnz;
  int n_cells = mGPU->n_cells;
  int maxIt = 1000;
  double tolmax = 1.0e-30;
  int mattype = 1;

  // Auxiliary vectors ("private")
  double* dr0 = mGPU->dr0;
  double* dr0h = mGPU->dr0h;
  double* dn0 = mGPU->dn0;
  double* dp0 = mGPU->dp0;
  double* dt = mGPU->dt;
  double* ds = mGPU->ds;
  double* dy = mGPU->dy;
  double* dz = mGPU->dz;
  double* dAx2 = mGPU->dAx2;

  //Input variables
  int* djA = mGPU->djA;
  int* diA = mGPU->diA;
  double* dA = mGPU->dA;
  double* ddiag = mGPU->ddiag;
  double* dx = mGPU->dx;
  double* dtempv = mGPU->dtempv;



  int len_cell = mGPU->nrows / mGPU->n_cells;
  int threads_block = len_cell;
  int blocks = mGPU->n_cells;
  int n_shr_memory = nextPowerOfTwoBCG(len_cell);
  int n_shr_empty = mGPU->n_shr_empty = n_shr_memory - threads_block;
  cudaGlobalCVode<<< blocks, threads_block,
    n_shr_memory * sizeof(double)>>>
      (*mGPU,dA, djA, diA, dx, dtempv, nrows, blocks, n_shr_empty, maxIt, mattype, n_cells,
       tolmax, ddiag, dr0, dr0h, dn0, dp0, dt, ds, dAx2, dy, dz);
}
