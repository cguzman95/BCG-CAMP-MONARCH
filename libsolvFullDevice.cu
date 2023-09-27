/* Copyright (C) 2021 Barcelona Supercomputing Center and University of
* Illinois at Urbana-Champaign
* SPDX-License-Identifier: MIT
*/

#include "libsolv.h"

__device__ void cudaDeviceSpmvCSR(double* dx, double* db, double* dA, int* djA, int* diA){
 __syncthreads();
 int i = threadIdx.x + blockDim.x*blockIdx.x;
 double sum = 0.0;
 int nnz=diA[blockDim.x];
 for(int j=diA[threadIdx.x]; j<diA[threadIdx.x+1]; j++){
   sum+= db[djA[j]+blockDim.x*blockIdx.x]*dA[j+nnz*blockIdx.x];
 }
 __syncthreads();
 dx[i]=sum;
 __syncthreads();
}

__device__ void cudaDeviceSpmvCSC(double* dx, double* db, double* dA, int* djA, int* diA, int n_shr_empty){
 double mult;
 extern __shared__ double sdata[];
 int i= threadIdx.x + blockDim.x*blockIdx.x;
 unsigned int tid = threadIdx.x;
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

__device__ void cudaDeviceSpmvCSD(double* dx, double* db, double* dA, int* djA, int* diA){
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
 int nnz=1118;
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

__device__ void cudaDeviceSpmvCSRReduce(double* dx, double* db, int nrows, double* dA, int* djA, int* diA){
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
#elif CSC
 cudaDeviceSpmvCSC(dx,db,dA,djA,diA,n_shr_empty);
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

__device__ void cudaDevicedotxy(double *g_idata1, double *g_idata2,
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
#endif
 __syncthreads();
 *g_odata = sdata[0];
 __syncthreads();
}

__device__
   void solveBcgCudaDeviceCVODE(ModelDataGPU *md){
 int i = blockIdx.x * blockDim.x + threadIdx.x;
 double alpha,rho0,omega0,beta,rho1,temp1,temp2;
 alpha=rho0=omega0=beta=rho1=temp1=temp2=1.0;
 md->dn0[i]=0.0;
 md->dp0[i]=0.0;
 cudaDeviceSpmv(md->dr0,md->dx,md->dA,md->djA,md->diA, md->n_shr_empty);
 md->dr0[i]=md->dtempv[i]-md->dr0[i];
 md->dr0h[i]=md->dr0[i];
 int it=0;
 while(it<1000 && temp1>1.0E-30){
   cudaDevicedotxy(md->dr0, md->dr0h, &rho1, md->n_shr_empty);
   beta = (rho1 / rho0) * (alpha / omega0);
   md->dp0[i]=beta*md->dp0[i]+md->dr0[i]-md->dn0[i]*omega0*beta;
   md->dy[i]=md->ddiag[i]*md->dp0[i];
   cudaDeviceSpmv(md->dn0, md->dy, md->dA, md->djA, md->diA, md->n_shr_empty);
   cudaDevicedotxy(md->dr0h, md->dn0, &temp1, md->n_shr_empty);
   alpha = rho1 / temp1;
   md->ds[i]=md->dr0[i]-alpha*md->dn0[i];
   md->dx[i]+=alpha*md->dy[i];
   md->dy[i]=md->ddiag[i]*md->ds[i];
   cudaDeviceSpmv(md->dt, md->dy, md->dA, md->djA, md->diA, md->n_shr_empty);
   md->dr0[i]=md->ddiag[i]*md->dt[i];
   cudaDevicedotxy(md->dy, md->dr0, &temp1, md->n_shr_empty);
   cudaDevicedotxy(md->dr0, md->dr0, &temp2, md->n_shr_empty);
   omega0 = temp1 / temp2;
   md->dx[i]+=omega0*md->dy[i];
   md->dr0[i]=md->ds[i]-omega0*md->dt[i];
   md->dt[i]=0.0;
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
 int len_cell = mGPU->nrows / mGPU->n_cells;
 int threads_block = len_cell;
 int blocks = mGPU->n_cells;
 int n_shr_memory = nextPowerOfTwoBCG(len_cell);
 mGPU->n_shr_empty = n_shr_memory - threads_block;
 cudaGlobalCVode<<< blocks, threads_block,
                   n_shr_memory * sizeof(double)>>>(*mGPU);
}
