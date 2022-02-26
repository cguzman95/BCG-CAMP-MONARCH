/* Copyright (C) 2022 Barcelona Supercomputing Center
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include<math.h>
#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
#include <cuda_runtime_api.h>

const int N = 16;
const int blocksize = 16;


static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


__global__
void hello(char *a, int *b)
{
	a[threadIdx.x] += b[threadIdx.x];
}

void hello_test(){

  char a[N] = "Hello \0\0\0\0\0\0";
  int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  char *ad;
  int *bd;
  const int csize = N*sizeof(char);
  const int isize = N*sizeof(int);

  printf("HANDLE_ERROR %s", a);

  HANDLE_ERROR(cudaMalloc( (void**)&ad, csize ));
  //cudaMalloc( (void**)&ad, csize );

  cudaMalloc( (void**)&bd, isize );
  cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice );
  cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice );

  dim3 dimBlock( blocksize, 1 );
  dim3 dimGrid( 1, 1 );
  hello<<<dimGrid, dimBlock>>>(ad, bd);
  cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost );
  cudaFree( ad );
  cudaFree( bd );

  printf("%s\n", a);

}

int nextPowerOfTwo(int v){

  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;

  return v;
}

__device__ void cudaDevicereduce(double *g_idata, double *g_odata, volatile double *sdata, int n_shr_empty)
{
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  __syncthreads();

  sdata[tid] = g_idata[i];

  __syncthreads();
  //first threads update empty positions
  if(tid<n_shr_empty)
    sdata[tid+blockDim.x]=0.;
  __syncthreads();

  for (unsigned int s=(blockDim.x+n_shr_empty)/2; s>0; s>>=1)
  {
    if (tid < s){
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  __syncthreads();
  *g_odata = sdata[0];
  __syncthreads();

}

__device__ void cudaDevicemaxD(double *g_idata, double *g_odata, volatile double *sdata, int n_shr_empty)
{
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  __syncthreads();

  sdata[tid] = g_idata[i];

  __syncthreads();
  //first threads update empty positions
  if(tid<n_shr_empty)
    sdata[tid+blockDim.x]=sdata[tid];
  __syncthreads();

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

__global__
void cudaIterative(double *x, double *y, int n_shr_empty)
{
  extern __shared__ double sdata[];
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  int it = 0;
  int maxIt = 100;
  double a = 0.0;
  while(it<maxIt){

    y[i] = threadIdx.x;
    cudaDevicereduce(y,&a,sdata,n_shr_empty);
    x[i]=a;
    cudaDevicemaxD(x,&a,sdata,n_shr_empty);

    it++;
  }

  __syncthreads();

  //if (i==0) printf("a %lf\n",a);
  y[i] = a;
  __syncthreads();
  //printf("y[i] %lf i %d\n",y[i],i);

}

void iterative_test(){

  int blocks = 100;
  int threads_block = 73;
  int n_shr_memory = nextPowerOfTwo(threads_block);
  int n_shr_empty = n_shr_memory-threads_block;
  int len = blocks*threads_block;

  double *x = (double *) malloc(len * sizeof(double));
  memset(x, 0, len * sizeof(double));
  double *y = (double *) malloc(len * sizeof(double));
  memset(y, 1, len * sizeof(double));

  double *dx,*dy;
  cudaMalloc((void **) &dx, len * sizeof(double));
  cudaMalloc((void **) &dy, len * sizeof(double));

  HANDLE_ERROR(cudaMemcpy(dx, x, len*sizeof(double), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dy, y, len*sizeof(double), cudaMemcpyHostToDevice));

  cudaIterative <<<blocks,threads_block,n_shr_memory*sizeof(double)>>>
                                          (dx,dy,n_shr_empty);

  HANDLE_ERROR(cudaMemcpy( y, dy, len*sizeof(double), cudaMemcpyDeviceToHost ));

  double cond = 0;
  for(int i=0; i<threads_block; i++){
    cond+=i;
  }
  for(int i=0; i<len; i++){
    //printf("y[i] %lf cond %lf i %d\n", y[i],cond,i);
    if (y[i] != cond ){
     printf("ERROR: Wrong result\n");
     printf("y[i] %lf cond %lf i %d\n", y[i],cond,i);
     exit(0);
    }
  }

  printf(" iterative_test SUCCESS\n");
}

int main()
{
  //hello_test();
  iterative_test();

	return 0;
}
