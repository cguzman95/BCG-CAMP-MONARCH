/* Copyright (C) 2022 Barcelona Supercomputing Center
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "libsolv.h"
#include "cuda_structs.h"

#ifndef SCALE
#define SCALE 1
#endif

#ifndef NUM_DEVICES
#define NUM_DEVICES 1
#endif

#define CAMP_DEBUG_GPU

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

int compare_doubles(double *x, double *y, int len, const char *s){

  int flag=1;
  double tol=0.0001;
  double rel_error, abs_error;
  int n_fails=0;
  for (int i=0; i<len; i++){
    abs_error=abs(x[i]-y[i]);
    if(x[i]==0)
      rel_error=0.;
    else
      rel_error=abs((x[i]-y[i])/x[i]);
    if((rel_error>tol && abs_error > 1.0E-30) || y[i]!=y[i]){
    //if(true){
      printf("compare_doubles %s rel_error %le abs_error %le for tol %le at [%d]: %le vs %le\n",
             s,rel_error,abs_error,tol,i,x[i],y[i]);
      flag=0;
      n_fails++;
      if(n_fails==4)
        return flag;
    }
  }

  return flag;

}

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
    if(threadIdx.x==0) x[i]=a;
    cudaDevicemaxD(x,&a,sdata,n_shr_empty);

    it++;
  }

  __syncthreads();

  //if (i==0) printf("a %le\n",a);
  y[i] = a;
  __syncthreads();
  //printf("y[i] %le i %d\n",y[i],i);

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
    //printf("y[i] %le cond %le i %d\n", y[i],cond,i);
    if (y[i] != cond ){
     printf("ERROR: Wrong result\n");
     printf("y[i] %le cond %le i %d\n", y[i],cond,i);
     exit(0);
    }
  }

  printf(" iterative_test SUCCESS\n");
}

__device__
void dvcheck_input_gpud(double *x, int len, const char* s)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //if(i<2)
  if(i<len)
  {
    printf("%s[%d]=%-le\n",s,i,x[i]);
  }
}

//Algorithm: Biconjugate gradient
__global__
void solveBcgCuda(
        double *dA, int *djA, int *diA, double *dx, double *dtempv //Input data
        ,int nrows, int blocks, int n_shr_empty, int maxIt, int mattype
        ,int n_cells, double tolmax, double *ddiag //Init variables
        ,double *dr0, double *dr0h, double *dn0, double *dp0
        ,double *dt, double *ds, double *dAx2, double *dy, double *dz// Auxiliary vectors
#ifdef CAMP_DEBUG_GPU
        ,int *it_pointer, int last_blockN
#endif
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    int active_threads = nrows;

    //if(tid==0)printf("blockDim.x %d\n",blockDim.x);


    //if(i<1){
    if(i<active_threads){

        double alpha,rho0,omega0,beta,rho1,temp1,temp2;
        alpha=rho0=omega0=beta=rho1=temp1=temp2=1.0;

        /*alpha  = 1.0;
        rho0   = 1.0;
        omega0 = 1.0;*/

        //gpu_yequalsconst(dn0,0.0,nrows,blocks,threads);  //n0=0.0 //memset???
        //gpu_yequalsconst(dp0,0.0,nrows,blocks,threads);  //p0=0.0
        cudaDevicesetconst(dn0, 0.0, nrows);
        cudaDevicesetconst(dp0, 0.0, nrows);

        //Not needed
        /*
        cudaDevicesetconst(dr0h, 0.0, nrows);
        cudaDevicesetconst(dt, 0.0, nrows);
        cudaDevicesetconst(ds, 0.0, nrows);
        cudaDevicesetconst(dAx2, 0.0, nrows);
        cudaDevicesetconst(dy, 0.0, nrows);
        cudaDevicesetconst(dz, 0.0, nrows);
         */

#ifndef CSR_SPMV_CPU
        cudaDeviceSpmvCSR(dr0,dx,nrows,dA,djA,diA); //y=A*x
#else
        cudaDeviceSpmvCSC_block(dr0,dx,nrows,dA,djA,diA,n_shr_empty)); //y=A*x
#endif

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

        //printf("%d ddiag %-le\n",i,ddiag[i]);
    //printf("%d dr0 %-le\n",i, dr0[i]);

#endif

        //gpu_axpby(dr0,dtempv,1.0,-1.0,nrows,blocks,threads); // r0=1.0*rhs+-1.0r0 //y=ax+by
        cudaDeviceaxpby(dr0,dtempv,1.0,-1.0,nrows);

        __syncthreads();
        //gpu_yequalsx(dr0h,dr0,nrows,blocks,threads);  //r0h=r0
        cudaDeviceyequalsx(dr0h,dr0,nrows);

#ifdef CAMP_DEBUG_GPU
        int it=*it_pointer;
#endif

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

        if(i==0){
      //printf("%d dr0[%d] %-le\n",it,i,dr0[i]);
      printf("%d %d rho1 %-le\n",it,i,rho1);
    }

    //dvcheck_input_gpud(dx,nrows,"dx");
    //dvcheck_input_gpud(dr0,nrows,"dr0");

#endif

        do
        {
            //rho1=gpu_dotxy(dr0, dr0h, aux, daux, nrows,(blocks + 1) / 2, threads);
            __syncthreads();

            cudaDevicedotxy(dr0, dr0h, &rho1, nrows, n_shr_empty);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

            if(i==0){
      //printf("%d dr0[%d] %-le\n",it,i,dr0[i]);
      printf("%d %d rho1 rho0 %-le %-le\n",it,i,rho1,rho0);
    }
    if(isnan(rho1) || rho1==0.0){
      dvcheck_input_gpud(dx,nrows,"dx");
      dvcheck_input_gpud(dr0h,nrows,"dr0h");
      dvcheck_input_gpud(dr0,nrows,"dr0");
    }

#endif

            __syncthreads();
            beta = (rho1 / rho0) * (alpha / omega0);

            __syncthreads();
            //gpu_zaxpbypc(dp0,dr0,dn0,beta,-1.0*omega0*beta,nrows,blocks,threads);   //z = ax + by + c
            cudaDevicezaxpbypc(dp0, dr0, dn0, beta, -1.0 * omega0 * beta, nrows);   //z = ax + by + c

            __syncthreads();
            //gpu_multxy(dy,ddiag,dp0,nrows,blocks,threads);  // precond y= p0*diag
            cudaDevicemultxy(dy, ddiag, dp0, nrows);

            __syncthreads();
            cudaDevicesetconst(dn0, 0.0, nrows);
            //gpu_spmv(dn0,dy,nrows,dA,djA,diA,mattype,blocks,threads);  // n0= A*y
#ifndef CSR_SPMV_CPU
            cudaDeviceSpmvCSR(dn0, dy, nrows, dA, djA, diA);
#else
            cudaDeviceSpmvCSC_block(dn0, dy, nrows, dA, djA, diA,n_shr_empty);
#endif

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

            if(it==0){
        printf("%d %d dy dn0 ddiag %-le %-le %le\n",it,i,dy[i],dn0[i],ddiag[i]);
        //printf("%d %d dn0 %-le\n",it,i,dn0[i]);
        //printf("%d %d &temp1 %p\n",it,i,&temp1);
        //printf("%d %d &test %p\n",it,i,&test);
        //printf("%d %d &i %p\n",it,i,&i);
      }

#endif

            //temp1=gpu_dotxy(dr0h, dn0, aux, daux, nrows,(blocks + 1) / 2, threads);
            cudaDevicedotxy(dr0h, dn0, &temp1, nrows, n_shr_empty);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

            if(i==0){
        printf("%d %d temp1 %-le\n",it,i,temp1);
        //printf("%d %d &temp1 %p\n",it,i,&temp1);
        //printf("%d %d &test %p\n",it,i,&test);
        //printf("%d %d &i %p\n",it,i,&i);
      }

#endif

            __syncthreads();
            alpha = rho1 / temp1;

            //gpu_zaxpby(1.0,dr0,-1.0*alpha,dn0,ds,nrows,blocks,threads); // a*x + b*y = z
            cudaDevicezaxpby(1.0, dr0, -1.0 * alpha, dn0, ds, nrows);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

            if(i==0){
        printf("%d ds[%d] %-le\n",it,i,ds[i]);
      }

#endif

            __syncthreads();
            //gpu_multxy(dz,ddiag,ds,nrows,blocks,threads); // precond z=diag*s
            cudaDevicemultxy(dz, ddiag, ds, nrows); // precond z=diag*s

            //gpu_spmv(dt,dz,nrows,dA,djA,diA,mattype,blocks,threads);
#ifndef CSR_SPMV_CPU
            cudaDeviceSpmvCSR(dt, dz, nrows, dA, djA, diA);
#else
            cudaDeviceSpmvCSC_block(dt, dz, nrows, dA, djA, diA,n_shr_empty);
#endif

            __syncthreads();
            //gpu_multxy(dAx2,ddiag,dt,nrows,blocks,threads);
            cudaDevicemultxy(dAx2, ddiag, dt, nrows);

            __syncthreads();
            //temp1=gpu_dotxy(dz, dAx2, aux, daux, nrows,(blocks + 1) / 2, threads);
            cudaDevicedotxy(dz, dAx2, &temp1, nrows, n_shr_empty);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

        //if(i>=0){
        //printf("%d ddiag[%d] %-le\n",it,i,ddiag[i]);
        //printf("%d dt[%d] %-le\n",it,i,dt[i]);
        //printf("%d dAx2[%d] %-le\n",it,i,dAx2[i]);
        //printf("%d dz[%d] %-le\n",it,i,dz[i]);
      //}

      if(i==0){
        printf("%d %d temp1 %-le\n",it,i,temp1);
      }

#endif

            __syncthreads();
            //temp2=gpu_dotxy(dAx2, dAx2, aux, daux, nrows,(blocks + 1) / 2, threads);
            cudaDevicedotxy(dAx2, dAx2, &temp2, nrows, n_shr_empty);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP

      if(i==0){
        printf("%d %d temp2 %-le\n",it,i,temp2);
      }

#endif

            __syncthreads();
            omega0 = temp1 / temp2;
            //gpu_axpy(dx,dy,alpha,nrows,blocks,threads); // x=alpha*y +x
            cudaDeviceaxpy(dx, dy, alpha, nrows); // x=alpha*y +x

            __syncthreads();
            //gpu_axpy(dx,dz,omega0,nrows,blocks,threads);
            cudaDeviceaxpy(dx, dz, omega0, nrows);

            __syncthreads();
            //gpu_zaxpby(1.0,ds,-1.0*omega0,dt,dr0,nrows,blocks,threads);
            cudaDevicezaxpby(1.0, ds, -1.0 * omega0, dt, dr0, nrows);
            cudaDevicesetconst(dt, 0.0, nrows);

            __syncthreads();
            //temp1=gpu_dotxy(dr0, dr0, aux, daux, nrows,(blocks + 1) / 2, threads);
            cudaDevicedotxy(dr0, dr0, &temp1, nrows, n_shr_empty);

            //temp1 = sqrt(temp1);
            temp1 = sqrtf(temp1);

            rho0 = rho1;
            /**/
            __syncthreads();
            /**/

            //if (tid==0) it++;
            it++;
        } while(it<maxIt+*it_pointer && temp1>tolmax);//while(it<maxIt && temp1>tolmax);

#ifdef DEBUG_SOLVEBCGCUDA_DEEP
        if(i==0)
      printf("%d %d %-le %-le\n",tid,it,temp1,tolmax);
#endif

        //if(it>=maxIt-1)
        //  dvcheck_input_gpud(dr0,nrows,999);
        //dvcheck_input_gpud(dr0,nrows,k++);
/*

#ifdef CAMP_DEBUG_GPU
    if(tid==0) {
      if(last_blockN==1){
        if(*it_pointer-it>*it_pointer)*it_pointer = it;
    }
    else{
      *it_pointer = it;
      }
    //printf("it_pointer %d\n",*it_pointer);
    }
#endif
*/

    }

}

void solveGPU_block_thr(int blocks, int threads_block, int n_shr_memory, int n_shr_empty, int offset_cells,
                        ModelDataGPU *mGPU, int last_blockN)
{

    //Init variables ("public")
    int nrows = mGPU->nrows;
    int nnz = mGPU->nnz;
    int n_cells = mGPU->n_cells;
    int maxIt = mGPU->maxIt;
    int mattype = mGPU->mattype;
    double tolmax = mGPU->tolmax;

    // Auxiliary vectors ("private")
    double *dr0 = mGPU->dr0;
    double *dr0h = mGPU->dr0h;
    double *dn0 = mGPU->dn0;
    double *dp0 = mGPU->dp0;
    double *dt = mGPU->dt;
    double *ds = mGPU->ds;
    double *dAx2 = mGPU->dAx2;
    double *dy = mGPU->dy;
    double *dz = mGPU->dz;

    int offset_nrows=(nrows/n_cells)*offset_cells;
    int offset_nnz=(nnz/n_cells)*offset_cells;
    int len_cell=nrows/n_cells;

  //Input variables
  int *djA=mGPU->djA;
  int *diA=mGPU->diA;
  double *dA=mGPU->dA+offset_nnz;
  double *ddiag=mGPU->ddiag+offset_nrows;
  double *dx=mGPU->dx+offset_nrows;
  double *dtempv=mGPU->dtempv+offset_nrows;
  

#ifdef DEBUG_SOLVEBCGCUDA
        printf("solveGPU_block_thr n_cells %d len_cell %d nrows %d nnz %d max_threads_block %d blocks %d threads_block %d n_shr_empty %d offset_cells %d\n",
               mGPU->n_cells,len_cell,mGPU->nrows,mGPU->nnz,n_shr_memory,blocks,threads_block,n_shr_empty,offset_cells);
#endif

    int it = 0;
    solveBcgCuda << < blocks, threads_block, n_shr_memory * sizeof(double) >> >
    //solveBcgCuda << < blocks, threads_block, threads_block * sizeof(double) >> >
    (dA, djA, diA, dx, dtempv, nrows, blocks, n_shr_empty, maxIt, mattype, n_cells,
            tolmax, ddiag, dr0, dr0h, dn0, dp0, dt, ds, dAx2, dy, dz
#ifdef CAMP_DEBUG_GPU
            , &it, last_blockN
#endif
    );


}

//solveGPU_block: Each block will compute only a cell/group of cells
//Algorithm: Biconjugate gradient
// dx: Input and output RHS
// dtempv: Input preconditioner RHS
void solveGPU_block(ModelDataGPU *mGPU)
{

#ifdef DEBUG_SOLVEBCGCUDA
    if(bicg->counterBiConjGrad==0) {
    printf("solveGPUBlock\n");
  }
#endif

    int len_cell = mGPU->nrows/mGPU->n_cells;
    int max_threads_block=nextPowerOfTwo(len_cell);
#ifdef IS_BLOCKCELLSN
    if(bicg->cells_method==BLOCKCELLSN) {
        max_threads_block = mGPU->threads;//1024;
    }else if(bicg->cells_method==BLOCKCELLSNHALF){
        max_threads_block = mGPU->threads/2;
    }
#endif

    int n_cells_block =  max_threads_block/len_cell;
    int threads_block = n_cells_block*len_cell;
    int blocks = (mGPU->nrows+threads_block-1)/threads_block;
    int n_shr_empty = max_threads_block-threads_block;

    int offset_cells=0;
    int last_blockN=0;

#ifdef IS_BLOCKCELLSN
    //Common kernel (Launch all blocks except the last)
    if(bicg->cells_method==BLOCKCELLSN ||
       bicg->cells_method==BLOCKCELLSNHALF
            ) {

        blocks=blocks-1;

        if(blocks!=0){
            solveGPU_block_thr(blocks, threads_block, max_threads_block, n_shr_empty, offset_cells,
                               mGPU, last_blockN);
            last_blockN = 1;
        }
#ifdef DEBUG_SOLVEBCGCUDA
        else{
      if(bicg->counterBiConjGrad==0){
        printf("solveGPU_block blocks==0\n");
      }
    }
#endif

        //Update vars to launch last kernel
        offset_cells=n_cells_block*blocks;
        int n_cells_last_block=mGPU->n_cells-offset_cells;
        threads_block=n_cells_last_block*len_cell;
        max_threads_block=nextPowerOfTwo(threads_block);
        n_shr_empty = max_threads_block-threads_block;
        blocks=1;

    }
#endif

    solveGPU_block_thr(blocks, threads_block, max_threads_block, n_shr_empty, offset_cells,
                       mGPU, last_blockN);

}

void BCG (){

  //ModelDataGPU mGPU_object;
  //ModelDataGPU *mGPU = &mGPU_object;

  int device = 0;
  int nDevices = 4;
  int n_cells_multiplier = 100;

  ModelDataGPU *mGPUs = (ModelDataGPU *)malloc(NUM_DEVICES * sizeof(ModelDataGPU));
  ModelDataGPU *mGPU = &mGPUs[0];
  ModelDataGPU original_object;
  ModelDataGPU *original_data = &original_object;

  FILE *fp;
  fp = fopen("confBCG.txt", "r");
  if (fp == NULL) {
    printf("File not found \n");
    exit(EXIT_FAILURE);
  }

  fscanf(fp, "%d", &mGPU0->n_cells);
  fscanf(fp, "%d", &mGPU0->nrows);
  fscanf(fp, "%d", &mGPU0->nnz);
  fscanf(fp, "%d", &mGPU0->maxIt);
  fscanf(fp, "%d", &mGPU0->mattype);
  fscanf(fp, "%le", &mGPU0->tolmax);

  //mGPU = mGPU0;

  //ModelDataGPU *mGPU2 = &mGPUs[0];
  //printf("mGPU->nnz %d mGPUs[0]->nnz %d\n",mGPU->nnz,mGPU2->nnz);

  int *jA_aux=(int*)malloc(mGPU0->nnz*sizeof(int));
  int *iA_aux=(int*)malloc((mGPU0->nrows+1)*sizeof(int));
  double *A_aux=(double*)malloc(mGPU0->nnz*sizeof(double));
  double *diag_aux=(double*)malloc(mGPU0->nrows*sizeof(double));
  double *x_aux=(double*)malloc(mGPU0->nrows*sizeof(double));
  double *tempv_aux=(double*)malloc(mGPU0->nrows*sizeof(double));

  for(int i=0; i<mGPU0->nnz; i++){
    fscanf(fp, "%d", &jA_aux[i]);
    //printf("%d %d\n",i, jA_aux[i]);
  }

  for(int i=0; i<mGPU0->nrows+1; i++){
    fscanf(fp, "%d", &iA_aux[i]);
    //printf("%d %d\n",i, iA[i]);
  }

  for(int i=0; i<mGPU0->nnz; i++){
    fscanf(fp, "%le", &A_aux[i]);
    //printf("%d %le\n",i, A[i]);
  }

  for(int i=0; i<mGPU0->nrows; i++){
    fscanf(fp, "%le", &diag_aux[i]);
    //printf("%d %le\n",i, diag[i]);
  }

  for(int i=0; i<mGPU0->nrows; i++){
    fscanf(fp, "%le", &x_aux[i]);
    //printf("%d %le\n",i, x[i]);
  }

  for(int i=0; i<mGPU0->nrows; i++){
    fscanf(fp, "%le", &tempv_aux[i]);
    //printf("%d %le\n",i, tempv[i]);
  }

  fclose(fp);

  for(int s=1;s<SCALE;s++){
    memcpy(diag+(s*original_data->nrows),diag,original_data->nrows*sizeof(double));
    memcpy(x+(s*original_data->nrows),x,original_data->nrows*sizeof(double));
    memcpy(tempv+(s*original_data->nrows),tempv,original_data->nrows*sizeof(double));
  }
  /*
  for(int icell=0; icell<mGPU->n_cells; icell++){
    printf("cell %d:\n",icell);
    for(int i=0; i<mGPU->nrows/original_data->n_cells+1; i++){
      printf("%d ", iA[i+icell*(mGPU->nrows/original_data->n_cells)]);
      //printf("%d %d\n",i, iA[i]);
    }
    printf("\n");
  }
*/

  int *jA=(int*)malloc(mGPU0->nnz*n_cells_multiplier*sizeof(int));
  int *iA=(int*)malloc((mGPU0->nrows*n_cells_multiplier+1)*sizeof(int));
  double *A=(double*)malloc(mGPU0->nnz*n_cells_multiplier*sizeof(double));
  double *diag=(double*)malloc(mGPU0->nrows*n_cells_multiplier*sizeof(double));
  double *x=(double*)malloc(mGPU0->nrows*n_cells_multiplier*sizeof(double));
  double *tempv=(double*)malloc(mGPU0->nrows*n_cells_multiplier*sizeof(double));

  iA[0]=0;
  //printf("iA_aux[mGPU->nrows] %d mGPU->nrows %d\n",iA_aux[mGPU->nrows],mGPU->nrows);
  for(int i=0; i<n_cells_multiplier; i++){
    memcpy(jA+i*mGPU0->nnz, jA_aux, mGPU0->nnz*sizeof(int));
    memcpy(A+i*mGPU0->nnz, A_aux, mGPU0->nnz*sizeof(double));
    memcpy(diag+i*mGPU0->nrows, diag_aux, mGPU0->nrows*sizeof(double));
    memcpy(x+i*mGPU0->nrows, x_aux, mGPU0->nrows*sizeof(double));
    memcpy(tempv+i*mGPU0->nrows, tempv_aux, mGPU0->nrows*sizeof(double));

    for(int j=1; j<mGPU0->nrows+1; j++) {
      iA[j + i * mGPU0->nrows] = iA_aux[j] + iA_aux[mGPU0->nrows] * i;
      //printf("%d ",iA[j + i * mGPU->nrows]);
    }
    /*
    for(int j=0; j<mGPU->nrows; j++) {
      printf("%le ",tempv[j + i * mGPU->nrows]);
    }
    printf("\n");
     */
  }

  //mGPU->n_cells=mGPU->n_cells*n_cells_multiplier;
  //mGPU->nnz=mGPU->nnz*n_cells_multiplier;
  //mGPU->nrows=mGPU->nrows*n_cells_multiplier;

  mGPU0->n_cells=mGPU->n_cells*n_cells_multiplier;
  mGPU0->nnz=mGPU->nnz*n_cells_multiplier;
  mGPU0->nrows=mGPU->nrows*n_cells_multiplier;

  int offset_nnz = 0;
  int offset_nrows = 0;
  int remainder = mGPU0->n_cells % nDevices;
  for (int iDevice = 0; iDevice < nDevices; iDevice++) {
    cudaSetDevice(iDevice);
    mGPU = &mGPUs[iDevice];

    int n_cells = int(mGPU0->n_cells / nDevices);
    if (remainder!=0 && iDevice==0){
      //printf("REMAINDER  nDevicesMODn_cells!=0\n");
      //printf("remainder %d n_cells_total %d nDevices %d n_cells %d\n",remainder,mGPU0->n_cells,nDevices,n_cells);
      n_cells+=remainder;
    }

    mGPU->n_cells=n_cells;
    mGPU->nrows=mGPU0->nrows/mGPU0->n_cells*mGPU->n_cells;
    mGPU->nnz=mGPU0->nnz/mGPU0->n_cells*mGPU->n_cells;
    mGPU->maxIt=mGPU0->maxIt;
    mGPU->mattype=mGPU0->mattype;
    mGPU->tolmax=mGPU0->tolmax;

    cudaMalloc((void **) &mGPU->djA, mGPU->nnz * sizeof(int));
    cudaMalloc((void **) &mGPU->diA, (mGPU->nrows + 1) * sizeof(int));
    cudaMalloc((void **) &mGPU->dA, mGPU->nnz * sizeof(double));
    cudaMalloc((void **) &mGPU->ddiag, mGPU->nrows * sizeof(double));
    cudaMalloc((void **) &mGPU->dx, mGPU->nrows * sizeof(double));
    cudaMalloc((void **) &mGPU->dtempv, mGPU->nrows * sizeof(double));

    cudaMemcpyAsync(mGPU->djA, jA, original_data->nnz * sizeof(int), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(mGPU->diA, iA, (original_data->nrows + 1) * sizeof(int), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(mGPU->dA, A, original_data->nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(mGPU->ddiag, diag+offset_nrows, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(mGPU->dx, x+offset_nrows, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(mGPU->dtempv, tempv+offset_nrows, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, 0);

    //Auxiliary vectors ("private")
    double **dr0 = &mGPU->dr0;
    double **dr0h = &mGPU->dr0h;
    double **dn0 = &mGPU->dn0;
    double **dp0 = &mGPU->dp0;
    double **dt = &mGPU->dt;
    double **ds = &mGPU->ds;
    double **dAx2 = &mGPU->dAx2;
    double **dy = &mGPU->dy;
    double **dz = &mGPU->dz;
    double **daux = &mGPU->daux;

    int nrows = mGPU->nrows;
    cudaMalloc(dr0, nrows * sizeof(double));
    cudaMalloc(dr0h, nrows * sizeof(double));
    cudaMalloc(dn0, nrows * sizeof(double));
    cudaMalloc(dp0, nrows * sizeof(double));
    cudaMalloc(dt, nrows * sizeof(double));
    cudaMalloc(ds, nrows * sizeof(double));
    cudaMalloc(dAx2, nrows * sizeof(double));
    cudaMalloc(dy, nrows * sizeof(double));
    cudaMalloc(dz, nrows * sizeof(double));
    cudaMalloc(daux, nrows * sizeof(double));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    mGPU->threads = prop.maxThreadsPerBlock;
    mGPU->blocks = (mGPU->nrows + mGPU->threads - 1) / mGPU->threads;

    solveGPU_block(mGPU);

    //HANDLE_ERROR(cudaMemcpyAsync(jA, mGPU->djA, original_data->nnz * sizeof(int), cudaMemcpyDeviceToHost, 0));
    //cudaMemcpyAsync(iA, mGPU->diA, (original_data->nrows + 1) * sizeof(int), cudaMemcpyDeviceToHost, 0);
    //cudaMemcpyAsync(A, mGPU->dA, original_data->nnz * sizeof(double), cudaMemcpyDeviceToHost, 0);
    cudaMemcpyAsync(diag+offset_nrows, mGPU->ddiag, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost, 0);
    cudaMemcpyAsync(x+offset_nrows, mGPU->dx, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost, 0);
    cudaMemcpyAsync(tempv+offset_nrows, mGPU->dtempv, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost, 0);

    //offset_nnz += mGPU->nnz;
    offset_nrows += mGPU->nrows;
  }
  

  double *A2=(double*)malloc(original_data->nnz*sizeof(double));
  double *x2=(double*)malloc(original_data->nrows*sizeof(double));
  double *tempv2=(double*)malloc(original_data->nrows*sizeof(double));

  fp = fopen("outBCG.txt", "r");

  for(int i=0; i<original_data->nnz; i++){
    fscanf(fp, "%le", &A2[i]);
    //printf("%d %le\n",i, A[i]);
  }

  for(int i=0; i<original_data->nrows; i++){
    fscanf(fp, "%le", &x2[i]);
    //printf("%d %le\n",i, x[i]);
  }

  for(int i=0; i<original_data->nrows; i++){
    fscanf(fp, "%le", &tempv2[i]);
    //printf("%d %le\n",i, tempv[i]);
  }

  fclose(fp);

  int flag=1;
  int s;
  for(s=0;s<SCALE;s++)
  {
    if(compare_doubles(A2,A,original_data->nnz,"A2")==0) flag=0;
    if(compare_doubles(x2,x+(s*original_data->nrows),original_data->nrows,"x2")==0)  flag=0;
    if(compare_doubles(tempv2,tempv+(s*original_data->nrows),original_data->nrows,"tempv2")==0)  flag=0;
    if(flag==0)
      break;
  }

  if(flag==0)
    printf("FAIL\n");
  else
    printf("SUCCESS\n");
}

int main()
{
  hello_test();
  //iterative_test();
  BCG();

	return 0;
}
