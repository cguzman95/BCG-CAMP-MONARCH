
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <helper_cuda.h>  // helper function CUDA error checking and initialization
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples



#ifndef __CUDACC__  
#define __CUDACC__
#endif

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

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
    __syncthreads();
    int d = 1;
    atomicAdd(c, d);//build error
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
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
  int nDevices = 1;
  int n_cells_multiplier = 3;

  ModelDataGPU *mGPUs = (ModelDataGPU *)malloc(nDevices * sizeof(ModelDataGPU));
  ModelDataGPU *mGPU = &mGPUs[0];
  ModelDataGPU mGPU0_object;
  ModelDataGPU *mGPU0 = &mGPU0_object;

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

  /*
  for(int icell=0; icell<mGPU0->n_cells; icell++){
    printf("cell %d:\n",icell);
    for(int i=0; i<mGPU0->nrows/mGPU0->n_cells+1; i++){
      printf("%d ", iA[i+icell*(mGPU0->nrows/mGPU0->n_cells)]);
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
  for(int i=0; i<n_cells_multiplier; i++){
    //memcpy(jA+i*mGPU0->nnz, jA_aux, mGPU0->nnz*sizeof(int));
    memcpy(A+i*mGPU0->nnz, A_aux, mGPU0->nnz*sizeof(double));
    memcpy(diag+i*mGPU0->nrows, diag_aux, mGPU0->nrows*sizeof(double));
    memcpy(x+i*mGPU0->nrows, x_aux, mGPU0->nrows*sizeof(double));
    memcpy(tempv+i*mGPU0->nrows, tempv_aux, mGPU0->nrows*sizeof(double));

    for(int j=1; j<mGPU0->nrows+1; j++) {
      iA[j + i * mGPU0->nrows] = iA_aux[j] +i*mGPU0->nnz;// iA_aux[mGPU0->nrows] * i;
      //printf("%d ",iA[j + i * mGPU0->nrows]);
    }
    //printf("\n");

    for(int j=0; j<mGPU0->nnz+1; j++) {
      jA[j + i * mGPU0->nnz] = jA_aux[j] +i*mGPU0->nrows;// iA_aux[mGPU0->nrows] * i;
      //printf("%d ",iA[j + i * mGPU0->nrows]);
    }

    /*
    for(int j=0; j<mGPU0->nrows; j++) {
      printf("%le ",tempv[j + i * mGPU0->nrows]);
      printf("%le ",diag[j + i * mGPU0->nrows]);
      printf("%le ",x[j + i * mGPU0->nrows]);
    }
    printf("\n");*/

    /*
    for(int j=0; j<mGPU0->nnz; j++) {
      printf("%d ",jA[j + i * mGPU0->nnz]);
    }
    printf("\n");

    for(int j=0; j<mGPU0->nnz; j++) {
      printf("%le ",A[j + i * mGPU0->nnz]);
    }
    printf("\n");
*/
  }

  mGPU0->n_cells=mGPU0->n_cells*n_cells_multiplier;
  mGPU0->nnz=mGPU0->nnz*n_cells_multiplier;
  mGPU0->nrows=mGPU0->nrows*n_cells_multiplier;

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

    //printf("mGPU->nrows%d\n",mGPU->nrows);

    cudaMalloc((void **) &mGPU->djA, mGPU->nnz * sizeof(int));
    cudaMalloc((void **) &mGPU->diA, (mGPU->nrows + 1) * sizeof(int));
    cudaMalloc((void **) &mGPU->dA, mGPU->nnz * sizeof(double));
    cudaMalloc((void **) &mGPU->ddiag, mGPU->nrows * sizeof(double));
    cudaMalloc((void **) &mGPU->dx, mGPU->nrows * sizeof(double));
    cudaMalloc((void **) &mGPU->dtempv, mGPU->nrows * sizeof(double));

    cudaMemcpyAsync(mGPU->djA, jA, mGPU->nnz * sizeof(int), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(mGPU->diA, iA, (mGPU->nrows + 1) * sizeof(int), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(mGPU->dA, A+offset_nnz, mGPU->nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(mGPU->ddiag, diag+offset_nrows, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(mGPU->dx, x+offset_nrows, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, 0);
    HANDLE_ERROR(cudaMemcpyAsync(mGPU->dtempv, tempv+offset_nrows, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, 0));

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
    cudaGetDeviceProperties(&prop, iDevice);
    mGPU->threads = prop.maxThreadsPerBlock;
    mGPU->blocks = (mGPU->nrows + mGPU->threads - 1) / mGPU->threads;

    solveGPU_block(mGPU);

    HANDLE_ERROR(cudaMemcpyAsync(jA, mGPU->djA, mGPU->nnz * sizeof(int), cudaMemcpyDeviceToHost, 0));
    cudaMemcpyAsync(iA, mGPU->diA, (mGPU->nrows + 1) * sizeof(int), cudaMemcpyDeviceToHost, 0);
    cudaMemcpyAsync(A+offset_nnz, mGPU->dA, mGPU->nnz * sizeof(double), cudaMemcpyDeviceToHost, 0);
    cudaMemcpyAsync(diag+offset_nrows, mGPU->ddiag, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost, 0);
    cudaMemcpyAsync(x+offset_nrows, mGPU->dx, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost, 0);
    cudaMemcpyAsync(tempv+offset_nrows, mGPU->dtempv, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost, 0);

    offset_nnz += mGPU->nnz;
    offset_nrows += mGPU->nrows;
  }

  cudaDeviceSynchronize();
/*
  for(int icell=0; icell<mGPU0->n_cells; icell++){
    printf("cell %d:\n",icell);
    for(int i=0; i<mGPU0->nrows/mGPU0->n_cells; i++){
      printf("%le ", x[i+icell*(mGPU0->nrows/mGPU0->n_cells)]);
      //printf("%d %d\n",i, iA[i]);
    }
    printf("\n");
  }*/

  mGPU0->n_cells=mGPU0->n_cells/n_cells_multiplier;
  mGPU0->nnz=mGPU0->nnz/n_cells_multiplier;
  mGPU0->nrows=mGPU0->nrows/n_cells_multiplier;

  double *A2_aux=(double*)malloc(mGPU0->nnz*sizeof(double));
  double *x2_aux=(double*)malloc(mGPU0->nrows*sizeof(double));
  double *tempv2_aux=(double*)malloc(mGPU0->nrows*sizeof(double));

  fp = fopen("outBCG.txt", "r");

  for(int i=0; i<mGPU0->nnz; i++){
    fscanf(fp, "%le", &A2_aux[i]);
    //printf("%d %le\n",i, A[i]);
  }

  for(int i=0; i<mGPU0->nrows; i++){
    fscanf(fp, "%le", &x2_aux[i]);
    //printf("%d %le\n",i, x[i]);
  }

  for(int i=0; i<mGPU0->nrows; i++){
    fscanf(fp, "%le", &tempv2_aux[i]);
    //printf("%d %le\n",i, tempv[i]);
  }

  fclose(fp);

  double *A2=(double*)malloc(mGPU0->nnz*n_cells_multiplier*sizeof(double));
  double *x2=(double*)malloc(mGPU0->nrows*n_cells_multiplier*sizeof(double));
  double *tempv2=(double*)malloc(mGPU0->nrows*n_cells_multiplier*sizeof(double));

  //printf("mGPU0->nrows %d\n",mGPU0->nrows);

  for(int i=0; i<n_cells_multiplier; i++){
    memcpy(A2+i*mGPU0->nnz, A2_aux, mGPU0->nnz*sizeof(double));
    memcpy(x2+i*mGPU0->nrows, x2_aux, mGPU0->nrows*sizeof(double));
    memcpy(tempv2+i*mGPU0->nrows, tempv2_aux, mGPU0->nrows*sizeof(double));

    /*
    for(int j=0; j<mGPU0->nrows; j++) {
      printf("%le ",x2[j + i * mGPU0->nrows]);
    }
    printf("\n");

    for(int j=0; j<mGPU0->nrows; j++) {
      printf("%le ",x[j + i * mGPU0->nrows]);
    }
    printf("\n");
*/

  }

  mGPU0->n_cells=mGPU0->n_cells*n_cells_multiplier;
  mGPU0->nnz=mGPU0->nnz*n_cells_multiplier;
  mGPU0->nrows=mGPU0->nrows*n_cells_multiplier;

  int flag=1;
  if(compare_doubles(A2,A,mGPU0->nnz,"A2")==0) flag=0;
  if(compare_doubles(x2,x,mGPU0->nrows,"x2")==0)  flag=0;
  if(compare_doubles(tempv2,tempv,mGPU0->nrows,"tempv2")==0)  flag=0;

  if(flag==0)
    printf("FAIL\n");
  else
    printf("SUCCESS\n");

}

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
