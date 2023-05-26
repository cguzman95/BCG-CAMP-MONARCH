/* Copyright (C) 2022 Barcelona Supercomputing Center
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <iostream>
#include <cmath>
#include <cstring>

#ifndef SCALE
#define SCALE 1
#endif

#ifndef NUM_DEVICES
#define NUM_DEVICES 1
#endif

#ifndef DEVICE_LIST
#define DEVICE_LIST 0
#endif

int compare_doubles(double *x, double *y, int len, const char *s){

  int flag=1;
  double tol=0.01;
  //float tol=0.0001;
  double rel_error, abs_error;
  int n_fails=0;
  for (int i=0; i<len; i++){
    abs_error=std::abs(x[i]-y[i]);
    if(x[i]==0)
      rel_error=0.;
    else
      rel_error=std::abs((x[i]-y[i])/x[i]);
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

#pragma omp declare target
// x=A*b (dn0, dy, nrows, A, jA, iA)
inline void target_spmvCSR(int row, int pre_nrows, int pre_nnz, double* x, double* b, double* A, int* jA, int* iA) 
{
    double sum = 0.0;
    int m_r = row%pre_nrows; // scale down to pre_nrows, as iA was not scaled up
    int s = row/pre_nrows;
    for(int j=iA[m_r]; j<iA[m_r+1]; j++){
      sum+= b[jA[j]+s*pre_nrows]*A[j];
    }
    x[row]=sum;
}

inline void target_adhoc_sum_reduction(double *reductor, int start, int end)
{
    int red_threads=128;
    while(red_threads>0){
        #pragma omp parallel for
        for(int i=start; i<start+red_threads; i++) {
            if(i+red_threads<end){
                reductor[i]+=reductor[i+red_threads];
            }
        }
        red_threads/=2;
    }
}
#pragma omp end declare target

//Algorithm: Biconjugate gradient
void solveBcg_spread(int blocks, int blocks_per_device, int threads, int csize, int pre_nrows, int pre_nnz, // devs config
        double *A, int *jA, int *iA, double *dx, double *tempv, //Input data
        int nrows, int maxIt, int mattype, int nnz,
        int n_cells, double tolmax, double *diag, //Init variables
        double *dr0, double *dr0h, double *dn0, double *dp0,
        double *dt, double *ds, double *dAx2, double *dy, double *dz, double *reductor)// Auxiliary vectors
{ 
  #pragma omp taskgroup
  {
    auto &&cap_size = [threads, nrows] (int start, int size) {
      return threads*(start+size) >= nrows ? nrows-start*threads : size*threads;
    };
      
    #pragma omp target spread teams distribute \
        nowait \
        devices(DEVICE_LIST) spread_schedule(static, blocks_per_device) num_teams(blocks_per_device) thread_limit(threads) \
        map(    diag[omp_spread_start*threads : cap_size(omp_spread_start,omp_spread_size)], \
                 dr0[omp_spread_start*threads : cap_size(omp_spread_start,omp_spread_size)], \
                dr0h[omp_spread_start*threads : cap_size(omp_spread_start,omp_spread_size)], \
                 dn0[omp_spread_start*threads : cap_size(omp_spread_start,omp_spread_size)], \
                 dp0[omp_spread_start*threads : cap_size(omp_spread_start,omp_spread_size)], \
                  ds[omp_spread_start*threads : cap_size(omp_spread_start,omp_spread_size)], \
                  dt[omp_spread_start*threads : cap_size(omp_spread_start,omp_spread_size)], \
                dAx2[omp_spread_start*threads : cap_size(omp_spread_start,omp_spread_size)], \
               tempv[omp_spread_start*threads : cap_size(omp_spread_start,omp_spread_size)], \
                  dx[omp_spread_start*threads : cap_size(omp_spread_start,omp_spread_size)], \
                  dy[omp_spread_start*threads : cap_size(omp_spread_start,omp_spread_size)], \
                  dz[omp_spread_start*threads : cap_size(omp_spread_start,omp_spread_size)], \
            reductor[omp_spread_start*threads : cap_size(omp_spread_start,omp_spread_size)], \
                  iA[0:pre_nrows+1], \
                  jA[0:pre_nnz], \
                   A[0:pre_nnz])
    for(int j=0; j<blocks; j++) {
      int start=j*threads;
      if(start<nrows){
        int end = start+threads<=nrows ? start+threads : nrows;
        // init to zero
        #pragma omp parallel for
        for(int i=start; i<end; i++) {
            dn0[i]=0; // cudaDevicesetconst(dn0, 0.0, nrows);
            dp0[i]=0; // cudaDevicesetconst(dp0, 0.0, nrows);
        }
        
        // spare matrix vector product
        #pragma omp parallel for
        for (int i=start; i<end; i++){
            target_spmvCSR(i,pre_nrows,pre_nnz,dr0,dx,A,jA,iA); // cudaDeviceSpmvCSR(dr0,x,nrows,A,jA,iA); //y=A*x
        }


        // axpby
        #pragma omp parallel for
        for (int i=start; i<end; i++){
            dr0[i]= (1.0*tempv[i]) + (-1.0*dr0[i]); // cudaDeviceaxpby(dr0,tempv,1.0,-1.0,nrows); // y= a*x+ b*y
            dr0h[i] = dr0[i]; // cudaDeviceyequalsx(dr0h,dr0,nrows);
        }
        
        double alpha,rho0,omega0,beta,rho1,temp1,temp2;
        alpha=rho0=omega0=beta=rho1=temp1=temp2=1.0;
        
        int it=0;
        do
        {
          // cudaDevicedotxy(dr0, dr0h, &rho1, nrows, n_shr_empty);
          #pragma omp parallel for //reduction(+: rho1[j])
          for(int i=start; i<end; i++) {
              reductor[i]=dr0[i]*dr0h[i]; 
          }
          target_adhoc_sum_reduction(reductor, start, end);
          rho1 = reductor[start];
      
          beta = (rho1 / rho0) * (alpha / omega0);
          
          //if(j==1) printf("alpha=%f, beta=%f, omega0=%f, rho0=%f, rho1=%f\n", alpha, beta, omega0, rho0, rho1);
      
          #pragma omp parallel for
          for (int i=start; i<end; i++){
            dp0[i] = beta*dp0[i] + dr0[i] + (-1.0*omega0)*beta*dn0[i]; // cudaDevicezaxpbypc(dp0, dr0, dn0, beta, -1.0 * omega0 * beta, nrows); 
            dy[i]=diag[i]*dp0[i]; // cudaDevicemultxy(dy, diag, dp0, nrows);
            dn0[i] = 0.0; // cudaDevicesetconst(dn0, 0.0, nrows);
          }
          
          // spare matrix vector multiplication
          #pragma omp parallel for
          for (int i=start; i<end; i++){
              target_spmvCSR(i,pre_nrows,pre_nnz,dn0, dy, A, jA, iA); // cudaDeviceSpmvCSR(dn0, dy, nrows, A, jA, iA);
          }
          
          /// cudaDevicedotxy(dr0h, dn0, &temp1, nrows, n_shr_empty);
          #pragma omp parallel for //reduction(+: rho1[j])
          for(int i=start; i<end; i++) {
              reductor[i]=dr0h[i]*dn0[i]; 
          }
          target_adhoc_sum_reduction(reductor, start, end);
          temp1 = reductor[start];
          
          alpha = rho1 / temp1;
          
          //if(j==1) printf("alpha=%f, rho1=%f, temp1=%f\n", alpha, rho1, temp1);

          #pragma omp parallel for
          for (int i=start; i<end; i++){
              ds[i]= 1.0*dr0[i] + (-1.0*alpha) * dn0[i]; // cudaDevicezaxpby(1.0, dr0, -1.0 * alpha[j], dn0, ds, nrows); // // z= a*x + b*y
              dz[i]=diag[i]*ds[i]; // cudaDevicemultxy(dz, diag, ds, nrows); // precond z=diag*s 
          }
          
          #pragma omp parallel for
          for (int i=start; i<end; i++){
              target_spmvCSR(i,pre_nrows,pre_nnz, dt, dz, A, jA, iA); // cudaDeviceSpmvCSR(dt, dz, nrows, A, jA, iA);
          }
          
          #pragma omp parallel for
          for (int i=start; i<end; i++){
              dAx2[i]=diag[i]*dt[i]; // cudaDevicemultxy(dAx2, diag, dt, nrows);
          }

          // cudaDevicedotxy(dz, dAx2, &temp1, nrows, n_shr_empty);
          #pragma omp parallel for //reduction(+: rho1[j])
          for(int i=start; i<end; i++) {
              reductor[i]=dz[i]*dAx2[i]; 
          }
          target_adhoc_sum_reduction(reductor, start, end);
          temp1 = reductor[start];
      
          // cudaDevicedotxy(dAx2, dAx2, &temp2, nrows, n_shr_empty);
          #pragma omp parallel for //reduction(+: rho1[j])
          for(int i=start; i<end; i++) {
              reductor[i]=dAx2[i]*dAx2[i]; 
          }
          target_adhoc_sum_reduction(reductor, start, end);
          temp2 = reductor[start];
          
          omega0 = temp1/temp2;
          
          //if(j==1) printf("omega0=%f, temp1=%f, temp2=%f\n", omega0, temp1, temp2);
          
          #pragma omp parallel for
          for (int i=start; i<end; i++){
              dx[i]= alpha*dy[i] + dx[i]; // cudaDeviceaxpy(x, dy, alpha, nrows); // x=alpha*y +x
              dx[i]= omega0*dz[i] + dx[i]; // cudaDeviceaxpy(x, dz, omega0, nrows);
          }
      
          #pragma omp parallel for
          for (int i=start; i<end; i++){
              dr0[i]=1.0*ds[i] + (-1.0*omega0) * dt[i]; // cudaDevicezaxpby(1.0, ds, -1.0 * omega0, dt, dr0, nrows); // z= a*x + b*y
              dt[i] = 0.0; // cudaDevicesetconst(dt, 0.0, nrows);
          }
          
          // cudaDevicedotxy(dr0, dr0, &temp1, nrows, n_shr_empty);
          #pragma omp parallel for //reduction(+: rho1[j])
          for(int i=start; i<end; i++) {
              reductor[i]=dr0[i]*dr0[i]; 
          }
          target_adhoc_sum_reduction(reductor, start, end);
          temp1 = reductor[start];
          
          temp1 = sqrtf(temp1);
          rho0 = rho1;
          
          //if(j==1) printf("temp1=%f, rho0=%f, rho1=%f, tolmax=%f, it=%d, maxIt=%d\n", temp1, rho0, rho1, tolmax, it, maxIt);
          
          it++;
        }while(it<maxIt && temp1>tolmax);
      }
    }
  }
}

void BCG (){
  FILE *fp;
  fp = fopen("confBCG.txt", "r");
  if (fp == NULL) {
    printf("File not found \n");
    exit(EXIT_FAILURE);
  }

  int pre_n_cells=0;
  int pre_nrows=0;
  int pre_nnz=0;
  int maxIt=0;
  int mattype=0;
  double tolmax=0.0;
  
  fscanf(fp, "%d",  &pre_n_cells);
  fscanf(fp, "%d",  &pre_nrows);
  fscanf(fp, "%d",  &pre_nnz);
  fscanf(fp, "%d",  &maxIt);
  fscanf(fp, "%d",  &mattype);
  fscanf(fp, "%le", &tolmax);
  
  int *jA=(int*)malloc(pre_nnz*sizeof(int));
  double *A=(double*)malloc(pre_nnz*sizeof(double));
  int *iA=(int*)malloc((pre_nrows+1)*sizeof(int));
  
  double *diag=(double*)malloc(SCALE*pre_nrows*sizeof(double));
  double *tempv=(double*)malloc(SCALE*pre_nrows*sizeof(double));

  double *dr0=(double*)malloc(SCALE*pre_nrows*sizeof(double));;
  double *dr0h=(double*)malloc(SCALE*pre_nrows*sizeof(double));
  double *dn0=(double*)malloc(SCALE*pre_nrows*sizeof(double));
  double *dp0=(double*)malloc(SCALE*pre_nrows*sizeof(double));
  double *dt=(double*)malloc(SCALE*pre_nrows*sizeof(double));
  double *ds=(double*)malloc(SCALE*pre_nrows*sizeof(double));
  double *dAx2=(double*)malloc(SCALE*pre_nrows*sizeof(double));
  
  double *dx=(double*)malloc(SCALE*pre_nrows*sizeof(double));
  double *dy=(double*)malloc(SCALE*pre_nrows*sizeof(double));
  double *dz=(double*)malloc(SCALE*pre_nrows*sizeof(double));
  
  double *reductor=(double*)malloc(SCALE*pre_nrows*sizeof(double));
  
  for(int i=0; i<pre_nnz; i++){
    fscanf(fp, "%d", &jA[i]);
  }

  for(int i=0; i<pre_nrows+1; i++){
    fscanf(fp, "%d", &iA[i]);
  }

  for(int i=0; i<pre_nnz; i++){
    fscanf(fp, "%le", &A[i]);
  }

  for(int i=0; i<pre_nrows; i++){
    fscanf(fp, "%le", &diag[i]);
  }

  for(int i=0; i<pre_nrows; i++){
    fscanf(fp, "%le", &dx[i]);
  }

  for(int i=0; i<pre_nrows; i++){
    fscanf(fp, "%le", &tempv[i]);
  }
  
  fclose(fp);
  
  for(int s=1;s<SCALE;s++){
    memcpy(diag+(s*pre_nrows),diag,pre_nrows*sizeof(double));
    memcpy(dx+(s*pre_nrows),dx,pre_nrows*sizeof(double));
    memcpy(tempv+(s*pre_nrows),tempv,pre_nrows*sizeof(double));
  }
  
  {
    int n_cells=pre_n_cells*SCALE;
    int nrows=pre_nrows*SCALE;
    int nnz=pre_nnz*SCALE;
    
    int threads=154;
    int blocks=pre_n_cells*SCALE; // 2.500.000
    int blocks_per_device=blocks/NUM_DEVICES;    // 725.000
    int csize=blocks_per_device*threads; // 725.000 * 154
    int offset_cells=0;
    int len_cell=nrows/n_cells; // 154
    
    #pragma omp parallel
    #pragma omp single
    {
      #pragma omp taskgroup
      {
        #pragma omp target enter data spread \
                nowait \
                devices(DEVICE_LIST) \
                range(0:nrows) \
                chunk_size(csize) \
                map(to:       iA[0:pre_nrows+1], \
                              jA[0:pre_nnz], \
                               A[0:pre_nnz], \
                            diag[omp_spread_start:omp_spread_size], \
                           tempv[omp_spread_start:omp_spread_size], \
                              dx[omp_spread_start:omp_spread_size]) \
                map(alloc:   dr0[omp_spread_start:omp_spread_size], \
                            dr0h[omp_spread_start:omp_spread_size], \
                             dn0[omp_spread_start:omp_spread_size], \
                             dp0[omp_spread_start:omp_spread_size], \
                              dt[omp_spread_start:omp_spread_size], \
                              ds[omp_spread_start:omp_spread_size], \
                            dAx2[omp_spread_start:omp_spread_size], \
                              dy[omp_spread_start:omp_spread_size], \
                              dz[omp_spread_start:omp_spread_size], \
                        reductor[omp_spread_start:omp_spread_size])
      }
      
      printf("solveGPU_block_thr n_cells %d len_cell %d nrows %d nnz %d blocks %d threads_block %d n_shr_empty %d offset_cells %d\n",
                  n_cells,len_cell,nrows,nnz,blocks,threads, blocks*threads-nrows,offset_cells);
      
      solveBcg_spread(blocks, blocks_per_device, threads, csize, pre_nrows, pre_nnz, A, jA, iA, dx, tempv, nrows, maxIt, mattype, nnz, n_cells, tolmax, diag, dr0, dr0h, dn0, dp0, dt, ds, dAx2, dy, dz, reductor);

      #pragma omp taskgroup
      {
        #pragma omp target exit data spread \
                nowait \
                devices(DEVICE_LIST) \
                range(0:nrows) \
                chunk_size(csize) \
                map(from:      dx[omp_spread_start:omp_spread_size]) \
                map(release:   iA[0:pre_nrows+1], \
                               jA[0:pre_nnz], \
                                A[0:pre_nnz], \
                             diag[omp_spread_start:omp_spread_size], \
                            tempv[omp_spread_start:omp_spread_size], \
                              dr0[omp_spread_start:omp_spread_size], \
                             dr0h[omp_spread_start:omp_spread_size], \
                              dn0[omp_spread_start:omp_spread_size], \
                              dp0[omp_spread_start:omp_spread_size], \
                               dt[omp_spread_start:omp_spread_size], \
                               ds[omp_spread_start:omp_spread_size], \
                             dAx2[omp_spread_start:omp_spread_size], \
                               dy[omp_spread_start:omp_spread_size], \
                               dz[omp_spread_start:omp_spread_size], \
                         reductor[omp_spread_start:omp_spread_size])                       
      }
    }
  }
  
  double *A2=(double*)malloc(pre_nnz*sizeof(double));
  double *x2=(double*)malloc(pre_nrows*sizeof(double));
  double *tempv2=(double*)malloc(pre_nrows*sizeof(double));

  fp = fopen("outBCG.txt", "r");

  
  for(int i=0; i<pre_nnz; i++){
    fscanf(fp, "%le", &A2[i]);
  }

  for(int i=0; i<pre_nrows; i++){
    fscanf(fp, "%le", &x2[i]);
  }

  for(int i=0; i<pre_nrows; i++){
    fscanf(fp, "%le", &tempv2[i]);
  }

  fclose(fp);

  int s;
  for(s=0;s<SCALE;s++)
  {
    if(compare_doubles(x2,dx+(s*pre_nrows),pre_nrows,"x2")==0){
      break;
    }
  }

  if(s<SCALE)
    printf("FAIL at section %d with size %d\n", s, pre_nrows*SCALE);
  else
    printf("SUCCESS with size %d\n", pre_nrows*SCALE);
  
  free(tempv2);
  free(x2);
  free(A2);
  
  free(jA);
  free(A);
  free(iA);
  
  free(diag);
  free(tempv);

  free(dr0);
  free(dr0h);
  free(dn0);
  free(dp0);
  free(dt);
  free(ds);
  free(dAx2);
  
  free(dx);
  free(dy);
  free(dz);
  free(reductor);
}

int main()
{
  BCG();

 return 0;
}
