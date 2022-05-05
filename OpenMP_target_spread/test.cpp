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
inline void target_spmvCSR(int row, int pre_nrows, double* x, double* b, double* A, int* jA, int* iA) 
{
    double sum = 0.0;
    int ind=row%pre_nrows;
    for(int j=iA[ind]; j<iA[ind+1]; j++)
    {
      sum+= b[jA[j]]*A[j];
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
void solveBcg_spread(int blocks, int blocks_per_device, int threads, int num_devices, int csize, int pre_nrows, // devs config
        double *A, int *jA, int *iA, double *x, double *tempv, //Input data
        int nrows, int maxIt, int mattype, int nnz,
        int n_cells, double tolmax, double *diag, //Init variables
        double *dr0, double *dr0h, double *dn0, double *dp0,
        double *dt, double *ds, double *dAx2, double *dy, double *dz)// Auxiliary vectors
{ 
    int active[blocks];
    double alpha[blocks],rho0[blocks],omega0[blocks],beta[blocks],rho1[blocks],temp1[blocks],temp2[blocks],reductor[nrows];
    
    for(int j=0;j<blocks;j++){
        alpha[j]=rho0[j]=omega0[j]=beta[j]=rho1[j]=temp1[j]=temp2[j]=1.0;
        active[j]=1;
    }
    
    int helper[2];
    
    #pragma omp target enter data spread \
        nowait \
        devices(0,1) \
        range(0:blocks) \
        chunk_size(blocks_per_device) \
        map(to:  alpha[omp_spread_start:omp_spread_size], \
                    rho0[omp_spread_start:omp_spread_size], \
                omega0[omp_spread_start:omp_spread_size], \
                    beta[omp_spread_start:omp_spread_size], \
                    rho1[omp_spread_start:omp_spread_size], \
                    temp1[omp_spread_start:omp_spread_size], \
                    temp2[omp_spread_start:omp_spread_size], \
                   active[omp_spread_start:omp_spread_size], \
                reductor[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)]) \
        depend(out: active[omp_spread_start:omp_spread_size])

    #pragma omp target spread teams distribute \
        nowait \
        devices(0,1) spread_schedule(static, blocks_per_device) num_teams(blocks_per_device) thread_limit(threads) \
        map(   iA[0:pre_nrows+1], \
            tempv[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)], \
                dr0[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)], \
                dr0h[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)], \
                dn0[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)], \
                dp0[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)], \
                x[0:nrows], \
                jA[0:nnz], \
                A[0:nnz], \
                active[omp_spread_start:omp_spread_size]) \
        depend(in: active[omp_spread_start:omp_spread_size]) \
        depend(out: dn0[omp_spread_start*threads:omp_spread_size], \
                    dp0[omp_spread_start*threads:omp_spread_size], \
                    dr0[omp_spread_start*threads:omp_spread_size], \
                    dr0h[omp_spread_start*threads:omp_spread_size])
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
                target_spmvCSR(i,pre_nrows,dr0,x,A,jA,iA); // cudaDeviceSpmvCSR(dr0,x,nrows,A,jA,iA); //y=A*x
            }


            // axpby
            #pragma omp parallel for
            for (int i=start; i<end; i++){
                dr0[i]= (1.0*tempv[i]) + (-1.0*dr0[i]); // cudaDeviceaxpby(dr0,tempv,1.0,-1.0,nrows); // y= a*x+ b*y
                dr0h[i] = dr0[i]; // cudaDeviceyequalsx(dr0h,dr0,nrows);
            }
        }
    }
    
    int it=0;
    bool keep_working=false;
    int first_active=0;
    do
    {
        #pragma omp target spread teams distribute \
            nowait \
            devices(0,1) spread_schedule(static, blocks_per_device) num_teams(blocks_per_device) thread_limit(threads) \
            map(diag[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)], \
                dr0[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)], \
                dr0h[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)], \
                dn0[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)], \
                dp0[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)], \
                dy[0:nrows], \
                alpha[omp_spread_start:omp_spread_size], \
                beta[omp_spread_start:omp_spread_size], \
                rho0[omp_spread_start:omp_spread_size], \
                rho1[omp_spread_start:omp_spread_size], \
                omega0[omp_spread_start:omp_spread_size], \
                active[omp_spread_start:omp_spread_size], \
            reductor[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)]) \
            depend(in:  dn0[omp_spread_start*threads:omp_spread_size], \
                        dp0[omp_spread_start*threads:omp_spread_size], \
                        dr0[omp_spread_start*threads:omp_spread_size], \
                        dr0h[omp_spread_start*threads:omp_spread_size]) \
            depend(out: dy[omp_spread_start:omp_spread_size])
        for(int j=0; j<blocks; j++) {
            int start=j*threads;
            if(start<nrows && active[j]==1){
                int end = start+threads<=nrows ? start+threads : nrows;
                
                // cudaDevicedotxy(dr0, dr0h, &rho1, nrows, n_shr_empty);
                #pragma omp parallel for //reduction(+: rho1[j])
                for(int i=start; i<end; i++) {
                    reductor[i]=dr0[i]*dr0h[i]; 
                }
                target_adhoc_sum_reduction(reductor, start, end);
                rho1[j] = reductor[start];
            
                beta[j] = (rho1[j] / rho0[j]) * (alpha[j] / omega0[j]);
            
                #pragma omp parallel for
                for (int i=start; i<end; i++){// dz[row]=a*dz[row]  + dx[row] + b*dy[row];
                    dp0[i] = beta[j]*dp0[i] + dr0[i] + (-1.0*omega0[j])*beta[j]*dn0[i]; // cudaDevicezaxpbypc(dp0, dr0, dn0, beta, -1.0 * omega0 * beta, nrows); 
                    dy[i]=diag[i]*dp0[i]; // cudaDevicemultxy(dy, diag, dp0, nrows);
                    dn0[i] = 0.0; // cudaDevicesetconst(dn0, 0.0, nrows);
                }
            }
        }
        
        #pragma omp taskgroup
        {
            /** spread the teams among the devices **/
            #pragma omp target update spread \
                nowait \
                devices(0,1) \
                range(0:nrows) \
                chunk_size(csize) \
                from(dy[omp_spread_start:omp_spread_size]) \
                depend(in: dy[omp_spread_start:omp_spread_size])
        } // sync all devices
        
        #pragma omp target update spread \
            nowait \
            devices(0,1) \
            range(0:nrows) \
            chunk_size(csize) \
            to(helper[omp_spread_start-omp_spread_start:omp_spread_size-omp_spread_size+2],dy[0:nrows]) \
            depend(out: active[omp_spread_start:omp_spread_size])
    
        #pragma omp target spread teams distribute \
            nowait \
            devices(0,1) spread_schedule(static, blocks_per_device) num_teams(blocks_per_device) thread_limit(threads) \
            map(   iA[0:pre_nrows+1], \
                 diag[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)], \
                  dr0[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)], \
                 dr0h[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)], \
                  dn0[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)], \
                   ds[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)], \
                   dy[0:nrows], \
                   dz[0:nrows], \
                   jA[0:nnz], \
                    A[0:nnz], \
                    temp1[omp_spread_start:omp_spread_size], \
                    alpha[omp_spread_start:omp_spread_size], \
                     rho1[omp_spread_start:omp_spread_size],\
                     reductor[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)]) \
          depend(in: active[omp_spread_start:omp_spread_size]) \
          depend(out: dz[omp_spread_start:omp_spread_size])
        for(int j=0; j<blocks; j++) {
            int start=j*threads;
            if(start<nrows && active[j]==1){
                int end = start+threads<=nrows ? start+threads : nrows;
                // spare matrix vector multiplication
                #pragma omp parallel for
                for (int i=start; i<end; i++){
                    target_spmvCSR(i,pre_nrows,dn0, dy, A, jA, iA); // cudaDeviceSpmvCSR(dn0, dy, nrows, A, jA, iA);
                }
                
                /// cudaDevicedotxy(dr0h, dn0, &temp1, nrows, n_shr_empty);
                #pragma omp parallel for //reduction(+: rho1[j])
                for(int i=start; i<end; i++) {
                    reductor[i]=dr0h[i]*dn0[i]; 
                }
                target_adhoc_sum_reduction(reductor, start, end);
                temp1[j] = reductor[start];
                
                alpha[j] = rho1[j] / temp1[j];

                #pragma omp parallel for
                for (int i=start; i<end; i++){
                    ds[i]= 1.0*dr0[i] + (-1.0*alpha[j]) * dn0[i]; // cudaDevicezaxpby(1.0, dr0, -1.0 * alpha[j], dn0, ds, nrows); // // z= a*x + b*y
                    dz[i]=diag[i]*ds[i]; // cudaDevicemultxy(dz, diag, ds, nrows); // precond z=diag*s 
                }
            }
        }
        
        #pragma omp taskgroup
        {
            #pragma omp target update spread \
                nowait \
                devices(0,1) \
                range(0:nrows) \
                chunk_size(csize) \
                from(dz[omp_spread_start:omp_spread_size]) \
                depend(in: dz[omp_spread_start:omp_spread_size])
        } // sync all devices
        
        #pragma omp target update spread \
            nowait \
            devices(0,1) \
            range(0:nrows) \
            chunk_size(csize) \
            to(helper[omp_spread_start-omp_spread_start:omp_spread_size-omp_spread_size+2],dz[0:nrows]) \
            depend(out: active[omp_spread_start:omp_spread_size])
    
        #pragma omp target spread teams distribute \
            nowait \
            devices(0,1) spread_schedule(static, blocks_per_device) num_teams(blocks_per_device) thread_limit(threads) \
            map(   iA[0:pre_nrows+1], \
                 diag[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)], \
                   dt[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)], \
                 dAx2[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)], \
                    x[0:nrows], \
                   dy[0:nrows], \
                   dz[0:nrows], \
                   jA[0:nnz], \
                    A[0:nnz], \
                    temp2[omp_spread_start:omp_spread_size], \
                   omega0[omp_spread_start:omp_spread_size], \
                    temp1[omp_spread_start:omp_spread_size], \
                    alpha[omp_spread_start:omp_spread_size], \
                    reductor[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)]) \
            depend(in: active[omp_spread_start:omp_spread_size]) \
            depend(out: x[omp_spread_start:omp_spread_size])
        for(int j=0; j<blocks; j++) {
            int start=j*threads;
            if(start<nrows && active[j]==1){
                int end = start+threads<=nrows ? start+threads : nrows;// spare matrix vector multiplication
                #pragma omp parallel for
                for (int i=start; i<end; i++){
                    target_spmvCSR(i,pre_nrows, dt, dz, A, jA, iA); // cudaDeviceSpmvCSR(dt, dz, nrows, A, jA, iA);
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
                temp1[j] = reductor[start];
                
                // cudaDevicedotxy(dAx2, dAx2, &temp2, nrows, n_shr_empty);
                #pragma omp parallel for //reduction(+: rho1[j])
                for(int i=start; i<end; i++) {
                    reductor[i]=dAx2[i]*dAx2[i]; 
                }
                target_adhoc_sum_reduction(reductor, start, end);
                temp2[j] = reductor[start];
                
                omega0[j] = temp1[j] / temp2[j];
                
                #pragma omp parallel for
                for (int i=start; i<end; i++){
                    x[i]= alpha[j]*dy[i] + x[i]; // cudaDeviceaxpy(x, dy, alpha, nrows); // x=alpha*y +x
                    x[i]= omega0[j]*dz[i] + x[i]; // cudaDeviceaxpy(x, dz, omega0, nrows);
                }
            }
        }

        
        #pragma omp taskgroup
        {
            #pragma omp target update spread \
                nowait \
                devices(0,1) \
                range(0:nrows) \
                chunk_size(csize) \
                from(x[omp_spread_start:omp_spread_size]) \
                depend(in: x[omp_spread_start:omp_spread_size])
        }
        
        
        #pragma omp target update spread \
            nowait \
            devices(0,1) \
            range(0:nrows) \
            chunk_size(csize) \
            to(helper[omp_spread_start-omp_spread_start:omp_spread_size-omp_spread_size+2],x[0:nrows]) \
            depend(out: active[omp_spread_start:omp_spread_size])

        #pragma omp target spread teams distribute \
            nowait \
            devices(0,1) spread_schedule(static, blocks_per_device) num_teams(blocks_per_device) thread_limit(threads) \
            map(  dr0[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)], \
                   dt[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)], \
                   ds[omp_spread_start*threads:(threads*(omp_spread_start+omp_spread_size)>=nrows?nrows-omp_spread_start*threads:omp_spread_size*threads)], \
                    temp1[omp_spread_start:omp_spread_size], \
                   omega0[omp_spread_start:omp_spread_size], \
                     rho0[omp_spread_start:omp_spread_size], \
                     rho1[omp_spread_start:omp_spread_size]) \
            depend(inout: active[omp_spread_start:omp_spread_size])
        for(int j=0; j<blocks; j++) {
            int start=j*threads;
            if(start<nrows && active[j]==1){
                int end = start+threads<=nrows ? start+threads : nrows;
                #pragma omp parallel for
                for (int i=start; i<end; i++){
                    dr0[i]=1.0*ds[i] + (-1.0*omega0[j]) * dt[i]; // cudaDevicezaxpby(1.0, ds, -1.0 * omega0, dt, dr0, nrows); // z= a*x + b*y
                    dt[i] = 0.0; // cudaDevicesetconst(dt, 0.0, nrows);
                }

                // cudaDevicedotxy(dr0, dr0, &temp1, nrows, n_shr_empty);
                #pragma omp parallel for //reduction(+: rho1[j])
                for(int i=start; i<end; i++) {
                    reductor[i]=dr0[i]*dr0[i]; 
                }
                target_adhoc_sum_reduction(reductor, start, end);
                temp1[j] = reductor[start];
                
                temp1[j] = sqrtf(temp1[j]);
                rho0[j] = rho1[j];
            }
            if(temp1[j]<=tolmax){
                active[j]=0;
            }
        }
    
        #pragma omp taskgroup
        {
            #pragma omp target update spread \
                nowait \
                devices(0,1) \
                range(0:blocks) \
                chunk_size(blocks_per_device) \
                from(active[omp_spread_start:omp_spread_size]) \
                depend(in: active[omp_spread_start:omp_spread_size])
        }
        
        for(int j=first_active;j<blocks;j++){
            if(active[j]==1){
                keep_working=true;
                first_active=j;
                break;
            }
        }
        
        it++;
    }while(it<maxIt && keep_working);
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
  int scale=1;
  double tolmax=0.0;
  
  fscanf(fp, "%d",  &pre_n_cells);
  fscanf(fp, "%d",  &pre_nrows);
  fscanf(fp, "%d",  &pre_nnz);
  fscanf(fp, "%d",  &maxIt);
  fscanf(fp, "%d",  &mattype);
  fscanf(fp, "%le", &tolmax);
  fscanf(fp, "%d",  &scale);

  int *jA=(int*)malloc(scale*pre_nnz*sizeof(int));
  int *iA=(int*)malloc((pre_nrows+1)*sizeof(int));
  double *A=(double*)malloc(scale*pre_nnz*sizeof(double));
  double *diag=(double*)malloc(scale*pre_nrows*sizeof(double));
  double *x=(double*)malloc(scale*pre_nrows*sizeof(double));
  double *tempv=(double*)malloc(scale*pre_nrows*sizeof(double));

  double *dr0=(double*)malloc(scale*pre_nrows*sizeof(double));;
  double *dr0h=(double*)malloc(scale*pre_nrows*sizeof(double));
  double *dn0=(double*)malloc(scale*pre_nrows*sizeof(double));
  double *dp0=(double*)malloc(scale*pre_nrows*sizeof(double));
  double *dt=(double*)malloc(scale*pre_nrows*sizeof(double));
  double *ds=(double*)malloc(scale*pre_nrows*sizeof(double));
  double *dAx2=(double*)malloc(scale*pre_nrows*sizeof(double));
  double *dy=(double*)malloc(scale*pre_nrows*sizeof(double));
  double *dz=(double*)malloc(scale*pre_nrows*sizeof(double));

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
    fscanf(fp, "%le", &x[i]);
  }

  for(int i=0; i<pre_nrows; i++){
    fscanf(fp, "%le", &tempv[i]);
  }
  
  fclose(fp);
  
  for(int s=1;s<scale;s++)
  {
    memcpy(jA+(s*pre_nnz),jA,pre_nnz*sizeof(int));
    memcpy(A+(s*pre_nnz),A,pre_nnz*sizeof(double));
    memcpy(diag+(s*pre_nrows),diag,pre_nrows*sizeof(double));
    memcpy(x+(s*pre_nrows),x,pre_nrows*sizeof(double));
    memcpy(tempv+(s*pre_nrows),tempv,pre_nrows*sizeof(double));
  }
  
  {
    int n_cells=pre_n_cells*scale;
    int nrows=pre_nrows*scale;
    int nnz=pre_nnz*scale;
    
    int num_devices=2;
    int threads=154;
    int blocks=pre_n_cells*scale;
    int blocks_per_device=blocks/num_devices;    
    int csize=blocks_per_device*threads; // 770
    int offset_cells=0;
    int len_cell=nrows/n_cells;
    
    #pragma omp taskgroup
    {
      #pragma omp target enter data spread \
              nowait \
              devices(0,1) \
              range(0:nrows) \
              chunk_size(csize) \
              map(to:       iA[0:pre_nrows+1], \
                          diag[omp_spread_start:omp_spread_size], \
                        tempv[omp_spread_start:omp_spread_size], \
                            x[0:nrows], \
                            jA[0:nnz], \
                            A[0:nnz]) \
              map(alloc:   dr0[omp_spread_start:omp_spread_size], \
                          dr0h[omp_spread_start:omp_spread_size], \
                          dn0[omp_spread_start:omp_spread_size], \
                          dp0[omp_spread_start:omp_spread_size], \
                            dt[omp_spread_start:omp_spread_size], \
                            ds[omp_spread_start:omp_spread_size], \
                          dAx2[omp_spread_start:omp_spread_size], \
                            dy[0:nrows], \
                            dz[0:nrows])
    }
    
    printf("solveGPU_block_thr n_cells %d len_cell %d nrows %d nnz %d blocks %d threads_block %d n_shr_empty %d offset_cells %d\n",
                n_cells,len_cell,nrows,nnz,blocks,threads, blocks*threads-nrows,offset_cells);
    
    solveBcg_spread(blocks, blocks_per_device, threads, num_devices, csize, pre_nrows, A, jA, iA, x, tempv, nrows, maxIt, mattype, nnz, n_cells, tolmax, diag, dr0, dr0h, dn0, dp0, dt, ds, dAx2, dy, dz);

    #pragma omp taskgroup
    {
      #pragma omp target exit data spread \
              nowait \
              devices(0,1) \
              range(0:nrows) \
              chunk_size(csize) \
              map(from:    diag[omp_spread_start:omp_spread_size], \
                          tempv[omp_spread_start:omp_spread_size], \
                              x[omp_spread_start:omp_spread_size]) \
              map(release:   dr0[omp_spread_start:omp_spread_size], \
                            dr0h[omp_spread_start:omp_spread_size], \
                            dn0[omp_spread_start:omp_spread_size], \
                            dp0[omp_spread_start:omp_spread_size], \
                              dt[omp_spread_start:omp_spread_size], \
                              ds[omp_spread_start:omp_spread_size], \
                            dAx2[omp_spread_start:omp_spread_size], \
                              dy[0:nrows], \
                              dz[0:nrows])
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

  int flag=1;
  int s;
  for(s=0;s<scale;s++)
  {
    if(compare_doubles(A2,A+(s*pre_nnz),pre_nnz,"A2")==0) flag=0;
    if(compare_doubles(x2,x+(s*pre_nrows),pre_nrows,"x2")==0)  flag=0;
    if(compare_doubles(tempv2,tempv+(s*pre_nrows),pre_nrows,"tempv2")==0)  flag=0;
    if(flag==0)
      break;
  }

  if(flag==0)
    printf("FAIL_spread at %d\n",s);
  else
    printf("SUCCESS_spread\n");
}

int main()
{
  BCG();

 return 0;
}
