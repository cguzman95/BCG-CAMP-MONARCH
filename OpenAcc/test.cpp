/* Copyright (C) 2022 Barcelona Supercomputing Center
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include<math.h>
#include<iostream>


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

void openaccIterative(){

  int blocks = 100;
  int threads_block = 73;
  int n_shr_memory = nextPowerOfTwo(threads_block);
  int len = blocks*threads_block;

  double *x = (double *) malloc(len * sizeof(double));
  memset(x, 0, len * sizeof(double));
  double *y = (double *) malloc(len * sizeof(double));
  memset(y, 1, len * sizeof(double));

  int it = 0;
  int maxIt = 5;
  double a = 0.0;

//#pragma acc data copy(x[1:len]) //create(xnew[len])
#pragma acc data copy(x[1:len],y[1:len])
  while(it<maxIt){

    //y[i] = threadIdx.x;
#pragma acc parallel loop
    for(int i=0; i<len; i++) {
      y[i] = i;
    }
    //cudaDevicereduce(y,&a,sdata,n_shr_empty);
#pragma acc parallel loop reduction(+:a)
    for(int i=0; i<len; i++) {
      a = y[i];
    }
    //cudaDevicemaxD(x,&a,sdata,n_shr_empty);
#pragma acc parallel loop
    for(int i=0; i<len; i++) {
      a = fmax(a, x[i]);
    }

    //printf("it %d \n",it);

    it++;
  }

#pragma acc parallel loop
  for(int i=0; i<len; i++) {
    y[i] = a;
  }

  /*
  double *dx,*dy;
  cudaMalloc((void **) &dx, len * sizeof(double));
  cudaMalloc((void **) &dy, len * sizeof(double));

  HANDLE_ERROR(cudaMemcpy(dx, x, len*sizeof(double), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dy, y, len*sizeof(double), cudaMemcpyHostToDevice));

  cudaIterative <<<blocks,threads_block,n_shr_memory*sizeof(double)>>>
                                        (dx,dy,n_shr_empty);

  HANDLE_ERROR(cudaMemcpy( y, dy, len*sizeof(double), cudaMemcpyDeviceToHost ));
*/

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
  openaccIterative();

	return 0;
}
