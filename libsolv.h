#ifndef LIBSOLV_H
#define LIBSOLV_H

#include <cuda.h>
#include <stdio.h>

typedef struct {
  int lenjA;
  int leniA;
  double *dA;
  int *djA;
  int *diA;
  double *dx;
  double* dtempv;
  int n_specs;
  int nrows;
  int nnz;
  int n_shr_empty;
  int n_cells;
  double *ddiag;
  double *dr0;
  double *dr0h;
  double *dn0;
  double *dp0;
  double *dt;
  double *ds;
  double *dy;
} ModelDataGPU;

void solveGPU_block(ModelDataGPU* mGPU);

#endif