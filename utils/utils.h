#include "mmio.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void write_MTX(const int rows, const int cols, const int nnz, int *row_ptr,
               int *col_ind, double *data) {
  // Example configuration taken from KLU Sparse pdf
  /*  const int rows = 3;
    const int cols = rows;
    const int nnz = 6;
    int row_ptr[rows + 1] = {0, 3, 5, 6};
    int col_ind[nnz] = {0, 1, 2, 1, 2, 2};
    double data[nnz] = {5., 4., 3., 2., 1., 8.};
  */

  char filename[64];
  time_t now = time(NULL);
  sprintf(filename, "matrix_%ld.mtx", now);
  FILE *f = fopen(filename, "w");
  MM_typecode matcode;
  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_coordinate(&matcode);
  mm_set_real(&matcode);
  mm_write_banner(f, matcode);
  mm_write_mtx_crd_size(f, rows, cols, nnz);
  for (int i = 0; i < rows; ++i)
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
      fprintf(f, "%d %d %g\n", i + 1, col_ind[j] + 1, data[j]);

  fclose(f);
  printf("Wrote MTX\n");
}
