#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <mpi.h>

#include "libsolv.h"

int compare_doubles(double* x, double* y, int len, const char* s) {

    int flag = 1;
    double tol = 0.0001;
    double rel_error, abs_error;
    int n_fails = 0;
    for (int i = 0; i < len; i++) {
        abs_error = abs(x[i] - y[i]);
        if (x[i] == 0)
            rel_error = 0.;
        else
            rel_error = abs((x[i] - y[i]) / x[i]);
        if ((rel_error > tol && abs_error > 1.0E-30) || y[i] != y[i]) {
            //if(true){
            printf("compare_doubles %s rel_error %le abs_error %le for tol %le at [%d]: %le vs %le\n",
                s, rel_error, abs_error, tol, i, x[i], y[i]);
            flag = 0;
            n_fails++;
            if (n_fails == 4)
                return flag;
        }
    }

    return flag;

}

int nextPowerOfTwo(int v) {

    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;

    return v;
}


//Based on
// https://github.com/scipy/scipy/blob/3b36a574dc657d1ca116f6e230be694f3de31afc/scipy/sparse/sparsetools/csr.h#L363
void swapCSC_CSR(int n_row, int n_col, int* Ap, int* Aj, double* Ax, int* Bp, int* Bi, double* Bx){

  int nnz=Ap[n_row];

  memset(Bp, 0, (n_row+1)*sizeof(int));

  for (int n = 0; n < nnz; n++){
    Bp[Aj[n]]++;
  }

  //cumsum the nnz per column to get Bp[]
  for(int col = 0, cumsum = 0; col < n_col; col++){
    int temp  = Bp[col];
    Bp[col] = cumsum;
    cumsum += temp;
  }
  Bp[n_col] = nnz;

  for(int row = 0; row < n_row; row++){
    for(int jj = Ap[row]; jj < Ap[row+1]; jj++){
      int col  = Aj[jj];
      int dest = Bp[col];

      Bi[dest] = row;
      Bx[dest] = Ax[jj];

      Bp[col]++;
    }
  }

  for(int col = 0, last = 0; col <= n_col; col++){
    int temp  = Bp[col];
    Bp[col] = last;
    last    = temp;
  }

}

void swapCSC_CSR_BCG(ModelDataGPU *mGPU,
                     int *Ap, int *Aj, double *Ax){

#ifdef TEST_CSCtoCSR

  //Example configuration taken from KLU Sparse pdf
  int n_row=3;
  int n_col=n_row;
  int nnz=6;
  int Ap[n_row+1]={0,3,5,6};
  int Aj[nnz]={0,1,2,1,2,2};
  double Ax[nnz]={5.,4.,3.,2.,1.,8.};
  int* Bp=(int*)malloc((n_row+1)*sizeof(int));
  int* Bi=(int*)malloc(nnz*sizeof(int));
  double* Bx=(double*)malloc(nnz*sizeof(double));

#elif TEST_CSRtoCSC

  //Example configuration taken from KLU Sparse pdf
  int n_row=3;
  int n_col=n_row;
  int nnz=6;
  int Ap[n_row+1]={0,1,3,6};
  int Aj[nnz]={0,0,1,0,1,2};
  double Ax[nnz]={5.,4.,2.,3.,1.,8.};
  int* Bp=(int*)malloc((n_row+1)*sizeof(int));
  int* Bi=(int*)malloc(nnz*sizeof(int));
  double* Bx=(double*)malloc(nnz*sizeof(double));

#else

  int n_row=mGPU->nrows;
  int n_col=mGPU->nrows;
  int nnz=mGPU->nnz;
  int* Bp=(int*)malloc((mGPU->nrows+1)*sizeof(int));
  int* Bi=(int*)malloc(mGPU->nnz*sizeof(int));
  double* Bx=(double*)malloc(nnz*sizeof(double));

#endif

  swapCSC_CSR(n_row,n_col,Ap,Aj,Ax,Bp,Bi,Bx);

#ifdef TEST_CSCtoCSR

  //Correct result:
  //int Cp[n_row+1]={0,1,3,6};
  //int Ci[nnz]={0,0,1,0,1,2};
  //int Cx[nnz]={5,4,2,3,1,8};

  printf("Bp:\n");
  for(int i=0;i<=n_row;i++)
    printf("%d ",Bp[i]);
  printf("\n");
  printf("Bi:\n");
  for(int i=0;i<nnz;i++)
    printf("%d ",Bi[i]);
  printf("\n");
  printf("Bx:\n");
  for(int i=0;i<nnz;i++)
    printf("%-le ",Bx[i]);
  printf("\n");

  exit(0);

#elif TEST_CSRtoCSC

  //Correct result:
  //int Cp[n_row+1]={0,3,5,6};
  //int Ci[nnz]={0,1,2,1,2,2};
  //int Cx[nnz]={5,4,3,2,1,8};

  printf("Bp:\n");
  for(int i=0;i<=n_row;i++)
    printf("%d ",Bp[i]);
  printf("\n");
  printf("Bi:\n");
  for(int i=0;i<nnz;i++)
    printf("%d ",Bi[i]);
  printf("\n");
  printf("Bx:\n");
  for(int i=0;i<nnz;i++)
    printf("%-le ",Bx[i]);
  printf("\n");
  exit(0);

#else

  for(int i=0;i<=n_row;i++)
    Ap[i] = Bp[i];
  for(int i=0;i<nnz;i++)
    Aj[i] = Bi[i];
  for(int i=0;i<nnz;i++)
    Ax[i] = Bx[i];

#endif

  free(Bp);
  free(Bi);
  free(Bx);

}

void swapCSR_CSD(int n_row, int n_col, int* Ap, int* Aj, double* Ax, int* Bp, int* Bi, double* Bx){

  int nnz=Ap[n_row];

  memset(Bp, 0, (n_row+1)*sizeof(int));
  int* Bpi=(int*)malloc((n_row)*sizeof(int));
  //int* BiBool=(int*)malloc((n_row*n_row)*sizeof(int));
  //memset(BiBool, 0, (n_row*n_row)*sizeof(int));

  Bpi[0]=0;
  for(int i = 1; i < n_row; i++){
    Bpi[i]=n_row-i;
    //printf("Bpi i %d %d \n",Bpi[i],i);
  } //0 2 1

  for(int row = 0; row < n_row; row++){
    for(int j = Ap[row]; j < Ap[row+1]; j++) {
      Bp[Bpi[Aj[j]]+1]++; //Add value to nº values for diagonal

      //printf("Bpi Aj[j] %d %d \n",Bpi[Aj[j]],Aj[j]);
      //0 2 1
      //1 0 2
      //2 1 0
    }
    //0 2 1
    for(int i = 0; i < n_row; i++){
      Bpi[i]++;
      if(Bpi[i]==n_row){
        Bpi[i]=0;
      }
      //printf("Bpi i %d %d \n",Bpi[i],i);
    }//1 0 2
  }
  //printf("n_row %d \n",n_row);

  for(int i = 0; i < n_row+1; i++){
    Bp[i+1]+=Bp[i];
  }

  /*
  printf("Bpi:\n");
  for(int i=0;i<n_row;i++)
    printf("%d ",Bpi[i]);
  printf("\n");
  printf("Bp:\n");
  for(int i=0;i<n_row+1;i++)
    printf("%d ",Bp[i]);
  printf("\n");
*/
  //exit(0);

  memset(Bx, 0, (nnz)*sizeof(double));
  int* offsetBx=(int*)malloc((n_row)*sizeof(int));
  memset(offsetBx, 0, (n_row)*sizeof(int));
  memset(Bi, 0, (nnz)*sizeof(int));
  for(int row = 0; row < n_row; row++){
    for(int j = Ap[row]; j < Ap[row+1]; j++) {
      if(Aj[j]<=row){
        int iDiag=Bpi[Aj[j]];
        int nElemTillDiag=Bp[iDiag];
        Bx[nElemTillDiag+offsetBx[iDiag]]=Ax[j];
        Bi[nElemTillDiag+offsetBx[iDiag]]=Aj[j];

        //printf("nElemTillDiag  offsetBx[iDiag] Aj[j] %d %d %d %d\n",nElemTillDiag, offsetBx[iDiag],iDiag,Aj[j]);
        offsetBx[iDiag]++;
      }
    }
    //0 2 1
    for(int i = 0; i < n_row; i++){
      Bpi[i]++;
      if(Bpi[i]==n_row){
        Bpi[i]=0;
      }
      //printf("Bpi i %d %d \n",Bpi[i],i);
    }//1 0 2
  }

  for(int row = 0; row < n_row; row++){
    for(int j = Ap[row]; j < Ap[row+1]; j++) {
      if(Aj[j]>row){
        int iDiag=Bpi[Aj[j]];
        int nElemTillDiag=Bp[iDiag];
        Bx[nElemTillDiag+offsetBx[iDiag]]=Ax[j];
        Bi[nElemTillDiag+offsetBx[iDiag]]=Aj[j];

        //printf("nElemTillDiag  offsetBx[iDiag] Aj[j] %d %d %d %d\n",nElemTillDiag, offsetBx[iDiag],iDiag,Aj[j]);
        offsetBx[iDiag]++;
      }
    }
    //0 2 1
    for(int i = 0; i < n_row; i++){
      Bpi[i]++;
      if(Bpi[i]==n_row){
        Bpi[i]=0;
      }
      //printf("Bpi i %d %d \n",Bpi[i],i);
    }//1 0 2
  }

#ifdef TEST_CSRtoCSD

  /*
  printf("BiBool:\n");
  for(int i=0;i<n_row*n_row;i++)
    printf("%d ",BiBool[i]);
  printf("\n");
   */

  printf("Bp:\n");
  for(int i=0;i<n_row+1;i++)
    printf("%d ",Bp[i]);
  printf("\n");

  printf("Bi:\n");
  for(int i=0;i<nnz;i++)
    printf("%d ",Bi[i]);
  printf("\n");

  printf("Bx:\n");
  for(int i=0;i<nnz;i++)
    printf("%le ",Bx[i]);
  printf("\n");

  exit(0);

  //free(BiBool);
#endif

  free(Bpi);
  free(offsetBx);
}

void swapCSC_CSD_BCG(ModelDataGPU *mGPU,
                     int *Ap0, int *Aj0, double *Ax0){

#ifdef TEST_CSRtoCSD

  //Example configuration based in  KLU Sparse pdf
  const int n_row=3;
  const int n_col=n_row;
  int nnz=6;
  int Ap[n_row+1]={0,1,3,6};
  int Aj[nnz]={0,1,2,0,1,2};
  double Ax[nnz]={5., 2., 7., 3.,1.,8.};
  /*
  int nnz=6;
  int Ap[n_row+1]={0,1,3,6};
  int Aj[nnz]={0,0,1,0,1,2};
  double Ax[nnz]={5.,4.,2.,3.,1.,8.};
   */
  int* Bp=(int*)malloc((n_row+1)*sizeof(int)); //Nº of values for each diagonal
  int* Bi=(int*)malloc(nnz*sizeof(int));
  double* Bx=(double*)malloc(nnz*sizeof(double));

#else

  int *Ap=Ap0;
  int *Aj=Aj0;
  double *Ax=Ax0;
  int n_row=mGPU->nrows;
  int n_col=mGPU->nrows;
  int nnz=mGPU->nnz;
  int* Bp=(int*)malloc((n_row+1)*sizeof(int));
  int* Bi=(int*)malloc(nnz*sizeof(int));
  double* Bx=(double*)malloc(nnz*sizeof(double));

#endif

  swapCSR_CSD(n_row,n_col,Ap,Aj,Ax,Bp,Bi,Bx);

  for(int i=0;i<=n_row;i++)
    Ap[i] = Bp[i];
  for(int i=0;i<nnz;i++)
    Aj[i] = Bi[i];
  for(int i=0;i<nnz;i++)
    Ax[i] = Bx[i];

  free(Bp);
  free(Bi);
  free(Bx);

}


void swapCSR_CUID(int n_row, int n_col, int* Ap, int* Aj, double* Ax, int* Bp, int* Bi, double* Bx){

  int nnz=Ap[n_row];

  memset(Bp, 0, (n_row+1)*sizeof(int));
  int* Bpi=(int*)malloc((n_row)*sizeof(int));
  //int* BiBool=(int*)malloc((n_row*n_row)*sizeof(int));
  //memset(BiBool, 0, (n_row*n_row)*sizeof(int));

  Bpi[0]=0;
  for(int i = 1; i < n_row; i++){
    Bpi[i]=n_row-i;
    //printf("Bpi i %d %d \n",Bpi[i],i);
  } //0 2 1

  for(int row = 0; row < n_row; row++){
    for(int j = Ap[row]; j < Ap[row+1]; j++) {
      Bp[Bpi[Aj[j]]]++; //Add value to nº values for diagonal

      //AiBool[row*n_row+Aj[j]]=1;

      //printf("Bpi Aj[j] %d %d \n",Bpi[Aj[j]],Aj[j]);
      //0 2 1
      //1 0 2
      //2 1 0
    }
    //0 2 1
    for(int i = 0; i < n_row; i++){
      Bpi[i]++;
      if(Bpi[i]==n_row){
        Bpi[i]=0;
      }
      //printf("Bpi i %d %d \n",Bpi[i],i);
    }//1 0 2
  }
  //printf("n_row %d \n",n_row);

/*
  printf("Bpi:\n");
  for(int i=0;i<n_row;i++)
    printf("%d ",Bpi[i]);
  printf("\n");
  printf("Bp:\n");
  for(int i=0;i<n_row;i++)
    printf("%d ",Bp[i]);
  printf("\n");
*/

  //exit(0);

  memset(Bx, 0, (nnz)*sizeof(double));
  int* offsetBx=(int*)malloc((n_row)*sizeof(int));
  memset(offsetBx, 0, (n_row)*sizeof(int));
  memset(Bi, -1, (n_row*n_row)*sizeof(int));
  for(int row = 0; row < n_row; row++){
    for(int j = Ap[row]; j < Ap[row+1]; j++) {
      if(Aj[j]<=row){
        int iDiag=Bpi[Aj[j]];
        int nElemTillDiag=0;
        for(int i = 0; i < iDiag; i++){
          nElemTillDiag+=Bp[i];
        }
        //printf("nElemTillDiag  offsetBx[iDiag] Aj[j] %d %d %d %d\n",nElemTillDiag, offsetBx[iDiag],iDiag,Aj[j]);
        Bx[nElemTillDiag+offsetBx[iDiag]]=Ax[j];
        int iBi=iDiag*n_row+Aj[j];
        Bi[iBi]=nElemTillDiag+offsetBx[iDiag];
        //printf("Bi[i] %d %d\n",Bi[iBi], iBi);
        //BiBool[nElemTillDiag+offsetBx[iDiag]]=1;
        offsetBx[iDiag]++;
      }
    }
    //0 2 1
    for(int i = 0; i < n_row; i++){
      Bpi[i]++;
      if(Bpi[i]==n_row){
        Bpi[i]=0;
      }
      //printf("Bpi i %d %d \n",Bpi[i],i);
    }//1 0 2
  }

  for(int row = 0; row < n_row; row++){
    for(int j = Ap[row]; j < Ap[row+1]; j++) {
      if(Aj[j]>row){
        int iDiag=Bpi[Aj[j]];
        int nElemTillDiag=0;
        for(int i = 0; i < iDiag; i++){
          nElemTillDiag+=Bp[i];
        }
        //printf("nElemTillDiag  offsetBx[iDiag] Aj[j] %d %d %d %d\n",nElemTillDiag, offsetBx[iDiag],iDiag,Aj[j]);
        Bx[nElemTillDiag+offsetBx[iDiag]]=Ax[j];
        int iBi=iDiag*n_row+Aj[j];
        Bi[iBi]=nElemTillDiag+offsetBx[iDiag];
        //printf("Bi[i] %d %d\n",Bi[iBi], iBi);
        //BiBool[nElemTillDiag+offsetBx[iDiag]]=1;
        offsetBx[iDiag]++;
      }
    }
    //0 2 1
    for(int i = 0; i < n_row; i++){
      Bpi[i]++;
      if(Bpi[i]==n_row){
        Bpi[i]=0;
      }
      //printf("Bpi i %d %d \n",Bpi[i],i);
    }//1 0 2
  }

#ifdef TEST_CSRtoCSD

  /*
  printf("BiBool:\n");
  for(int i=0;i<n_row*n_row;i++)
    printf("%d ",BiBool[i]);
  printf("\n");
   */

  printf("Bi:\n");
  for(int i=0;i<n_row*n_row;i++)
    printf("%d ",Bi[i]);
  printf("\n");

  printf("Bx:\n");
  for(int i=0;i<nnz;i++)
    printf("%le ",Bx[i]);
  printf("\n");

  exit(0);

  //free(BiBool);
#endif

  free(Bpi);
  free(offsetBx);

}

void swapCSR_CUID_BCG(ModelDataGPU *mGPU,
                     int *Ap0, int *Aj0, double *Ax0, int *Aj1){

#ifdef TEST_CSRtoCSD

  //Example configuration based in  KLU Sparse pdf
  int n_row=3;
  int n_col=n_row;
  int nnz=6;
  int Ap[n_row+1]={0,1,3,6};
  int Aj[nnz]={0,1,2,0,1,2};
  double Ax[nnz]={5., 2., 7., 3.,1.,8.};
  /*
  int nnz=6;
  int Ap[n_row+1]={0,1,3,6};
  int Aj[nnz]={0,0,1,0,1,2};
  double Ax[nnz]={5.,4.,2.,3.,1.,8.};
   */
  int* Bp=(int*)malloc((n_row+1)*sizeof(int)); //Nº of values for each diagonal
  int* Bi=(int*)malloc(n_row*n_row*sizeof(int));
  double* Bx=(double*)malloc(nnz*sizeof(double));

#else

  int *Ap=Ap0;
  int *Aj=Aj0;
  double *Ax=Ax0;
  int n_row=mGPU->nrows;
  int n_col=mGPU->nrows;
  int nnz=mGPU->nnz;
  int* Bp=(int*)malloc((n_row+1)*sizeof(int));
  int* Bi=(int*)malloc(n_row*n_row*sizeof(int));
  double* Bx=(double*)malloc(nnz*sizeof(double));

#endif

  swapCSR_CUID(n_row,n_col,Ap,Aj,Ax,Bp,Bi,Bx);

  for(int i=0;i<n_row*n_row;i++)
    Aj1[i] = Bi[i];
  for(int i=0;i<nnz;i++)
    Ax[i] = Bx[i];

  free(Bi);
  free(Bp);
  free(Bx);

}


void BCG() {

  char confPath [255];
  int nDevices;
  int n_cells;

  FILE* fp;
  fp = fopen("../data/conf.txt", "r");
  if (fp == NULL)
    printf("Could not open file %s\n",confPath);
  fscanf(fp, "%s", confPath);
  fscanf(fp, "%d", &nDevices);
  fscanf(fp, "%d", &n_cells);
  fclose(fp);

  ModelDataGPU* mGPUs = (ModelDataGPU*)malloc(nDevices * sizeof(ModelDataGPU));
  ModelDataGPU* mGPU = &mGPUs[0];
  ModelDataGPU mGPU0_object;
  ModelDataGPU* mGPU0 = &mGPU0_object;

  double timeBiConjGrad = 0;
  cudaEvent_t startBCG;
  cudaEvent_t stopBCG;
  cudaEventCreate(&startBCG);
  cudaEventCreate(&stopBCG);

  fp = fopen("../data/confBCG.txt", "r");
  if (fp == NULL) {
      printf("File not found \n");
      exit(EXIT_FAILURE);
  }

  fscanf(fp, "%d", &mGPU0->n_cells);
  int cellsConfBCG = mGPU0->n_cells;
  int n_cells_multiplier = n_cells/cellsConfBCG;
  fscanf(fp, "%d", &mGPU0->nrows);
  fscanf(fp, "%d", &mGPU0->nnz);
  int maxIt, mattype;
  fscanf(fp, "%d", &maxIt);
#ifdef DEBUG_MAXIT
  mGPU0->maxIt=1;
#endif
  fscanf(fp, "%d", &mattype);
  double tolmax;
  fscanf(fp, "%le", &tolmax);

  int* jA_aux = (int*)malloc(mGPU0->nnz * sizeof(int));
  int* iA_aux = (int*)malloc((mGPU0->nrows + 1) * sizeof(int));
  double* A_aux = (double*)malloc(mGPU0->nnz * sizeof(double));
  double* diag_aux = (double*)malloc(mGPU0->nrows * sizeof(double));
  double* x_aux = (double*)malloc(mGPU0->nrows * sizeof(double));
  double* tempv_aux = (double*)malloc(mGPU0->nrows * sizeof(double));

  for (int i = 0; i < mGPU0->nnz; i++) {
      fscanf(fp, "%d", &jA_aux[i]);
      //printf("%d %d\n",i, jA_aux[i]);
  }

  for (int i = 0; i < mGPU0->nrows + 1; i++) {
      fscanf(fp, "%d", &iA_aux[i]);
      //printf("%d %d\n",i, iA[i]);
  }

  for (int i = 0; i < mGPU0->nnz; i++) {
      fscanf(fp, "%le", &A_aux[i]);
      //printf("%d %le\n",i, A[i]);
  }

  for (int i = 0; i < mGPU0->nrows; i++) {
      fscanf(fp, "%le", &diag_aux[i]);
      //printf("%d %le\n",i, diag[i]);
  }

  for (int i = 0; i < mGPU0->nrows; i++) {
      fscanf(fp, "%le", &x_aux[i]);
      //printf("%d %le\n",i, x[i]);
  }

  for (int i = 0; i < mGPU0->nrows; i++) {
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

#ifdef CSR
  printf("CSR\n");
#elif CSC
  printf("CSC_SHARED\n");
  swapCSC_CSR_BCG(mGPU0,iA_aux,jA_aux,A_aux);
#elif CSC_LOOP_ROWS
  printf("CSC_LOOP_ROWS\n");
  swapCSC_CSR_BCG(mGPU0,iA_aux,jA_aux,A_aux);
#elif CSD
  printf("CSD\n");
  swapCSC_CSD_BCG(mGPU0,iA_aux,jA_aux,A_aux);
#elif CUID
  printf("CUID\n");
  int* jA1=(int*)malloc((mGPU0->nrows*mGPU0->nrows)*sizeof(int));
  swapCSR_CUID_BCG(mGPU0,iA_aux,jA_aux,A_aux,jA1);
  free(jA_aux);
#else
  printf("CSR\n");
#endif

  int* jA = jA_aux;
  int* iA = iA_aux;
  double* A = (double*)malloc(mGPU0->nnz * n_cells_multiplier * sizeof(double));
  double* diag = (double*)malloc(mGPU0->nrows * n_cells_multiplier * sizeof(double));
  double* x = (double*)malloc(mGPU0->nrows * n_cells_multiplier * sizeof(double));
  double* tempv = (double*)malloc(mGPU0->nrows * n_cells_multiplier * sizeof(double));

  iA[0] = 0;
  for (int i = 0; i < n_cells_multiplier; i++) {
      memcpy(A + i * mGPU0->nnz, A_aux, mGPU0->nnz * sizeof(double));
      memcpy(diag + i * mGPU0->nrows, diag_aux, mGPU0->nrows * sizeof(double));
      memcpy(x + i * mGPU0->nrows, x_aux, mGPU0->nrows * sizeof(double));
      memcpy(tempv + i * mGPU0->nrows, tempv_aux, mGPU0->nrows * sizeof(double));
  }

  mGPU0->lenjA=mGPU0->nnz;
#ifdef CUID
  mGPU0->lenjA=mGPU0->nrows*mGPU0->nrows;
#endif
  mGPU0->leniA=mGPU0->nrows+1;
  mGPU0->n_cells = mGPU0->n_cells * n_cells_multiplier;
  mGPU0->nnz = mGPU0->nnz * n_cells_multiplier;
  mGPU0->nrows = mGPU0->nrows * n_cells_multiplier;

  int remainder = mGPU0->n_cells % nDevices;
  for (int iDevice = 0; iDevice < nDevices; iDevice++) {
    cudaSetDevice(iDevice);
    mGPU = &mGPUs[iDevice];
    int n_cells = int(mGPU0->n_cells / nDevices);
    if (remainder != 0 && iDevice == 0) {
      //printf("REMAINDER  nDevicesMODn_cells!=0\n");
      //printf("remainder %d n_cells_total %d nDevices %d n_cells %d\n",remainder,mGPU0->n_cells,nDevices,n_cells);
      n_cells += remainder;
    }
    mGPU->n_cells = n_cells;
    mGPU->nrows = mGPU0->nrows / mGPU0->n_cells * mGPU->n_cells;
    mGPU->nnz = mGPU0->nnz / mGPU0->n_cells * mGPU->n_cells;
    mGPU->lenjA = mGPU0->lenjA;
    mGPU->leniA = mGPU0->leniA;
    cudaMalloc((void**)&mGPU->djA, mGPU->lenjA * sizeof(int));
    cudaMalloc((void**)&mGPU->diA, mGPU0->leniA * sizeof(int));
    cudaMalloc((void**)&mGPU->dA, mGPU->nnz * sizeof(double));
    cudaMalloc((void**)&mGPU->ddiag, mGPU->nrows * sizeof(double));
    cudaMalloc((void**)&mGPU->dx, mGPU->nrows * sizeof(double));
    cudaMalloc((void**)&mGPU->dtempv, mGPU->nrows * sizeof(double));
    //Auxiliary vectors ("private")
    double** dr0 = &mGPU->dr0;
    double** dr0h = &mGPU->dr0h;
    double** dn0 = &mGPU->dn0;
    double** dp0 = &mGPU->dp0;
    double** dt = &mGPU->dt;
    double** ds = &mGPU->ds;
    double** dy = &mGPU->dy;
    int nrows = mGPU->nrows;
    cudaMalloc(dr0, nrows * sizeof(double));
    cudaMalloc(dr0h, nrows * sizeof(double));
    cudaMalloc(dn0, nrows * sizeof(double));
    cudaMalloc(dp0, nrows * sizeof(double));
    cudaMalloc(dt, nrows * sizeof(double));
    cudaMalloc(ds, nrows * sizeof(double));
    cudaMalloc(dy, nrows * sizeof(double));
  }

  int offset_nnz = 0;
  int offset_nrows = 0;
  remainder = mGPU0->n_cells % nDevices;
  cudaSetDevice(0);
  cudaEventRecord(startBCG);
  for (int iDevice = 0; iDevice < nDevices; iDevice++) {
    cudaSetDevice(iDevice);
    mGPU = &mGPUs[iDevice];
    int n_cells = int(mGPU0->n_cells / nDevices);
    if (remainder != 0 && iDevice == 0) {
        //printf("REMAINDER  nDevicesMODn_cells!=0\n");
        //printf("remainder %d n_cells_total %d nDevices %d n_cells %d\n",remainder,mGPU0->n_cells,nDevices,n_cells);
        n_cells += remainder;
    }
    mGPU->n_cells = n_cells;
    mGPU->nrows = mGPU0->nrows / mGPU0->n_cells * mGPU->n_cells;
    mGPU->nnz = mGPU0->nnz / mGPU0->n_cells * mGPU->n_cells;
    mGPU->lenjA = mGPU0->lenjA;
    mGPU->leniA = mGPU0->leniA;
    int len_cell = mGPU->nrows / mGPU->n_cells;
    int n_shr_memory = nextPowerOfTwo(len_cell);

#ifdef CUID
    cudaMemcpyAsync(mGPU->djA, jA1, mGPU->lenjA * sizeof(int), cudaMemcpyHostToDevice, 0);
#else
    cudaMemcpyAsync(mGPU->djA, jA, mGPU->lenjA * sizeof(int), cudaMemcpyHostToDevice, 0);
#endif
    cudaMemcpyAsync(mGPU->diA, iA, mGPU0->leniA * sizeof(int), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(mGPU->dA, A + offset_nnz, mGPU->nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(mGPU->ddiag, diag + offset_nrows, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(mGPU->dx, x + offset_nrows, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(mGPU->dtempv, tempv + offset_nrows, mGPU->nrows * sizeof(double), cudaMemcpyHostToDevice, 0);

    solveGPU_block(mGPU);

    cudaMemcpyAsync(x + offset_nrows, mGPU->dx, mGPU->nrows * sizeof(double), cudaMemcpyDeviceToHost, 0);

    offset_nnz += mGPU->nnz;
    offset_nrows += mGPU->nrows;
  }

  for (int iDevice = 1; iDevice < nDevices; iDevice++) {
    cudaSetDevice(iDevice);
    cudaDeviceSynchronize();
  }

  cudaSetDevice(0);
  cudaEventRecord(stopBCG);
  cudaEventSynchronize(stopBCG);
  float msBiConjGrad = 0.0;
  cudaEventElapsedTime(&msBiConjGrad, startBCG, stopBCG);
  timeBiConjGrad += msBiConjGrad / 1000;

  /*
    for(int icell=0; icell<mGPU0->n_cells; icell++){
      printf("cell %d:\n",icell);
      for(int i=0; i<mGPU0->nrows/mGPU0->n_cells; i++){
        printf("%le ", x[i+icell*(mGPU0->nrows/mGPU0->n_cells)]);
        //printf("%d %d\n",i, iA[i]);
      }
      printf("\n");
    }*/

  mGPU0->nrows = mGPU0->nrows / n_cells_multiplier;
  double* x2_aux = (double*)malloc(mGPU0->nrows * sizeof(double));
  char fout[]="../data/outBCG.txt";
  fp = fopen(fout, "r");
  for (int i = 0; i < mGPU0->nrows; i++) {
      fscanf(fp, "%le", &x2_aux[i]);
  }
  fclose(fp);
  double* x2 = (double*)malloc(mGPU0->nrows * n_cells_multiplier * sizeof(double));
  for (int i = 0; i < n_cells_multiplier; i++) {
      memcpy(x2 + i * mGPU0->nrows, x2_aux, mGPU0->nrows * sizeof(double));
  }
  mGPU0->nrows = mGPU0->nrows * n_cells_multiplier;
  int flag = 1;
  if (compare_doubles(x2, x, mGPU0->nrows, "x2") == 0)  flag = 0;
  if (flag == 0)
      printf("FAIL\n");
  else
      printf("SUCCESS\n");
  printf("timeBiConjGrad %.2e\n",timeBiConjGrad);
  fp = fopen("out/timesAndCounters.csv", "w");
  fprintf(fp,"timeBiConjGrad,counterBCGInternal\n");
  fprintf(fp,"%.2e\n",timeBiConjGrad);
  fclose(fp);

}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  BCG();
  MPI_Finalize();
  return 0;
}
