#!/usr/bin/env bash

echo "Start make.power9.sh"

cd build
#make -j 4 #VERBOSE=1

use_nvprof="false"
use_nsight="false"
#use_nsight="true"
#use_nvprof="true"

make -j 4
if [ use_nvprof == "true" ]; then

nvprof --analysis-metrics -f -o ../profile.nvprof ./test

elif [ $use_nsight == "true" ]; then

/apps/NVIDIA-HPC-SDK/20.9/Linux_ppc64le/2020/profilers/Nsight_Compute/ncu --set full -f -o ../profile ./test

else

  IS_DDT_OPEN=false
  if pidof -x $(ps cax | grep ddt) >/dev/null; then
        IS_DDT_OPEN=true
  fi
  if [ "$IS_DDT_OPEN" = true ]; then
    ddt --connect ./test
  else
    ./test
   fi

#time python ../TestMonarch.py

#cmake . -DENABLE_CSR=OFF
#make VERBOSE=1 CPPFLAGS="-DCSR" CXXFLAGS="-DCSR" CFLAGS="-DCSR" CUDA="-DCSR" CUDAFLAGS="-DCSR"
#make VERBOSE=1
#make -j 4


fi