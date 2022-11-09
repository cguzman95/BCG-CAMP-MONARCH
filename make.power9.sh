#!/usr/bin/env bash

echo "Start make.power9.sh"

cd build
#make -j 4 #VERBOSE=1

use_nvprof="false"
use_nsight="false"
#use_nsight="true"
#use_nvprof="true"

if [ use_nvprof == "true" ]; then

nvprof --analysis-metrics -f -o ../profile.nvprof ./test

elif [ $use_nsight == "true" ]; then

/apps/NVIDIA-HPC-SDK/21.3/Linux_ppc64le/21.3/profilers/Nsight_Compute/ncu --set full -f -o ../profile ./test

#cd /apps/NVIDIA-HPC-SDK/22.3/Linux_ppc64le/22.3/profilers/Nsight_Compute

#module load NVHPC
#ncu --set full -f -o ../profile ./test

else

time python ../TestMonarch.py
#make -j 4
#./test

#cmake . -DENABLE_CSR=OFF
#make VERBOSE=1 CPPFLAGS="-DCSR" CXXFLAGS="-DCSR" CFLAGS="-DCSR" CUDA="-DCSR" CUDAFLAGS="-DCSR"
#make VERBOSE=1
#make -j 4


fi