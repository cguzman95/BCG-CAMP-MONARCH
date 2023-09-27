#!/usr/bin/env bash

cd build
make -j 4

run(){
  IS_DDT_OPEN=false
  if pidof -x $(ps cax | grep ddt) >/dev/null; then
        IS_DDT_OPEN=true
  fi
  if [ "$IS_DDT_OPEN" = true ]; then
    ddt --connect ./test
  else
    ./test
   fi
}

run_nsight(){
  /apps/NVIDIA-HPC-SDK/21.3/Linux_ppc64le/21.3/profilers/Nsight_Compute/ncu --set full -f -o ../profile ./test
  #/apps/NVIDIA-HPC-SDK/22.3/Linux_ppc64le/22.3/profilers/Nsight_Compute/ncu
}

run_nvprof(){
nvprof --analysis-metrics -f -o ../profile.nvprof ./test
}

run
#run_nsight
#run_nvprof