#!/usr/bin/env bash
set -e
cd build
make -j 4 VERBOSE=1

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
  /apps/NVIDIA-HPC-SDK/20.9/Linux_ppc64le/2020/profilers/Nsight_Compute/ncu --set full -f -o ../profile ./test
}

run_nvprof(){
nvprof --analysis-metrics -f -o ../profile.nvprof ./test
}

run
#run_nsight
#run_nvprof