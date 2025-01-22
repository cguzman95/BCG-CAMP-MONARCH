#!/usr/bin/env bash
set -e
cd build
make -j 4 VERBOSE=1
#Disable warning
unset I_MPI_PMI_LIBRARY

run(){
  exec_str=""
  IS_DDT_OPEN=false
  if pidof -x $(ps cax | grep forge) >/dev/null; then
    IS_DDT_OPEN=true
  fi
  if [ "$IS_DDT_OPEN" = true ]; then
    echo "OPEN"
    exec_str="ddt --connect "
  fi
  exec_str="${exec_str} mpirun -np 1 ./main"
  echo $exec_str
  $exec_str
}

run_ncu(){
  #profile must run in allocated node
  /apps/ACC/NVIDIA-HPC-SDK/23.9/Linux_x86_64/23.9/profilers/Nsight_Compute/ncu --target-processes application-only --set full -f -o ../profile ./main
}

run_nsys(){
  #profile must run in allocated node
  /apps/ACC/NVIDIA-HPC-SDK/23.9/Linux_x86_64/23.9/profilers/Nsight_Systems/bin/nsys profile -f true -o ../profile ./main
}

run
#run_ncu
#run_nsys
