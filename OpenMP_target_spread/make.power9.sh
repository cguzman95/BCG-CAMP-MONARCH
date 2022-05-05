#!/usr/bin/env bash
export LD_LIBRARY_PATH=/gpfs/scratch/bsc00/bsc00806/llvm-spread-power/lib:/gpfs/scratch/bsc00/bsc00806/llvm-spread-power/projects/openmp/libomptarget:$LD_LIBRARY_PATH

cd build
make -j 4 #VERBOSE=1


time ./test
/apps/NVIDIA-HPC-SDK/21.3/Linux_ppc64le/21.3/compilers/bin/nsys profile --stats=true --force-overwrite=true --cuda-memory-usage=true -o target_spread_two_devices.nsys.txt ./test

