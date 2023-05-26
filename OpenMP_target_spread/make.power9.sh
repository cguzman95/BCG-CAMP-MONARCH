#!/usr/bin/env bash
export LD_LIBRARY_PATH=/gpfs/scratch/bsc00/bsc00806/llvm-spread-power/lib:/gpfs/scratch/bsc00/bsc00806/llvm-spread-power/projects/openmp/libomptarget:$LD_LIBRARY_PATH

cd build
make -j 4 #VERBOSE=1




#export OMP_PLACES="{0:8:1}:10:8";export OMP_PROC_BIND=spread;export OMP_NUM_THREADS=2;
#export OMP_PLACES="{0:8:1}:10:8";export OMP_PROC_BIND=spread;export OMP_NUM_THREADS=4;
export OMP_PLACES="{0:8:1}:10:8";export OMP_PROC_BIND=spread;export OMP_NUM_THREADS=8;

time numactl -N 0 ./test

#/apps/NVIDIA-HPC-SDK/21.3/Linux_ppc64le/21.3/compilers/bin/nsys profile --stats=true --force-overwrite=true --cuda-memory-usage=true -o target_spread_1_devices.nsys.txt numactl -N 0 ./test

# weak scaling
#/apps/NVIDIA-HPC-SDK/21.3/Linux_ppc64le/21.3/compilers/bin/nsys profile --stats=true --force-overwrite=true --cuda-memory-usage=true -o target_spread_2_devices_w.nsys.txt numactl -N 0 ./test
#/apps/NVIDIA-HPC-SDK/21.3/Linux_ppc64le/21.3/compilers/bin/nsys profile --stats=true --force-overwrite=true --cuda-memory-usage=true -o target_spread_4_devices_w.nsys.txt numactl -N 0 ./test

# strong scaling
#/apps/NVIDIA-HPC-SDK/21.3/Linux_ppc64le/21.3/compilers/bin/nsys profile --stats=true --force-overwrite=true --cuda-memory-usage=true -o target_spread_2_devices_s.nsys.txt numactl -N 0 ./test
/apps/NVIDIA-HPC-SDK/21.3/Linux_ppc64le/21.3/compilers/bin/nsys profile --stats=true --force-overwrite=true --cuda-memory-usage=true -o target_spread_4_devices_s.nsys.txt numactl -N 0 ./test
