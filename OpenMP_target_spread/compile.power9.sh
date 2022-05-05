#MONARCH P9 compilation

module load CMake/3.15.4
module load CUDA/10.1

export LD_LIBRARY_PATH=/gpfs/scratch/bsc00/bsc00806/llvm-spread-power/lib:/gpfs/scratch/bsc00/bsc00806/llvm-spread-power/projects/openmp/libomptarget:$LD_LIBRARY_PATH

rm -rf build
mkdir build
cd build

cmake -DCMAKE_C_COMPILER=/gpfs/scratch/bsc00/bsc00806/llvm-spread-power/bin/clang -DCMAKE_CXX_COMPILER=/gpfs/scratch/bsc00/bsc00806/llvm-spread-power/bin/clang++ ..

make -j 4 VERBOSE=1
cd ../..
