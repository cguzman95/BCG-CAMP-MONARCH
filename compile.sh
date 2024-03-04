#MONARCH P9 compilation

#module load GCC/7.3.0-2.30
#module load OpenMPI/3.1.0-GCC-7.3.0-2.30
#module load CMake/3.15.3-GCCcore-7.3.0
#module load CUDA/10.1.105-ES

rm -rf build
mkdir build
cd build
mkdir -p out
cmake ..
cd ../
./run.sh
