#MONARCH P9 compilation

#If GCC=8.3.0
#module load CUDA/10.1.243-GCC-8.3.0
#module load CMake/3.15.3-GCCcore-8.3.0
#module load Python/3.7.4-GCCcore-8.3.0
#module load matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
#module load OpenMPI/3.1.4-GCC-8.3.0

#else
#module load GCC/7.3.0-2.30
#module load OpenMPI/3.1.0-GCC-7.3.0-2.30
#module load CMake/3.15.3-GCCcore-7.3.0
#module load CUDA/10.1.105-ES


rm -rf build
mkdir build
cd build
mkdir out

cmake ..

make -j 4 VERBOSE=1
./test
cd ../..