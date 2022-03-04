#MONARCH P9 compilation

module load GCC/7.3.0-2.30
module load OpenMPI/3.1.0-GCC-7.3.0-2.30
module load CMake/3.15.3-GCCcore-7.3.0
module load CUDA/10.1.105-ES
module load Python/3.7.0-foss-2018b
module load pgi

rm -rf build
mkdir build
cd build

cmake ..
#cmake ../ -DOpenACC_ACCEL_TARGET==tesla:cc60 -DCMAKE_C_COMPILER=pgcc -DCMAKE_CXX_COMPILER=pgc++

make -j 4 VERBOSE=1
cd ../..