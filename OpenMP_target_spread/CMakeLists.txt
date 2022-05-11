cmake_minimum_required(VERSION 3.10)

project(MultiGPUOpenMPCAMP)

INCLUDE(${PROJECT_SOURCE_DIR}/data/CMakeLists.txt)

#set(LLVM "/gpfs/scratch/bsc00/bsc00806/llvm-official-power")
set(LLVM "/gpfs/scratch/bsc00/bsc00806/llvm-spread-power")
#set(LLVM "/gpfs/projects/bsc00/bsc00806/llvm-build-power")
set(OMPTARGETS "nvptx64")
set(EXTRA_FLAGS "--gcc-toolchain=/apps/GCC/7.3.0")

#set(LLVM_IR_FLAGS "-O3 -Xclang -disable-llvm-passes -emit-llvm -c -S")
set(LLVM_IR_FLAGS "-O3")
include_directories(./)
include_directories(${LLVM}/projects/openmp/runtime/src)

#add_compile_definitions(SCALE=100000 NUM_DEVICES=1 DEVICE_LIST=0)

# weak scaling
#add_compile_definitions(SCALE=200000 NUM_DEVICES=2 DEVICE_LIST=0,1)
#add_compile_definitions(SCALE=400000 NUM_DEVICES=4 DEVICE_LIST=0,1,2,3)

# strong scaling
#add_compile_definitions(SCALE=100000 NUM_DEVICES=2 DEVICE_LIST=0,1)
add_compile_definitions(SCALE=100000 NUM_DEVICES=4 DEVICE_LIST=0,1,2,3)


set(CMAKE_CXX_FLAGS "${LLVM_IR_FLAGS} -Wall -g -fopenmp -fopenmp-targets=${OMPTARGETS} --libomptarget-nvptx-bc-path=${LLVM}/projects/openmp/libomptarget")
set(CMAKE_EXE_LINKER_FLAGS "-g -L${LLVM}/lib -L${LLVM}/projects/openmp/libomptarget -lm ${EXTRA_FLAGS}")

add_executable(test test.cpp)


