cmake_minimum_required(VERSION 2.6.4)

project(MultiGPUOpenMPCAMP)

enable_language(CUDA)

add_executable(test test.cu)

set(STD_CUDA_FLAGS "-lineinfo -arch=compute_70 -code=sm_70")

set_target_properties(test PROPERTIES COMPILE_FLAGS ${STD_CUDA_FLAGS})
