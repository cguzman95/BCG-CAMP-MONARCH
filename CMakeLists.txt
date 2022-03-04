cmake_minimum_required(VERSION 3.10)

project(MultiGPUOpenMPCAMP)

enable_language(CUDA)
enable_language(CXX)

#SET(CMAKE_C_FLAGS "-fopenacc -ta=tesla -Minfo=accel")

#find_package(OpenACC REQUIRED)

#FindOpenACC()

set(OpenACC_FLAGS "-acc -ta=nvidia -Minfo=accel") #-ta=tesla
add_executable(test test.cpp)
#set_target_properties(test PROPERTIES COMPILE_FLAGS ${OpenACC_FLAGS})

#target_compile_options(test PRIVATE ${OpenACC_C_OPTIONS})

#add_executable(test test.cu)
#set(STD_CUDA_FLAGS "-lineinfo -arch=compute_70 -code=sm_70")
#set_target_properties(test PROPERTIES COMPILE_FLAGS ${STD_CUDA_FLAGS})
