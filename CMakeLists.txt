cmake_minimum_required(VERSION 3.10)
project(MultiGPUOpenMPCAMP)

add_definitions(-DCSR)
#add_definitions(-DDEV_CSP)
#add_definitions(-DPA) #development
#add_definitions(-DCUID) #slower

find_package(MPI)
#find_path(MPI_INCLUDE_DIR mpi.h #needed for mn5
#        PATHS
#        /apps/ACC/OPENMPI/4.1.5/GCC/include)
#include_directories(${MPI_INCLUDE_DIR})#needed for mn5
#enable_language(MPI)
#include_directories(${MPI_INCLUDE_PATH})
enable_language(CUDA)

set(STD_CUDA_FLAGS "-g") #if debug: -g -G

set(CAMP_CUDA_SRC libsolv.cu)
set_source_files_properties(${CAMP_CUDA_SRC} PROPERTIES COMPILE_FLAGS
        ${STD_CUDA_FLAGS})
set_source_files_properties(${CAMP_CUDA_SRC} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${CAMP_CUDA_SRC} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

set(CAMP_LIB_SRC ${CAMP_CUDA_SRC})
add_library(bcglib ${CAMP_LIB_SRC})
target_link_libraries(bcglib ${CUDA_LIBRARIES})
set_target_properties(bcglib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(bcglib PROPERTIES CUDA_ARCHITECTURES "70")
set_target_properties(bcglib PROPERTIES
        OUTPUT_NAME camp
        PREFIX lib
        VERSION ${PACKAGE_VERSION}
        SOVERSION ${PACKAGE_VERSION}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(test test.cu)
target_link_libraries(test bcglib)

set_target_properties(test PROPERTIES COMPILE_FLAGS ${STD_CUDA_FLAGS})
set_target_properties(test PROPERTIES CUDA_ARCHITECTURES "70")


