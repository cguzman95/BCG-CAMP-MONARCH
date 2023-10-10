cmake_minimum_required(VERSION 3.10)
project(MultiGPUOpenMPCAMP)

add_definitions(-DCSR)
#add_definitions(-DCUID) #slower

enable_language(CUDA)

set(STD_CUDA_FLAGS "-g -arch=compute_70 -code=sm_70") #if debug: -g -G

set(CAMP_CUDA_SRC libsolv.cu)
set_source_files_properties(${CAMP_CUDA_SRC} PROPERTIES COMPILE_FLAGS
        ${STD_CUDA_FLAGS})
set_source_files_properties(${CAMP_CUDA_SRC} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${CAMP_CUDA_SRC} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

set(CAMP_LIB_SRC ${CAMP_CUDA_SRC})
add_library(camplib ${CAMP_LIB_SRC})
target_link_libraries(camplib ${CUDA_LIBRARIES})
set_target_properties(camplib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(camplib PROPERTIES CUDA_ARCHITECTURES "70")
set_target_properties(camplib PROPERTIES
        OUTPUT_NAME camp
        PREFIX lib
        VERSION ${PACKAGE_VERSION}
        SOVERSION ${PACKAGE_VERSION}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(test test.cu)
target_link_libraries(test camplib)

set_target_properties(test PROPERTIES COMPILE_FLAGS ${STD_CUDA_FLAGS})
set_target_properties(test PROPERTIES CUDA_ARCHITECTURES "70")


