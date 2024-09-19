cmake_minimum_required(VERSION 3.10)
project(MultiGPUOpenMPCAMP)

# Enable CUDA language
enable_language(CUDA)

# Add definitions
add_definitions(-DCSR)
# add_definitions(-DDEV_CSP) #development
# add_definitions(-DPA) # development
# add_definitions(-DCUID) # slower

# Find MPI package
find_package(MPI REQUIRED)

# Set CUDA flags
set(STD_CUDA_FLAGS "-g -O3 --default-stream per-thread") # if debug assembly(slow): -G

# Set CUDA source files
set(CAMP_CUDA_SRC libsolv.cu)
set_source_files_properties(${CAMP_CUDA_SRC} PROPERTIES COMPILE_FLAGS ${STD_CUDA_FLAGS})
set_source_files_properties(${CAMP_CUDA_SRC} PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${CAMP_CUDA_SRC} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

# Create a library from CUDA sources
set(CAMP_LIB_SRC ${CAMP_CUDA_SRC})
add_library(mainlib ${CAMP_LIB_SRC})

# Link the CUDA and MPI libraries to the mainlib
target_link_libraries(mainlib ${CUDA_LIBRARIES} MPI::MPI_CXX)

# Set library properties
set_target_properties(mainlib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(mainlib PROPERTIES CUDA_ARCHITECTURES "70")
set_target_properties(mainlib PROPERTIES
        OUTPUT_NAME camp
        PREFIX lib
        VERSION ${PACKAGE_VERSION}
        SOVERSION ${PACKAGE_VERSION}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR})

# Create the main executable
add_executable(main main.cu)

# Link the main executable to the mainlib and MPI
target_link_libraries(main mainlib MPI::MPI_CXX)

# Set properties for the main executable
set_target_properties(main PROPERTIES COMPILE_FLAGS ${STD_CUDA_FLAGS})
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "70")
