cmake_minimum_required(VERSION 3.10)

project(cuda-gpu-hashtable)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
find_package(CUDA 10.2 REQUIRED)

project(xact-build LANGUAGES CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_52")

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(info src/info.cu src/utils/utils.cu)
add_executable(slab src/slab.cu src/utils/utils.cu)
add_executable(wc src/wc.cu src/utils/utils.cu)
add_executable(gerbil src/gerbil.cu src/utils/utils.cu)
