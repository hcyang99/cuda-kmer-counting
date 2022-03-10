cmake_minimum_required(VERSION 3.10)

project(cuda-gpu-hashtable)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
find_package(CUDA 10.2 REQUIRED)

project(xact-build LANGUAGES CUDA)

set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CUDA_FLAGS "${NVCC_COMPILE_FLAGS} -dlto")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-dc>")


include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable (slab
    src/benchmarks/slab.cu
    src/benchmarks/HashtableWalker.cu 
    src/benchmarks/Stats.cu 
    src/GpuHashtable/GpuHashtable.cu 
    src/SlabHash/SlabHash.cu 
    src/SlabHash/SlabPool.cu 
    src/utils/Fasta.cu 
    src/utils/JobQueue.cu
    )

target_link_libraries(slab)

