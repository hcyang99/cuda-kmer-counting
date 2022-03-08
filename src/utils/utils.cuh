#pragma once
#include <cstdint>

#define CUDA_CHECK_ERROR(x) do {auto err = x; if (err != cudaSuccess) {std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ", " << __LINE__ << std::endl; std::exit(1);}} while (0)

namespace utils
{
    __device__ __forceinline__
    int global_thread_id()
    {
        int tx = threadIdx.x;
        int bx = blockIdx.x;
        int bs = blockDim.x;
        int gs = gridDim.x;

        return tx + bx * bs;
    }

    __host__ __forceinline__
    int blockSize()
    {
        return 128;
    }

    // CUDA is little-endian
    union byte_32
    {
        char c[32];
        uint32_t u32[8];

        byte_32() : u32() {}
    };

    union byte_4
    {
        char c[4];
        uint32_t u32;

        byte_4() : u32() {}
    };

    using Compressed128Mer = byte_32;

}

