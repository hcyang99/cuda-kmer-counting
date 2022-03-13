#pragma once
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <string>
#include <stdio.h>
#include <assert.h>

#define CUDA_CHECK_ERROR(x) do {auto err = x; if (err != cudaSuccess) {std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ", " << __LINE__ << std::endl; std::exit(1);}} while (0)

namespace utils
{
    __device__ __forceinline__ 
    int global_thread_id()
    {
        int tx = threadIdx.x;
        int bx = blockIdx.x;
        int bs = blockDim.x;

        return tx + bx * bs;
    }

    class BlockSize
    {
        static int value;

        public:
        static int get();
        static void set(int v);
    };

    class GridSize
    {
        static int value;

        public:
        static int get();
        static void set(int v);
    };

    class BatchSize
    {
        static int value;

        public:
        static int get();
        static void set(int v);
    };

    class SlabFactor
    {
        static float value;

        public:
        static float get();
        static void set(int v);
    };

    class OAFactor
    {
        static float value;

        public:
        static float get();
        static void set(int v);
    };

    // CUDA is little-endian
    union byte_32
    {
        char c[32];
        uint32_t u32[8];

        __device__ __forceinline__ byte_32() : u32() {}

        __device__ __forceinline__
        bool operator==(const byte_32& other) const
        {
            for (int i = 0; i < 8; ++i)
            {
                if (u32[i] != other.u32[i])
                    return false;
            }
            return true;
        }
    };

    union byte_4
    {
        char c[4];
        uint32_t u32;

        __device__ __forceinline__ byte_4() : u32() {}
    };

    using Compressed128Mer = byte_32;

    /**
     * @brief Load Compressed128Mer from global string
     * @param data global string
     * @param offset
     * @param out the 128Mer
     */
    __device__ __forceinline__ 
    void Read128Mer(uint32_t* data, uint32_t offset, Compressed128Mer& out)
    {
        int tx = threadIdx.x;
        if (tx < 8)
        {
            uint32_t index = offset / 16 + tx;
            uint32_t shift_1 = offset % 16 * 2;
            uint32_t shift_2 = (16 - shift_1) * 2;
            out.u32[tx] = (data[index] >> shift_1) | (data[index + 1] << shift_2);
        }
        __syncthreads();
    }

    void printGpuProperties();
}

