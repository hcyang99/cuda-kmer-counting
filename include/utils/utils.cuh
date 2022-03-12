#pragma once
#include <cstdint>
#include <iostream>
#include <stdio.h>
#include <assert.h>

#define CUDA_CHECK_ERROR(x) do {auto err = x; if (err != cudaSuccess) {std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ", " << __LINE__ << std::endl; std::exit(1);}} while (0)

namespace utils
{
    __device__ 
    int global_thread_id()
    {
        int tx = threadIdx.x;
        int bx = blockIdx.x;
        int bs = blockDim.x;

        return tx + bx * bs;
    }

    __host__ 
    int blockSize()
    {
        return 128;
    }

    __host__ 
    int gridSize()
    {
        return 128;
    }

    __host__ 
    int batchSize()
    {
        return 128;
    }

    __host__ 
    float slabFactor()
    {
        return 0.05;
    }

    __host__ 
    float OAFactor()
    {
        return 2.0;
    }

    // CUDA is little-endian
    union byte_32
    {
        char c[32];
        uint32_t u32[8];

        __device__ byte_32() : u32() {}

        __device__
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

        __device__ byte_4() : u32() {}
    };

    using Compressed128Mer = byte_32;

    /**
     * @brief Load Compressed128Mer from global string
     * @param data global string
     * @param offset
     * @param out the 128Mer
     */
    __device__ 
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

    void printGpuProperties () {
    int nDevices;

    // Store the number of available GPU device in nDevicess
    cudaError_t err = cudaGetDeviceCount(&nDevices);

    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaGetDeviceCount failed!\n");
        exit(1);
    }

    // For each GPU device found, print the information (memory, bandwidth etc.)
    // about the device
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Device memory: %lu\n", prop.totalGlobalMem);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
}
}

