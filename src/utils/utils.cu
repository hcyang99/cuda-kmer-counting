#include "utils/utils.cuh"

namespace utils
{
    int utils::BlockSize::value = 128;
    int utils::GridSize::value = 128;
    int utils::BatchSize::value = 128;
    float utils::SlabFactor::value = 0.1;
    float utils::OAFactor::value = 2.0;

    void printGpuProperties () 
    {
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

    int BlockSize::get()
    {
        return BlockSize::value;
    }

    void BlockSize::set(int v)
    {
        BlockSize::value = v;
    }

    int GridSize::get()
    {
        return GridSize::value;
    }

    void GridSize::set(int v)
    {
        GridSize::value = v;
    }

    int BatchSize::get()
    {
        return BatchSize::value;
    }

    void BatchSize::set(int v)
    {
        BatchSize::value = v;
    }

    float SlabFactor::get()
    {
        return SlabFactor::value;
    }

    void SlabFactor::set(int v)
    {
        SlabFactor::value = v;
    }

    float OAFactor::get() 
    {
        return OAFactor::value;
    }

    void OAFactor::set(int v)
    {
        OAFactor::value = v;
    }
}

