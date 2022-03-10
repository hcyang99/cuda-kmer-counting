#include "benchmarks/Stats.cuh"

__device__
void Stats::increment(uint32_t id, uint32_t inc)
{
    if (inc == 0)
        return;
    if (id > 127)
        id = 127;
    atomicAdd(buckets + id, inc);
}

__device__
Stats& Stats::operator+=(const Stats& other)
{
    int tx = threadIdx.x;
    int bs = blockDim.x;
    if (tx == 0)
    {
        while (true)
        {
            if (atomicCAS(&lock, 0UL, 1UL) == 0)
                break;  // lock acquired
        }
    }
    for (int i = tx; i < 128; i += bs)
        buckets[tx] += other.buckets[tx];
    if (tx == 0)
        lock = 0;   // unlock
    return *this;
}

__global__ 
void new_stats_kernel(Stats* s)
{
    if (utils::global_thread_id() == 0)
    {
        new (s) Stats();
    }
}

__host__
Stats* new_stats()
{
    Stats* d_stats;
    CUDA_CHECK_ERROR(cudaMalloc(&d_stats, (sizeof(Stats) / 256 + 1) * 256));
    new_stats_kernel<<<1, 32>>>(d_stats);
    return d_stats;
}