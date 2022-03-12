#pragma once
#include "GpuHashtable/OAGpuHashtable.cuh"

/**
 * @brief A normal open-addressing hashtable
 */
class WarpCore : public OAGpuHashtable
{
    public:
    __device__ WarpCore(uint32_t* ref, uint32_t* d, uint32_t n, JobQueue* j)
        :OAGpuHashtable(ref, d, n, j) {}
    
    protected:
    /**
     * @brief Insert/increment the key in WarpCore
     */
    __device__ virtual void process(const Compressed128Mer& key) override;
};

__device__
void WarpCore::process(const Compressed128Mer& key)
{
    int tx = threadIdx.x;
    uint32_t probe_count = 0;
    if (tx == 0)
    {
        uint32_t h = hash(key);
        probe_pos = data + h % num_buckets * 128;
        status = ProbeStatus::PROBE_CURRENT;
    }
    __syncthreads();

    while (true)
    {
        if (status == ProbeStatus::PROBE_CURRENT)
        {
            simd_probe(probe_pos, key);
        } 
        else if (status == ProbeStatus::SUCCEESS)
        {
            return;
        }
        else if (status == ProbeStatus::PROBE_NEXT)
        {
            if (tx == 0)
            {
                ++probe_count;
                probe_pos = get_next_probe_pos(probe_count);
                status = ProbeStatus::PROBE_CURRENT;
            }
            __syncthreads();
        }
    }
}