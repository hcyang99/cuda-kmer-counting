#pragma once
#include "GpuHashtable/GpuHashtable.cuh"

/**
 * @brief Abstract base class for open-addressing GPU hashtables
 */
class OAGpuHashtable : public GpuHashtable
{
    protected:
    __device__ OAGpuHashtable(uint32_t* ref, uint32_t* d, uint32_t n, JobQueue* j)
        :GpuHashtable(ref, d, n, j) {}
    
    
    /**
     * @brief Get next address using quadratic probing
     * @param key The query key
     * @return next address
     */
    __device__ uint32_t* get_next_probe_pos(uint32_t probe_count) const
    {
        uint32_t curr_idx = (probe_pos - data) / 128;
        uint32_t new_idx = (curr_idx + 2 * probe_count - 1) % num_buckets;
        return data + new_idx * 128;
    }
};
