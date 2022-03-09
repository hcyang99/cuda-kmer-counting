#pragma once
#include "GpuHashtable.cuh"

/**
 * @brief Abstract base class for open-addressing GPU hashtables
 */
class OAGpuHashtable : public GpuHashtable
{
    protected:
    OAGpuHashtable(uint32_t* d, uint32_t n, JobQueue* j)
        :GpuHashtable(d, n, j) {}
    
    
    /**
     * @brief Get next address using double hashing
     * @param key The query key
     * @return next address
     */
    uint32_t* get_next_probe_pos(const Compressed128Mer& key) const
    {
        uint32_t curr_idx = probe_pos - data;
        uint32_t new_idx = hash(curr_idx + key.u32[5]) % num_buckets;
        return data + new_idx;
    }
};
