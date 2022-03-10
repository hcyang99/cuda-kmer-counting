#pragma once
#include "GpuHashtable/OAGpuHashtable.cuh"

/**
 * @brief A normal open-addressing hashtable
 */
class WarpCore : public OAGpuHashtable
{
    public:
    WarpCore(uint32_t* ref, uint32_t* d, uint32_t n, JobQueue* j)
        :OAGpuHashtable(ref, d, n, j) {}
    
    protected:
    /**
     * @brief Insert/increment the key in WarpCore
     */
    virtual void process(const Compressed128Mer& key) override;
};