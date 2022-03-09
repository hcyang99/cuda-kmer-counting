#pragma once
#include "GpuHashtable/GpuHashtable.cuh"
#include "SlabPool.cuh"
#include "utils/JobQueue.cuh"

/**
 * @brief SlabHash, a separate chaining GPU hashtable, should be allocated on shared memory of each block
 */
class SlabHash : public GpuHashtable
{
    public:
    SlabHash(uint32_t* d, uint32_t n, JobQueue* j, SlabPool* all_pools, SlabPool* own_pool)
        :GpuHashtable(d, n, j), all_slab_pools(all_pools), mem_pool(own_pool), new_slab(nullptr) {}

    protected:
    SlabPool* const all_slab_pools; // Array of all Slab pools
    SlabPool* mem_pool;             // Current memory pool allocating from; Initally self, when full, some other block's pool
    uint32_t* new_slab;

    /**
     * @brief Get a new SlabPool when current is full, if every pool is full, will set `mem_pool` to NULL
     * @param seed Seed for random pool selection
     */
    void get_new_pool(uint32_t seed);

    /**
     * @brief Allocate a new slab from current pool, store in `new_slab`, switch to new pool if necessary
     * @param seed Seed for random pool selection
     */
    void allocate_slab(uint32_t seed);

    /**
     * @brief Insert/increment the key in SlabHash
     */
    virtual void process(const Compressed128Mer& key) override;
};

