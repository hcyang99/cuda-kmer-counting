#pragma once
#include "utils/utils.cuh"

/**
 * @brief Block-owned slab memory pool, should be allocated in global memory (can share with another block in case own pool is full)
 */
class SlabPool
{
    public:
    SlabPool(uint32_t* buf, uint32_t sz)
        : mem(buf), mem_pool_head(0), mem_pool_size(sz) {}

    /**
     * @brief Thread-safely allocates Slabs (memory chunks of 512 B); should only be called in thread 0 of each block
     * @return pointer to newly allocated slab; nullptr at failure
     */
    uint32_t* allocate();

    /**
     * @return True if current pool is full
     */
    bool full() const;

    protected:
    uint32_t* mem;              // Pool's buffer
    uint32_t mem_pool_head;     // Pool's head (in uint32_t)
    uint32_t mem_pool_size;     // Pool's size (in uint32_t)
};