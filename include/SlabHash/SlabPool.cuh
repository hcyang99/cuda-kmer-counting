#pragma once
#include "utils/utils.cuh"

/**
 * @brief Block-owned slab memory pool, should be allocated in global memory (can share with another block in case own pool is full)
 */
class SlabPool
{
    public:
    __device__ SlabPool(uint32_t* buf, uint32_t sz)
        : mem(buf), mem_pool_head(0), mem_pool_size(sz) {}

    /**
     * @brief Thread-safely allocates Slabs (memory chunks of 512 B); should only be called in thread 0 of each block
     * @return pointer to newly allocated slab; nullptr at failure
     */
    __device__ uint32_t* allocate();

    /**
     * @return True if current pool is full
     */
    __device__ bool full() const;

    protected:
    uint32_t* mem;              // Pool's buffer
    uint32_t mem_pool_head;     // Pool's head (in uint32_t)
    uint32_t mem_pool_size;     // Pool's size (in uint32_t)
};

__device__ 
uint32_t* SlabPool::allocate()
{
    uint32_t old_head = atomicAdd(&mem_pool_head, 128UL);
    if (old_head >= mem_pool_size)
        return nullptr;     // allocation failed: out of memory
    return mem + old_head;
}

__device__ 
bool SlabPool::full() const
{
    return mem_pool_head < mem_pool_size;
}