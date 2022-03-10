#include "SlabHash/SlabPool.cuh"

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