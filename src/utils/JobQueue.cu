#include "JobQueue.cuh"

__device__ __forceinline__
uint32_t JobQueue::dispatch()
{
    return atomicAdd(&current_head, dispatch_size);
}