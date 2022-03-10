#pragma once
#include "utils/utils.cuh"

/**
 * @brief Singleton class for job dispatching to long-running blocks, should reside on global memory
 */
class JobQueue
{
    protected:
    uint32_t current_head;

    public:
    const uint32_t total_jobs;
    const uint32_t dispatch_size;

    __device__ JobQueue(uint32_t size, uint32_t step)
        : total_jobs(size), dispatch_size(step), current_head(0) {}
    
    /**
     * @brief Thread-safely dispatch jobs; should only be called in thread 0 of each block
     * @return Head of dispatched jobs, must be compared with `total_jobs` and `dispatch_size`
     */
    __device__ uint32_t dispatch();
};

JobQueue* new_job(uint32_t size, uint32_t step);