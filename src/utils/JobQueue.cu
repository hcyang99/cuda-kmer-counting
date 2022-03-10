#include "utils/JobQueue.cuh"

__device__ 
uint32_t JobQueue::dispatch()
{
    return atomicAdd(&current_head, dispatch_size);
}

__global__
void new_job_kernel(JobQueue* j, uint32_t size, uint32_t step)
{
    if (utils::global_thread_id() == 0)
    {
        new (j) JobQueue(size, step);
    }
}

__host__
JobQueue* new_job(uint32_t size, uint32_t step)
{
    JobQueue* d_jobs;
    CUDA_CHECK_ERROR(cudaMalloc(&d_jobs, (sizeof(JobQueue) / 256 + 1) * 256));
    new_job_kernel<<<1, 32>>>(d_jobs, size, step);
    return d_jobs;
}