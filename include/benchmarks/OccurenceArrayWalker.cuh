#pragma once
#include "utils/utils.cuh"
#include "utils/JobQueue.cuh"
#include "benchmarks/Stats.cuh"

class OccurenceArrayWalker
{
    public:
    __device__ OccurenceArrayWalker(uint32_t* d, Stats* g_stats, JobQueue* j)
        : data(d), global_stats(g_stats), job_queue(j), job_begin(), job_end(), block_stats() {}

    __device__ void run();

    protected:
    uint32_t* const data;
    Stats* const global_stats;
    JobQueue* const job_queue;
    uint32_t job_begin;
    uint32_t job_end;
    Stats block_stats;

    __device__ void process_batch();

    __device__ void get_job_batch();
};

__device__
void OccurenceArrayWalker::process_batch()
{
    int tx = threadIdx.x;
    int bs = blockDim.x;

    for (uint32_t i = job_begin + tx; i < job_end; i += bs)
    {
        block_stats.increment(data[i], 1);
    }
}

__device__
void OccurenceArrayWalker::get_job_batch()
{
    uint32_t start = job_queue->dispatch();
    uint32_t size = job_queue->dispatch_size;
    uint32_t end = start + size;
    if (end > job_queue->total_jobs)
        end = job_queue->total_jobs;
    if (start >= job_queue->total_jobs)
        start = 0xFFFFFFFFUL;   // Flag to exit
    
    job_begin = start;
    job_end = end;
}

__device__
void OccurenceArrayWalker::run()
{
    int tx = threadIdx.x;

    while (true)
    {
        if (tx == 0)
            get_job_batch();
        __syncthreads();

        if (job_begin >= job_end)
            break;  // no job to do
        
        process_batch();
    }

    // merge block stats to the global stat
    *global_stats += block_stats;
}

__global__ 
void occurence_array_walk_kernel(uint32_t* array, Stats* global_stats, JobQueue* j)
{
    __shared__ char s[sizeof(OccurenceArrayWalker)];

    OccurenceArrayWalker* h = (OccurenceArrayWalker*) s;

    int tx = threadIdx.x;
    if (tx == 0)
    {
        new (h) OccurenceArrayWalker(array, global_stats, j);
    }
    __syncthreads();

    h->run();
}

__host__ 
void occurence_table_walk(uint32_t* array, size_t size, Stats* stats, uint32_t* out)
{
    JobQueue* d_jobs = new_job(size, 2048);
    occurence_array_walk_kernel<<<1, 32>>>
        (array, stats, d_jobs);
    CUDA_CHECK_ERROR(cudaMemcpy(out, (uint32_t*)stats, 128 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaFree(d_jobs));
    CUDA_CHECK_ERROR(cudaFree(stats));
}