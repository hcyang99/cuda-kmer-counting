#include "benchmarks/HashtableWalker.cuh"

__device__
void HashtableWalker::process_window(uint32_t* window)
{
    int tx = threadIdx.x;
    if (tx < 14)
    {
        int offset = tx * 9 + 8;
        int count = window[offset];
        block_stats.increment(count, 1);
    }
    else if (tx == 31)
    {
        next_window = *(uint32_t**)(window + 126);
    }
    __syncthreads();

    if (next_window == nullptr)
        return;

    process_window(next_window);
}

__device__
void HashtableWalker::get_job_batch()
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
void HashtableWalker::run()
{
    int tx = threadIdx.x;

    while (true)
    {
        if (tx == 0)
            get_job_batch();
        __syncthreads();

        if (job_begin >= job_end)
            break;  // no job to do
        
        for (uint32_t i = job_begin; i < job_end; ++i)
        {
            process_window(data + i * 128);
        }
    }

    // merge block stats to the global stat
    *global_stats += block_stats;
}

__global__
void hashtable_walk_kernel(uint32_t* table, Stats* global_stats, JobQueue* j)
{
    __shared__ uint32_t s[1024];

    HashtableWalker* h_ptr = (HashtableWalker*)s;

    int tx = threadIdx.x;
    if (tx == 0)
    {
        new (h_ptr) HashtableWalker(table, global_stats, j);
    }
    __syncthreads();

    h_ptr->run();
}

__host__
void hashtable_walk(uint32_t* table, uint32_t num_buckets, uint32_t* out)
{
    JobQueue* d_jobs = new_job(num_buckets, utils::batchSize());
    Stats* d_stats = new_stats();
    hashtable_walk_kernel<<<utils::gridSize(), 32>>>(table, d_stats, d_jobs);
    CUDA_CHECK_ERROR(cudaMemcpy(out, (uint32_t*)d_stats, 128 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_jobs);
    cudaFree(d_stats);
}