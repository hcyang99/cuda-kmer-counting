#include "HashtableWalker.cuh"

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