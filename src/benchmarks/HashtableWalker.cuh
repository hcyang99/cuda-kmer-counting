#pragma once
#include "utils/utils.cuh"
#include "utils/JobQueue.cuh"
#include "Stats.cuh"

/**
 * @brief Hashtable walker class; should reside in shared memory; assuming 32 threads per block
 */
class HashtableWalker
{
    public:
    HashtableWalker(uint32_t* d, Stats* g_stats, JobQueue* j)
        : data(d), global_stats(g_stats), job_queue(j), job_begin(), job_end(), block_stats(), next_window(nullptr) {}

    void run();

    protected:
    uint32_t* const data;
    Stats* const global_stats;
    JobQueue* const job_queue;
    uint32_t job_begin;
    uint32_t job_end;
    Stats block_stats;
    uint32_t* next_window;  // for slabs only

    /**
     * @brief process current window and all subsequently chained windows (Slab list)
     */
    void process_window(uint32_t* window);

    /**
     * @brief Get dispatch from `job_queue`; writes `job_begin` and `job_end`; 
     * should only be called from thread 0 of each block
     */
    void get_job_batch();
};