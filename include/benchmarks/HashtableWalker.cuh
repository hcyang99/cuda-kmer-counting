#pragma once
#include "utils/utils.cuh"
#include "utils/JobQueue.cuh"
#include "benchmarks/Stats.cuh"

/**
 * @brief Hashtable walker class; should reside in shared memory; assuming 32 threads per block
 */
class HashtableWalker
{
    public:
    __device__ HashtableWalker(uint32_t* d, Stats* g_stats, JobQueue* j)
        : data(d), global_stats(g_stats), job_queue(j), job_begin(), job_end(), block_stats(), next_window(nullptr) {}

    __device__ void run();

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
    __device__ void process_window(uint32_t* window);

    /**
     * @brief Get dispatch from `job_queue`; writes `job_begin` and `job_end`; 
     * should only be called from thread 0 of each block
     */
    __device__ void get_job_batch();
};

/**
 * @brief Do a walk across the hashtable and generate stats
 * @param table Pointer to hashtable buffer
 * @param num_buckets
 * @param out On host
 */
void hashtable_walk(uint32_t* table, uint32_t num_buckets, uint32_t* out);