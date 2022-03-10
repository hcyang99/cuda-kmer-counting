#pragma once
#include "utils/utils.cuh"
#include "utils/JobQueue.cuh"

using utils::Compressed128Mer;

/**
 * @brief Common abstract base class for GPU hashtables
 */
class GpuHashtable
{
    public:
    __device__ void run();

    protected:
    enum class ProbeStatus {SUCCEESS, PROBE_NEXT, PROBE_CURRENT};

    uint32_t* reference;
    uint32_t* const data;
    const uint32_t num_buckets;
    JobQueue* const job_queue;
    uint32_t job_begin;
    uint32_t job_end;

    ProbeStatus status;
    uint32_t* probe_pos;
    uint32_t match_status[15];  // Value residing in shared memory, for matching status of 14 sub-warps
    uint32_t empty_status[15];  // Value residing in shared memory, for empty status of 14 sub-warps

    Compressed128Mer current_key;

    /**
     * @brief Protected constructor prevents instantiation; Derived classes should reside on shared memory
     */
    __device__ GpuHashtable(uint32_t* ref, uint32_t* d, uint32_t n, JobQueue* j)
        : reference(ref), data(d), num_buckets(n), job_queue(j), job_begin(0), job_end(0),
        status(ProbeStatus::PROBE_CURRENT), probe_pos(0), match_status(), empty_status(), current_key() {}

    /**
     * @brief Get dispatch from `job_queue`; writes `job_begin` and `job_end`; 
     * should only be called from thread 0 of each block
     */
    __device__ void get_job_batch();

    /**
     * @brief Calculates the hash of the key (Murmur Hash)
     */
    __device__ uint32_t hash(const Compressed128Mer& key) const;

    __device__ uint32_t hash(uint32_t value) const;

    /**
     * @brief Probe single SIMD window (512 B), do increment/insertion if possible, Assuming 128 threads.
     * Writes to `status`: SUCCESS: increment complete; PROBE: no free space found in current SIMD window
     * @param data Underlying hashtable 
     * @param key The query key
     * @return 
     */
    __device__ void simd_probe(uint32_t* data, const Compressed128Mer& key);

    /**
     * @brief Tries to insert new key into hashtable; should only be called from thread 0 of each block
     * @param kv_pair Key-value pair location in hashtable buffer
     * @return True: insertion succeeded; False: insertion failed
     */
    __device__ bool try_insert(uint32_t* kv_pair, const Compressed128Mer& key);

    /**
     * @brief Insert/increment the key in hashtable
     */
    __device__ virtual void process(const Compressed128Mer& key) = 0;
};