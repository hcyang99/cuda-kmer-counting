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
    uint32_t match_status[15];          // Indicates matches of 14 sub-warps
    uint32_t empty_status[15];          // Value residing in shared memory, for empty status of 14 sub-warps
    uint32_t lock_status[15];           // Indicates if key-value pairs in table are locked (value == 0xFFFFFFFFU)

    Compressed128Mer current_key;

    /**
     * @brief Protected constructor prevents instantiation; Derived classes should reside on shared memory
     * @param ref The compressed input sequence to be counted
     * @param d Pointer to underlying hashtable data structure (buckets)
     * @param n Number of buckets
     * @param j The JobQueue object for job dispatching
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
    __device__ void simd_probe(volatile uint32_t* window, const Compressed128Mer& key);

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


__device__
void GpuHashtable::simd_probe(volatile uint32_t* window, const Compressed128Mer& key)
{
    int tx = threadIdx.x;
    int subwarp_idx = tx / 9;  // 14 (32B key, 4B value) pairs
    int sub_tx = tx % 9;        // 8 threads process the key, 1 thread processes the value
    if (tx < 126)               // let the last 2 threads rest
    {
        if (sub_tx == 0)   
        {
            match_status[subwarp_idx] = 1;  // initialize as matched
        }
    }
        
    __syncthreads();

    if (tx < 126)
    {
        uint32_t curr_segment = window[tx];
        if (sub_tx != 8)
        {
            if (curr_segment != key.u32[sub_tx])
            {
                match_status[subwarp_idx] = 0;  // set to 0 if not matching
            }
        }
        else
        {
            empty_status[subwarp_idx] = curr_segment == 0 ? 1 : 0;
            lock_status[subwarp_idx] = curr_segment == 0xFFFFFFFFU ? 1 : 0;
        }
    }
    __syncthreads();

    uint32_t match_mask, empty_mask, lock_mask;
    if (tx >= 0 && tx < 14)
    {
        match_mask = __ballot_sync(__activemask(), match_status[tx]);
        empty_mask = __ballot_sync(__activemask(), empty_status[tx]);
        lock_mask = __ballot_sync(__activemask(), lock_status[tx]);
    }

    if (tx == 0)
    {
        match_mask = match_mask & (~empty_mask);
        match_status[14] = match_mask;
        empty_status[14] = empty_mask;
        int match_sub_block = __ffs(match_mask) - 1;
        int empty_sub_block = __ffs(empty_mask) - 1;
        //printf("Block %d: Determining match/free status\n", blockIdx.x);
        if (lock_mask)
        {
            // Window must be probed again
            status = ProbeStatus::PROBE_CURRENT;
        }
        else if (match_sub_block >= 0)
        {
            // match found, incrementing
            // deletions from hashtables not implemented
            // printf("Block %d: Match\n", blockIdx.x); 
            int offset = 9 * match_sub_block + 8;
            atomicAdd((uint32_t*)window + offset, 1UL);
            status = ProbeStatus::SUCCEESS;
        }
        else if (empty_sub_block >= 0)
        {
            // printf("Block %d: Insert key\n", blockIdx.x); 
            // match not found, try insertion to available space
            if (try_insert((uint32_t*)window + 9 * empty_sub_block, key))
            {
                //printf("Block %d: Insert success\n", blockIdx.x); 
                status = ProbeStatus::SUCCEESS;
            }
            else 
            {
                //printf("Block %d: Insert failed\n", blockIdx.x); 
                // insertion failed; probe current window again
                status = ProbeStatus::PROBE_CURRENT;
            }
        } 
        else 
        {
            //printf("Block %d: No match or free space, goto next window\n", blockIdx.x); 
            status = ProbeStatus::PROBE_NEXT;   // no match or space; goto next window
        }
            
    }
    __syncthreads();
}

__device__ 
uint32_t GpuHashtable::hash(const Compressed128Mer& key) const
{
    const uint32_t c1 = 0xcc9e2d51UL;
    const uint32_t c2 = 0x1b873593UL;
    const uint32_t r1 = 15UL;
    const uint32_t r2 = 13UL;
    const uint32_t m = 5UL;
    const uint32_t n = 0xe6546b64UL;

    uint32_t hash = 0x6789fUL;  // seed
    for (int i = 0; i < 8; ++i)
    {
        uint32_t k = key.u32[i];
        k *= c1;
        k >>= r1;
        k *= c2;

        hash ^= k;
        hash >>= r2;
        hash = hash * m + n;
    }

    hash ^= 128UL;
    hash = hash ^ (hash >> 16UL);
    hash *= 0x85ebca6bUL;
    hash = hash ^ (hash >> 13UL);
    hash *= 0xc2b2ae35UL;
    hash = hash ^ (hash >> 16UL);
    return hash;
}

__device__
uint32_t GpuHashtable::hash(uint32_t value) const
{
    const uint32_t c1 = 0xcc9e2d51UL;
    const uint32_t c2 = 0x1b873593UL;
    const uint32_t r1 = 15UL;
    const uint32_t r2 = 13UL;
    const uint32_t m = 5UL;
    const uint32_t n = 0xe6546b64UL;

    uint32_t hash = 0x64859ecUL;  // seed   

    uint32_t k = value;
    k *= c1;
    k >>= r1;
    k *= c2;

    hash ^= k;
    hash >>= r2;
    hash = hash * m + n;

    hash ^= 4UL;
    hash = hash ^ (hash >> 16UL);
    hash *= 0x85ebca6bUL;
    hash = hash ^ (hash >> 13UL);
    hash *= 0xc2b2ae35UL;
    hash = hash ^ (hash >> 16UL);

    return hash;
}

__device__
void GpuHashtable::get_job_batch()
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
void GpuHashtable::run()
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
            utils::Read128Mer(reference, i, current_key);
            Compressed128Mer key = current_key;
            process(key);
        }
    }
}

__device__ 
bool GpuHashtable::try_insert(uint32_t* kv_pair, const Compressed128Mer& key)
{
    if (atomicCAS(kv_pair + 8, 0, 0xFFFFFFFFU) == 0)
    {
        // sucessfully occupied the empty entry, copying keys
        for (int i = 0; i < 8; ++i)
            kv_pair[i] = key.u32[i];
         __threadfence();
        kv_pair[8] = 1UL;   // unlock
        return true;
    }
    else 
    {
        // insertion failed
        return false;
    }
}