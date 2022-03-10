#include "GpuHashtable/GpuHashtable.cuh"

__device__
void GpuHashtable::simd_probe(uint32_t* data, const Compressed128Mer& key)
{
    int tx = threadIdx.x;
    int subwarp_idx = tx / 14;  // 14 (32B key, 4B value) pairs
    int sub_tx = tx % 9;        // 8 threads process the key, 1 thread processes the value
    if (tx < 126)               // let the last 2 threads rest
        if (sub_tx == 0)   
        {
            match_status[subwarp_idx] = 1;  // initialize as matched
        }
    __syncthreads();

    if (tx < 126)
        if (sub_tx != 8)
            if (data[tx] != key.u32[sub_tx])
            {
                match_status[subwarp_idx] = 0;  // set to 0 if not matching
            }
        else
        {
            empty_status[subwarp_idx] = data[tx] == 0;
        }
    __syncthreads();

    uint32_t match_mask, empty_mask;
    if (tx >= 0 && tx < 14)
    {
        match_mask = __ballot_sync(__activemask(), match_status[tx]);
        empty_mask = __ballot_sync(__activemask(), empty_status[tx]);
    }

    if (tx == 0)
    {
        match_status[14] = match_mask;
        empty_status[14] = empty_mask;
        int match_sub_block = __ffs(match_mask) - 1;
        int empty_sub_block = __ffs(empty_mask) - 1;
        if (match_sub_block >= 0)
        {
            // match found, incrementing
            // deletions from hashtables not implemented
            int offset = 9 * match_sub_block + 8;
            atomicAdd(data + offset, 1UL);
            status = ProbeStatus::SUCCEESS;
        }
        else if (empty_sub_block >= 0)
        {
            // match not found, try insertion to available space
            if (try_insert(data + 9 * empty_sub_block, key))
            {
                status = ProbeStatus::SUCCEESS;
            }
            else 
            {
                // insertion failed; probe current window again
                status = ProbeStatus::PROBE_CURRENT;
            }
        } 
        else 
            status = ProbeStatus::PROBE_NEXT;   // no match or space; goto next window
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
    if (atomicCAS(kv_pair + 8, 0UL, 1UL) == 0)
    {
        // sucessfully occupied the empty entry, copying keys
        for (int i = 0; i < 8; ++i)
            kv_pair[i] = key.u32[i];
        return true;
    }
    else 
    {
        // insertion failed
        return false;
    }
}