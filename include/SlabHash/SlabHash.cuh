#pragma once
#include "GpuHashtable/GpuHashtable.cuh"
#include "SlabHash/SlabPool.cuh"
#include "utils/JobQueue.cuh"

/**
 * @brief SlabHash, a separate chaining GPU hashtable, should be allocated on shared memory of each block
 */
class SlabHash : public GpuHashtable
{
    public:
    __device__ SlabHash(uint32_t* ref, uint32_t* d, uint32_t n, JobQueue* j, SlabPool* all_pools, SlabPool* own_pool)
        :GpuHashtable(ref, d, n, j), all_slab_pools(all_pools), mem_pool(own_pool), new_slab(nullptr) {}

    protected:
    SlabPool* const all_slab_pools; // Array of all Slab pools
    SlabPool* mem_pool;             // Current memory pool allocating from; Initally self, when full, some other block's pool
    uint32_t* new_slab;

    /**
     * @brief Get a new SlabPool when current is full, if every pool is full, will set `mem_pool` to NULL
     * @param seed Seed for random pool selection
     */
    __device__ void get_new_pool(uint32_t seed);

    /**
     * @brief Allocate a new slab from current pool, store in `new_slab`, switch to new pool if necessary
     * @param seed Seed for random pool selection
     */
    __device__ void allocate_slab(uint32_t seed);

    /**
     * @brief Insert/increment the key in SlabHash
     */
    __device__ virtual void process(const Compressed128Mer& key) override;
};

__device__
void SlabHash::get_new_pool(uint32_t seed)
{
    int tx = threadIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;
    if (tx == 0)
        mem_pool = nullptr;
    __syncthreads();

    for (int i = tx; i < gs; i += bs)
    {
        if (tx % 32 == 0 && mem_pool == nullptr)
        {
            size_t max_available_size[3] = {0, 0, 0};
            int max_available_idx[3] = {-1, -1, -1};
            for (int j = i; j < i + 32 && j < gs; ++j)
            {
                size_t curr_space = all_slab_pools[j].space();
                if (curr_space > max_available_size[0])
                {
                    max_available_size[2] = max_available_size[1];
                    max_available_size[1] = max_available_size[0];
                    max_available_size[0] = curr_space;

                    max_available_idx[2] = max_available_idx[1];
                    max_available_idx[1] = max_available_idx[0];
                    max_available_idx[0] = j;
                }
                else if (curr_space > max_available_size[1])
                {
                    max_available_size[2] = max_available_size[1];
                    max_available_size[1] = curr_space;

                    max_available_idx[2] = max_available_idx[1];
                    max_available_idx[1] = j;
                }
                else if (curr_space > max_available_size[2])
                {
                    max_available_size[2] = curr_space;
                    max_available_idx[2] = j;
                }
            }

            int selected = -1;
            if (max_available_idx[2] >= 0)
            {
                selected = max_available_idx[seed % 1001 % 3];
            }
            else if (max_available_idx[1] >= 0)
            {
                selected = max_available_idx[seed % 1001 % 2];
            }
            else if (max_available_idx[0] >= 0)
            {
                selected = max_available_idx[0];
            }

            if (selected >= 0 && mem_pool == nullptr)
            {
                mem_pool = all_slab_pools + selected;
                // printf("Block: %d, Thread: %d, selected = %d\n", blockIdx.x, threadIdx.x, selected);
            }
        }
    }
    __syncthreads();
}

__device__
void SlabHash::allocate_slab(uint32_t seed)
{
    if (threadIdx.x == 0)
    {
        new_slab = mem_pool->allocate();
    }
    __syncthreads();

    while (new_slab == nullptr)
    {
        get_new_pool(seed);
        if (threadIdx.x == 0)
        {
            new_slab = mem_pool->allocate();
        }
        __syncthreads();
    }
}

__device__
void SlabHash::process(const Compressed128Mer& key)
{
    int tx = threadIdx.x;

    if (tx == 0)
    {
        uint32_t h = hash(key);
        probe_pos = data + h % num_buckets * 128;
        status = ProbeStatus::PROBE_CURRENT;
    }
    __syncthreads();

    while (true)
    {
        if (status == ProbeStatus::PROBE_CURRENT)
        {
            simd_probe(probe_pos, key);
        } 
        else if (status == ProbeStatus::SUCCEESS)
        {
            return;
        }
        else if (status == ProbeStatus::PROBE_NEXT)
        {
            // no empty space in current slab
            uint32_t** next_ptr = reinterpret_cast<uint32_t**>(probe_pos + 126);
            uint32_t* next = *next_ptr; // the "next" pointer
            if (next)
            {
                if (tx == 0)
                {
                    // printf("Block %d: Moving to next slab: %p\n", blockIdx.x, next); 
                    probe_pos = next;
                    status = ProbeStatus::PROBE_CURRENT;
                }
                __syncthreads();
            } 
            else 
            {
                // try to allocate new slab
                if (new_slab == nullptr)    // don't allocate if previous is unused
                    allocate_slab(key.u32[3]);
                
                if (tx == 0)
                {
                    // writes key, value into head of new slab
                    // printf("Block %d: Using allocated Slab %p\n", blockIdx.x, new_slab); 
                    for (int i = 0; i < 8; ++i)
                        new_slab[i] = key.u32[i];
                    new_slab[8] = 1UL;
                     __threadfence();

                    // tries to append `new_slab` to end of current slab
                    uint32_t* old_next = (uint32_t*)atomicCAS((unsigned long long*)next_ptr, 0ULL, (uint64_t)new_slab);
                    if (old_next == nullptr)
                    {
                        // slab insertion success
                        // printf("Block %d: Successfully inserted new slab %p\n", blockIdx.x, new_slab); 
                        new_slab = nullptr;
                        status = ProbeStatus::SUCCEESS;
                    }
                    else 
                    {
                        // slab insertion failed
                        // printf("Block %d: Failed to insert new slab\n", blockIdx.x); 
                        probe_pos = old_next;
                        status = ProbeStatus::PROBE_CURRENT;
                    }
                }
                __syncthreads();
            }
        }
    }
}
