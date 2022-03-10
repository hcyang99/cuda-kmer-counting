#include "SlabHash/SlabHash.cuh"
#include "assert.h"

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
        uint32_t is_curr_pool_full = all_slab_pools[i].full();
        uint32_t full_pool_mask = __ballot_sync(__activemask(), is_curr_pool_full);
        if (tx % 32 == 0)
        {
            int num_free = __clz(full_pool_mask);   // count number of non-filled pools
            if (num_free != 0)
            {
                int selected_zero_bit = seed % num_free;
                full_pool_mask = ~full_pool_mask;
                for (int j = 0; j < selected_zero_bit; ++j)
                {
                    uint32_t flip_mask = 1UL << (__ffs(full_pool_mask) - 1);
                    full_pool_mask ^= flip_mask;    // flip first `selected_zero_bit` bits of 1
                }
                int selected_offset = __ffs(full_pool_mask) - 1;
                if (selected_offset >= 0)
                {
                    int selected = i - i % 32 + selected_offset;
                    if (mem_pool == nullptr)
                        mem_pool = all_slab_pools + selected;
                }
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
            simd_probe(probe_pos, key);
        else if (status == ProbeStatus::SUCCEESS)
            return;
        else if (status == ProbeStatus::PROBE_NEXT)
        {
            // no empty space in current slab
            uint32_t** next_ptr = reinterpret_cast<uint32_t**>(probe_pos + 126);
            uint32_t* next = *next_ptr; // the "next" pointer
            if (next)
            {
                if (tx == 0)
                {
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
                    for (int i = 0; i < 8; ++i)
                        new_slab[i] = key.u32[i];
                    new_slab[8] = 1UL;

                    // tries to append `new_slab` to end of current slab
                    uint32_t* old_next = (uint32_t*)atomicCAS((unsigned long long*)next_ptr, 0ULL, (uint64_t)next_ptr);
                    if (old_next == nullptr)
                    {
                        // slab insertion success
                        new_slab = nullptr;
                        status = ProbeStatus::SUCCEESS;
                    }
                    else 
                    {
                        // slab insertion failed
                        probe_pos = old_next;
                        status = ProbeStatus::PROBE_CURRENT;
                    }
                }
                __syncthreads();
            }
        }
        else 
        {
            assert(0);  // should not reach here
        }
    }
}

