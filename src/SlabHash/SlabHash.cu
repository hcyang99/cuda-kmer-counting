#include "SlabHash.cuh"

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