#include "WarpCore/WarpCore.cuh"
#include <assert.h>

__device__
void WarpCore::process(const Compressed128Mer& key)
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
            if (tx == 0)
            {
                uint32_t* next_pos = get_next_probe_pos(key);
                status = ProbeStatus::PROBE_CURRENT;
            }
            __syncthreads();
        }
        else 
        {
            assert(0);  // should not reach here
        }
    }
}