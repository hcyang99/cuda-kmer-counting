#include "Gerbil/Gerbil.cuh"
#include <assert.h>

__device__
void Gerbil::process(const Compressed128Mer& key)
{
    int tx = threadIdx.x;
    uint32_t current_probe_count = 0;

    if (tx == 0)
    {
        uint32_t h = hash(key);
        probe_pos = data + h % num_buckets * 128;
        status = ProbeStatus::PROBE_CURRENT;
    }
    __syncthreads();

    while (current_probe_count < max_attempts)
    {
        if (status == ProbeStatus::PROBE_CURRENT)
            simd_probe(probe_pos, key);
        else if (status == ProbeStatus::SUCCEESS)
            return;
        else if (status == ProbeStatus::PROBE_NEXT)
        {
            ++current_probe_count;
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

    // max_attempts reached, adding to failure zone
    if (tx == 0)
        failure_zone->insert(key);
}
