#pragma once
#include "GpuHashtable/OAGpuHashtable.cuh"
#include "utils/FailureZone.cuh"

/**
 * @brief An open-addressing hashtable, if a key is not inserted after several attempts, 
 * the key is placed to `failure_zone`, which will be counted using sort + prefix scan
 */
class Gerbil : public OAGpuHashtable
{
    public:
    Gerbil(uint32_t* ref, uint32_t* d, uint32_t n, JobQueue* j, uint32_t max_trials, FailureZone<Compressed128Mer>* f_zone)
        : OAGpuHashtable(ref, d, n, j), max_attempts(max_trials), failure_zone(f_zone) {}

    protected:
    const uint32_t max_attempts;
    FailureZone<Compressed128Mer>* const failure_zone;

    /**
     * @brief Insert/increment the key in Gerbil
     */
    virtual void process(const Compressed128Mer& key) override;
};


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
