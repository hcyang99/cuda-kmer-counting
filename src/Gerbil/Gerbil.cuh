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
    Gerbil(uint32_t* d, uint32_t n, JobQueue* j, uint32_t max_trials, FailureZone<Compressed128Mer>* f_zone)
        : OAGpuHashtable(d, n, j), max_attempts(max_trials), failure_zone(f_zone) {}

    protected:
    const uint32_t max_attempts;
    FailureZone<Compressed128Mer>* const failure_zone;

    /**
     * @brief Insert/increment the key in Gerbil
     */
    virtual void process(const Compressed128Mer& key) override;
};