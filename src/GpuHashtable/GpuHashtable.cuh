#pragma once
#include "utils/utils.cuh"

using utils::Compressed128Mer;

/**
 * @brief Common abstract base class for GPU hashtables
 */
class GpuHashtable
{
    protected:
    /**
     * @brief Protected constructor prevents instantiation
     */
    GpuHashtable(){}

    enum class ProbeStatus {SUCCEESS, INSERT, PROBE};

    /**
     * @brief Probe single SIMD window (512 B), do increment if possible, Assuming 128 threads
     * @param data Underlying hashtable 
     * @param key The query key
     * @param match_status Value residing in shared memory, for matching status of 14 sub-warps
     * @param empty_status Value residing in shared memory, for empty status of 14 sub-warps
     * @return SUCCESS: increment complete; INSERT: try insertion; PROBE: no free space found in current SIMD window
     */
    ProbeStatus simd_probe(uint32_t* data, const Compressed128Mer& key, uint32_t* match_status, uint32_t* empty_status);
};