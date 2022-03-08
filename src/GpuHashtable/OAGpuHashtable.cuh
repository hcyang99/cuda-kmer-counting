#pragma once
#include "GpuHashtable.cuh"

/**
 * @brief Open Addressing GPU Hashtable
 */
class OAGpuHashtable : public GpuHashtable
{
    protected:
    OAGpuHashtable(uint32_t d)
        :data(d) {}

    uint32_t data;
};