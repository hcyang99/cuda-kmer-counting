#pragma once
#include "utils/utils.cuh"

class Stats
{
    public:
    Stats() : buckets(), lock() {}

    /**
     * @brief increment `id` by `inc`
     * @param id number of occurances
     * @param inc number of keys with `id` occurances
     */
    void increment(uint32_t id, uint32_t inc);

    /**
     * @brief merge `other` into `this`
     */
    Stats& operator+=(const Stats& other);

    uint32_t* get_buffer()
    {
        return buckets;
    }

    protected:
    uint32_t buckets[128];
    uint32_t lock;
};