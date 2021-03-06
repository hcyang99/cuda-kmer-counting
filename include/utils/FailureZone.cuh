#pragma once
#include "utils/utils.cuh"

/**
 * @brief Singleton class for FailureZone
 */
template<typename T>
class FailureZone
{
    public:
    __device__ FailureZone(T* data, size_t size)
        :buffer(data), end(data + size), head(data) {}
    
    /**
     * @brief Thread-safely inserts an item to end of list; should only be called from thread 0 of each block
     * @return True: success; False: out of space
     */
    __device__ bool insert(const T& item)
    {
        T* old_head = (T*)atomicAdd((unsigned long long*)&head, sizeof(T));
        if (old_head >= end)
            return false;
        *old_head = item;
        return true;
    }

    __device__ const T* get_buffer() const 
    {
        return buffer;
    }

    __device__ size_t size() const 
    {
        return head - buffer;
    }

    protected:    
    T* const buffer;
    T* const end;
    T* head;
};