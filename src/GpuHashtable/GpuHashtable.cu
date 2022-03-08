#include "GpuHashtable.cuh"

__device__ __forceinline__
GpuHashtable::ProbeStatus GpuHashtable::simd_probe(uint32_t* data, byte_32& key, uint32_t* match_status, uint32_t* empty_status)
{
    int tx = threadIdx.x;
    int subwarp_idx = tx / 14;  // 14 (32B key, 4B value) pairs
    int sub_tx = tx % 9;        // 8 threads process the key, 1 thread processes the value
    if (tx != 127)              // let the last thread rest
        if (sub_tx == 0)   
        {
            match_status[subwarp_idx] = 1;  // initialize as matched
        }
    __syncthreads();

    if (tx != 127)
        if (sub_tx != 8)
            if (data[tx] != key.u32[sub_tx])
            {
                match_status[subwarp_idx] = 0;  // set to 0 if not matching
            }
        else
        {
            empty_status[subwarp_idx] = data[tx] == 0;
        }
    __syncthreads();

    if (tx >= 0 && tx < 14)
    {
        match_status[14] = __ballot_sync(__activemask(), match_status[tx]);
    }
    else if (tx >= 32 && tx < 32 + 14)
    {
        empty_status[14] = __ballot_sync(__activemask(), empty_status[tx]);
    }
    __syncthreads();

    // match found, incrementing
    // deletions from hashtables not implemented
    int match_sub_block = __ffs(match_status[14]);
    int empty_sub_block = __ffs(empty_status[14]);
    if (tx == 0 && match_sub_block)
    {
        int offset = 9 * match_sub_block + 8;
        atomicAdd(data + offset, 1UL);
    }
    if (match_sub_block)
        return ProbeStatus::SUCCEESS;
    if (empty_sub_block)
        return ProbeStatus::INSERT;
    return ProbeStatus::PROBE;
}