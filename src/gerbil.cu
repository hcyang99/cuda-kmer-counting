#include "Gerbil/Gerbil.cuh"
#include "benchmarks/HashtableWalker.cuh"
#include "utils/Fasta.cuh"
#include "utils/FailureZone.cuh"
#include "benchmarks/OccurenceArrayWalker.cuh"
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
using namespace std;

__global__
void init_failure_zone_kernel(FailureZone<Compressed128Mer>* location, Compressed128Mer* buffer, size_t size)
{
    if (utils::global_thread_id() == 0)
    {
        new (location) FailureZone<Compressed128Mer>(buffer, size);
    }
}

__global__ 
void gerbil_kernel(uint32_t* ref, uint32_t* d, uint32_t n, JobQueue* j, uint32_t max_trials, FailureZone<Compressed128Mer>* f_zone)
{
    __shared__ char s[sizeof(Gerbil)];
    int tx = threadIdx.x;

    Gerbil* h = (Gerbil*) s;
    if (tx == 0)
    {
        new (h) Gerbil(ref, d, n, j, max_trials, f_zone);
    }
    __syncthreads();

    h->run();
}

__host__ 
FailureZone<Compressed128Mer>* get_failure_zone(size_t size)
{
    Compressed128Mer* buffer; 
    FailureZone<Compressed128Mer>* location;
    CUDA_CHECK_ERROR(cudaMalloc(&buffer, size * sizeof(Compressed128Mer)));
    CUDA_CHECK_ERROR(cudaMalloc(&location, sizeof(FailureZone<Compressed128Mer>)));
    init_failure_zone_kernel<<<1, 32>>>(location, buffer, size);
    
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    return location;
}

__host__ 
uint32_t* gerbil(uint32_t* ref, size_t n_bps, size_t& n_buckets_out, uint32_t max_trials, FailureZone<Compressed128Mer>*& f_zone_out)
{
    size_t n_jobs = n_bps - 127;
    JobQueue* j = new_job(n_jobs, utils::batchSize());
    size_t num_buckets = n_jobs * utils::OAFactor() / 14;

    uint32_t* d_buf;
    cerr << "Allocating " << num_buckets << " buckets" << endl;
    CUDA_CHECK_ERROR(cudaMalloc(&d_buf, num_buckets * 512));
    CUDA_CHECK_ERROR(cudaMemset(d_buf, 0, num_buckets * 512));

    auto f_zone = get_failure_zone(n_jobs);

    gerbil_kernel<<<utils::gridSize(), utils::blockSize()>>>
        (ref, d_buf, num_buckets, j, max_trials, f_zone);
    
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    n_buckets_out = num_buckets;
    f_zone_out = f_zone;
    return d_buf;
}

__global__ 
void kmer_diff_scan(Compressed128Mer* kmers, uint32_t* out, size_t size)
{
    int tx = utils::global_thread_id();
    if (tx == 0)
        out[tx] = 1;
    else if (tx < size)
    {
        out[tx] = kmers[tx - 1] == kmers[tx];
    }
}

struct KmerLess
{
    __device__ __host__ KmerLess() {}

    __device__ __host__ bool operator()(const Compressed128Mer& lhs, const Compressed128Mer& rhs) const 
    {
        for (int i = 0; i < 8; ++i)
        {
            if (lhs.u32[i] < rhs.u32[i])
                return true;
            else if (lhs.u32[i] > rhs.u32[i])
                return false;
        }
        return false;
    }
};

__global__ 
void symbol_compaction_kernel(uint32_t* diff_array, uint32_t* symbol_array, size_t size, uint32_t* out)
{
    int tx = utils::global_thread_id();
    if (tx == 0)
    {
        out[0] = 0;
    }
    if (tx < size && diff_array[tx])
    {
        out[symbol_array[tx]] = tx;
    }
}

__global__ 
void decompact_occurences_kernel(uint32_t* c, uint32_t* out, size_t size)
{
    int tx = utils::global_thread_id();
    if (tx < size)
        out[tx] = c[tx + 1] - c[tx];
}

__host__ 
void process_failure_zone(FailureZone<Compressed128Mer>* f_zone, Stats* stats, uint32_t* out)
{
    Compressed128Mer* f_zone_ptrs[3];
    CUDA_CHECK_ERROR(cudaMemcpy(f_zone_ptrs, f_zone, 24, cudaMemcpyDeviceToHost));

    Compressed128Mer* buffer = f_zone_ptrs[0];
    Compressed128Mer* end = f_zone_ptrs[2];
    size_t size = end - buffer;

    cerr << "Walking through " << size << " failed entries\n";

    // Step 1: sort
    thrust::sort(thrust::device, buffer, end, KmerLess());

    // Step 2: locate diffs
    uint32_t* diff_array;
    CUDA_CHECK_ERROR(cudaMalloc(&diff_array, size * sizeof(uint32_t)));
    kmer_diff_scan<<<size / utils::blockSize() + 1, utils::blockSize()>>>
        (buffer, diff_array, size);
    CUDA_CHECK_ERROR(cudaFree(buffer));

    // Step 3: prefix scan on diff array
    uint32_t* symbol_array;
    CUDA_CHECK_ERROR(cudaMalloc(&symbol_array, size * sizeof(uint32_t)));
    thrust::inclusive_scan(thrust::device, diff_array, diff_array + size, symbol_array);

    // Step 4: stream compaction (result is prefix scan of array of occurences of each kmer)
    uint32_t* occurences_compacted;
    uint32_t num_keys;
    CUDA_CHECK_ERROR(cudaMemcpy(&num_keys, symbol_array + (size - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMalloc(&occurences_compacted, (num_keys + 1) * sizeof(uint32_t)));
    symbol_compaction_kernel<<<size / utils::blockSize() + 1, utils::blockSize()>>>
        (diff_array, symbol_array, size, occurences_compacted);

    CUDA_CHECK_ERROR(cudaFree(diff_array));
    CUDA_CHECK_ERROR(cudaFree(symbol_array));

    // Step 5: map, output is array of occurences of each kmer
    uint32_t* occurence_array;
    CUDA_CHECK_ERROR(cudaMalloc(&occurence_array, num_keys * sizeof(uint32_t)));
    decompact_occurences_kernel<<<num_keys / utils::blockSize() + 1, utils::blockSize()>>>
        (occurences_compacted, occurence_array, num_keys);

    CUDA_CHECK_ERROR(cudaFree(occurences_compacted));

    // Step 6: occurence array walker
    occurence_table_walk(occurence_array, num_keys, stats, out);
}



int main()
{
    utils::printGpuProperties();

    uint32_t result[128];
    Fasta fasta("data/cut.fa");
    cerr << "Reading " << fasta.size() << " base pairs" << endl;

    uint32_t* d_ref = fasta.toGpuCompressed();
    size_t num_buckets;
    FailureZone<Compressed128Mer>* f_zone;
    uint32_t* d_table_buf = gerbil(d_ref, fasta.size(), num_buckets, 4, f_zone);

    cerr << "Walking through " << num_buckets << " buckets: " << d_table_buf << endl;
   
    Stats* stats = hashtable_walk(d_table_buf, num_buckets);
    process_failure_zone(f_zone, stats, result);

    for (int i = 1; i < 128; ++i)
    {
        cout << "result[" << i << "] = " << result[i] << endl;
    }
}