#include "SlabHash/SlabHash.cuh"
#include "benchmarks/HashtableWalker.cuh"
#include "utils/Fasta.cuh"
#include <chrono>
using namespace std;

__global__ 
void slab_kernel(uint32_t* ref, uint32_t* d, uint32_t n, JobQueue* j, SlabPool* all_pools)
{
    __shared__ char s[sizeof(SlabHash)];
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    SlabHash* slab_hash_ptr = (SlabHash*) s;
    if (tx == 0)
    {
        new (slab_hash_ptr) SlabHash(ref, d, n, j, all_pools, all_pools + bx);
    }
    __syncthreads();

    slab_hash_ptr->run();
}

__global__
void create_slab_pools(SlabPool* loc, uint32_t* buf, size_t total_size)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int gs = gridDim.x;
    uint32_t my_size = total_size / gs;
    uint32_t* my_buf = buf + bx * (my_size / sizeof(uint32_t));

    if (tx == 0)
    {
        new (loc + bx) SlabPool(my_buf, my_size / sizeof(uint32_t));
    }
}

__host__
uint32_t* slab(uint32_t* ref, size_t n_bps, size_t& n_buckets_out, int64_t& us)
{
    size_t n_jobs = n_bps - 127;
    JobQueue* j = new_job(n_jobs, utils::BatchSize::get());
    size_t num_buckets = n_jobs * utils::SlabFactor::get() / 14;
    size_t num_slabs = (n_jobs / 14 / utils::GridSize::get() + 1) * utils::GridSize::get();

    uint32_t* d_buf;
    uint32_t* d_slab_buf;
    SlabPool* d_slab_pools;
    // cerr << "Allocating " << num_buckets << " buckets" << endl;
    // cerr << "Allocating " << num_slabs << " slabs" << endl;

    CUDA_CHECK_ERROR(cudaMalloc(&d_buf, num_buckets * 512));
    CUDA_CHECK_ERROR(cudaMalloc(&d_slab_buf, num_slabs * 512 * 2));
    CUDA_CHECK_ERROR(cudaMalloc(&d_slab_pools, utils::GridSize::get() * sizeof(SlabPool)));
    CUDA_CHECK_ERROR(cudaMemset(d_buf, 0, num_buckets * 512));
    CUDA_CHECK_ERROR(cudaMemset(d_slab_buf, 0, num_slabs * 512 * 2));

    // cerr << "Allocated memory for slabs: " << d_slab_buf << " to " << d_slab_buf + num_slabs * 128 << endl;
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    auto t1 = chrono::high_resolution_clock::now();

    create_slab_pools<<<utils::GridSize::get(), 32>>>
        (d_slab_pools, d_slab_buf, num_slabs * 512);

    // CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    
    slab_kernel<<<utils::GridSize::get(), utils::BlockSize::get()>>>
        (ref, d_buf, num_buckets, j, d_slab_pools);

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    auto t2 = chrono::high_resolution_clock::now();

    us = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();

    n_buckets_out = num_buckets;
    return d_buf;
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        cerr << "Usage: " << argv[0] << " <path-to-fasta> <slab-factor>" << endl;
        exit(1);
    }

    string path(argv[1]);
    float slab_factor = strtof(argv[2], nullptr);

    uint32_t result[128];
    Fasta fasta(path);
    // cerr << "Reading " << fasta.size() << " base pairs" << endl;

    int64_t us_run, us_walk;

    uint32_t* d_ref = fasta.toGpuCompressed();
    size_t num_buckets;
    uint32_t* d_table_buf = slab(d_ref, fasta.size(), num_buckets, us_run);

    cout << "slab (" << slab_factor << ") Insertion Time: " << us_run << " us\n";

    // cerr << "Walking through " << num_buckets << " buckets" << endl;

    auto t1 = chrono::high_resolution_clock::now();
   
    hashtable_walk(d_table_buf, num_buckets, result);

    auto t2 = chrono::high_resolution_clock::now();
    us_walk = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();

    cout << "slab (" << slab_factor << ") Walk Time: " << us_walk << " us\n";
}