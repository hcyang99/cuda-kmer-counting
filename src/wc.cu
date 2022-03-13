#include "WarpCore/WarpCore.cuh"
#include "benchmarks/HashtableWalker.cuh"
#include "utils/Fasta.cuh"
#include <chrono>
using namespace std;

__global__ 
void warp_core_kernel(uint32_t* ref, uint32_t* d, uint32_t n, JobQueue* j)
{
    __shared__ char s[sizeof(WarpCore)];
    int tx = threadIdx.x;

    WarpCore* h = (WarpCore*) s;
    if (tx == 0)
    {
        new (h) WarpCore(ref, d, n, j);
    }
    __syncthreads();

    h->run();
}

__host__ 
uint32_t* warp_core(uint32_t* ref, size_t n_bps, size_t& n_buckets_out, int64_t& us_out)
{
    size_t n_jobs = n_bps - 127;
    JobQueue* j = new_job(n_jobs, utils::BatchSize::get());
    size_t num_buckets = n_jobs * utils::OAFactor::get() / 14;

    uint32_t* d_buf;
    // cerr << "Allocating " << num_buckets << " buckets" << endl;
    CUDA_CHECK_ERROR(cudaMalloc(&d_buf, num_buckets * 512));
    CUDA_CHECK_ERROR(cudaMemset(d_buf, 0, num_buckets * 512));

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    auto t1 = chrono::high_resolution_clock::now();

    warp_core_kernel<<<utils::GridSize::get(), utils::BlockSize::get()>>>
        (ref, d_buf, num_buckets, j);
    
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    auto t2 = chrono::high_resolution_clock::now();
    us_out = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();

    n_buckets_out = num_buckets;
    return d_buf;
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        cerr << "Usage: " << argv[0] << " <path-to-fasta> <oa-factor>" << endl;
        exit(1);
    }

    string path(argv[1]);
    float oa_factor = strtof(argv[2], nullptr);
    utils::OAFactor::set(oa_factor);

    uint32_t result[128];
    Fasta fasta(path);
    // cerr << "Reading " << fasta.size() << " base pairs" << endl;

    int64_t us_run, us_walk;

    uint32_t* d_ref = fasta.toGpuCompressed();
    size_t num_buckets;
    uint32_t* d_table_buf = warp_core(d_ref, fasta.size(), num_buckets, us_run);

    // cerr << "Walking through " << num_buckets << " buckets: " << d_table_buf << endl;
    cout << "wc (" << oa_factor << ") Insertion Time: " << us_run << " us\n";
   
    auto t1 = chrono::high_resolution_clock::now();
    hashtable_walk(d_table_buf, num_buckets, result);
    auto t2 = chrono::high_resolution_clock::now();
    us_walk = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();

    cout << "wc (" << oa_factor << ") Walk Time: " << us_walk << " us\n";
}