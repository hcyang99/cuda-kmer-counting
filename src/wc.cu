#include "WarpCore/WarpCore.cuh"
#include "benchmarks/HashtableWalker.cuh"
#include "utils/Fasta.cuh"
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
uint32_t* warp_core(uint32_t* ref, size_t n_bps, size_t& n_buckets_out)
{
    size_t n_jobs = n_bps - 127;
    JobQueue* j = new_job(n_jobs, utils::batchSize());
    size_t num_buckets = n_jobs * utils::OAFactor() / 14;

    uint32_t* d_buf;
    cerr << "Allocating " << num_buckets << " buckets" << endl;
    CUDA_CHECK_ERROR(cudaMalloc(&d_buf, num_buckets * 512));
    CUDA_CHECK_ERROR(cudaMemset(d_buf, 0, num_buckets * 512));

    warp_core_kernel<<<utils::gridSize(), utils::blockSize()>>>
        (ref, d_buf, num_buckets, j);
    
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    n_buckets_out = num_buckets;
    return d_buf;
}



int main()
{
    utils::printGpuProperties();

    uint32_t result[128];
    Fasta fasta("data/cut.fa");
    cerr << "Reading " << fasta.size() << " base pairs" << endl;

    uint32_t* d_ref = fasta.toGpuCompressed();
    size_t num_buckets;
    uint32_t* d_table_buf = warp_core(d_ref, fasta.size(), num_buckets);

    cerr << "Walking through " << num_buckets << " buckets: " << d_table_buf << endl;
   
    hashtable_walk(d_table_buf, num_buckets, result);
    for (int i = 1; i < 128; ++i)
    {
        cout << "result[" << i << "] = " << result[i] << endl;
    }
}