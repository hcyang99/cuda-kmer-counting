#include "Gerbil/Gerbil.cuh"
#include "benchmarks/HashtableWalker.cuh"
#include "utils/Fasta.cuh"
#include "utils/FailureZone.cuh"
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
    __shared__ uint32_t s[1024];
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
uint32_t* gerbil(uint32_t* ref, size_t n_bps, size_t& n_buckets_out, uint32_t max_trials)
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
    return d_buf;
}

// __global__ 
// void kmer_diff_scan(Compressed128Mer* kmers, uint32_t* out, size_t size)
// {

// }



int main()
{
    utils::printGpuProperties();

    uint32_t result[128];
    Fasta fasta("data/cut.fa");
    cerr << "Reading " << fasta.size() << " base pairs" << endl;

    uint32_t* d_ref = fasta.toGpuCompressed();
    size_t num_buckets;
    uint32_t* d_table_buf = gerbil(d_ref, fasta.size(), num_buckets, 4);

    cerr << "Walking through " << num_buckets << " buckets: " << d_table_buf << endl;
   
    hashtable_walk(d_table_buf, num_buckets, result);
    for (int i = 1; i < 128; ++i)
    {
        cout << "result[" << i << "] = " << result[i] << endl;
    }
}