#include "SlabHash/SlabHash.cuh"
#include "benchmarks/HashtableWalker.cuh"
#include "utils/Fasta.cuh"
using namespace std;

__global__ 
void slab_kernel(uint32_t* ref, uint32_t* d, uint32_t n, JobQueue* j, SlabPool* all_pools)
{
    __shared__ uint32_t s[1024];
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    SlabHash* slab_hash_ptr = (SlabHash*) s;
    if (tx == 0)
    {
        new (s) SlabHash(ref, d, n, j, all_pools, all_pools + bx);
    }
    __syncthreads();

    slab_hash_ptr->run();
}

__global__
void create_slab_pools(SlabPool* loc, uint32_t* buf, uint32_t total_size)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int gs = gridDim.x;
    uint32_t my_size = total_size / gs;
    uint32_t* my_buf = buf + gs * (my_size / sizeof(uint32_t));

    if (tx == 0)
    {
        new (loc + bx) SlabPool(my_buf, my_size / sizeof(uint32_t));
    }
}

__host__
uint32_t* slab(uint32_t* ref, uint32_t n_bps, uint32_t& n_buckets_out)
{
    uint32_t n_jobs = n_bps - 127;
    JobQueue* j = new_job(n_jobs, utils::batchSize());
    uint32_t num_buckets = n_jobs * utils::slabFactor() / 14;
    uint32_t num_slabs = (n_jobs / 14 / utils::gridSize() + 1) * utils::gridSize();

    uint32_t* d_buf;
    uint32_t* d_slab_buf;
    SlabPool* d_slab_pools;
    CUDA_CHECK_ERROR(cudaMalloc(&d_buf, num_buckets * 512));
    CUDA_CHECK_ERROR(cudaMalloc(&d_slab_buf, num_buckets * 512));
    CUDA_CHECK_ERROR(cudaMalloc(&d_slab_pools, utils::gridSize() * sizeof(SlabPool)));

    create_slab_pools<<<utils::gridSize(), 32>>>
        (d_slab_pools, d_slab_buf, num_slabs * 512);
    
    slab_kernel<<<utils::gridSize(), utils::blockSize()>>>
        (ref, d_buf, num_buckets, j, d_slab_pools);

    n_buckets_out = num_buckets;
    return d_buf;
}

int main()
{
    uint32_t result[128];
    Fasta fasta("data/reference.fa");
    uint32_t* d_ref = fasta.toGpuCompressed();
    uint32_t num_buckets;
    uint32_t* d_table_buf = slab(d_ref, fasta.size(), num_buckets);
    hashtable_walk(d_table_buf, num_buckets, result);
    for (int i = 0; i < 128; ++i)
    {
        cout << "result[" << i << "] = " << result[i] << endl;
    }
}