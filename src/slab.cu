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
uint32_t* slab(uint32_t* ref, size_t n_bps, size_t& n_buckets_out)
{
    size_t n_jobs = n_bps - 127;
    JobQueue* j = new_job(n_jobs, utils::batchSize());
    size_t num_buckets = n_jobs * utils::slabFactor() / 14;
    size_t num_slabs = (n_jobs / 14 / utils::gridSize() + 1) * utils::gridSize();

    uint32_t* d_buf;
    uint32_t* d_slab_buf;
    SlabPool* d_slab_pools;
    cerr << "Allocating " << num_buckets << " buckets" << endl;
    cerr << "Allocating " << num_slabs << " slabs" << endl;

    CUDA_CHECK_ERROR(cudaMalloc(&d_buf, num_buckets * 512));
    CUDA_CHECK_ERROR(cudaMalloc(&d_slab_buf, num_slabs * 512 * 2));
    CUDA_CHECK_ERROR(cudaMalloc(&d_slab_pools, utils::gridSize() * sizeof(SlabPool)));
    CUDA_CHECK_ERROR(cudaMemset(d_buf, 0, num_buckets * 512));
    CUDA_CHECK_ERROR(cudaMemset(d_slab_buf, 0, num_slabs * 512 * 2));

    cerr << "Allocated memory for slabs: " << d_slab_buf << " to " << d_slab_buf + num_slabs * 128 << endl;

    create_slab_pools<<<utils::gridSize(), 32>>>
        (d_slab_pools, d_slab_buf, num_slabs * 512);

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    
    slab_kernel<<<utils::gridSize(), utils::blockSize()>>>
        (ref, d_buf, num_buckets, j, d_slab_pools);

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    n_buckets_out = num_buckets;
    return d_buf;
}

void printGpuProperties () {
    int nDevices;

    // Store the number of available GPU device in nDevicess
    cudaError_t err = cudaGetDeviceCount(&nDevices);

    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaGetDeviceCount failed!\n");
        exit(1);
    }

    // For each GPU device found, print the information (memory, bandwidth etc.)
    // about the device
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Device memory: %lu\n", prop.totalGlobalMem);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
}

int main()
{
     printGpuProperties();

    uint32_t result[128];
    Fasta fasta("data/cut.fa");
    cerr << "Reading " << fasta.size() << " base pairs" << endl;

    uint32_t* d_ref = fasta.toGpuCompressed();
    size_t num_buckets;
    uint32_t* d_table_buf = slab(d_ref, fasta.size(), num_buckets);

    cerr << "Walking through " << num_buckets << " buckets" << endl;
   
    hashtable_walk(d_table_buf, num_buckets, result);
    for (int i = 1; i < 128; ++i)
    {
        cout << "result[" << i << "] = " << result[i] << endl;
    }
}