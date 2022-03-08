#include "Fasta.cuh"
#include "utils.cuh"
#include <iostream>
#include <fstream>
#include <stdexcept>
using namespace std;

__host__ 
Fasta::Fasta(string path)
{
    ifstream f(path);
    if (!f)
        throw runtime_error("Cannot open file");
    
    string line;
    do 
    {
        getline(f, line);
        if (line.size() != 0 && line[0] != '>' && line[0] != ';')
        {
            this->buffer += line;
        }
    } 
    while(!f.eof() && !(line.size() != 0 && line[0] == '>'));
    f.close();

    this->sz = buffer.size();

    int remainder = this->buffer.size() % 16;
    int pad = remainder == 0 ? 0 : 16 - remainder;

    // pad with 'A' for compressing
    this->buffer += string(pad, 'A');
}

using utils::byte_32;
using utils::byte_4;

__device__ __forceinline__
uint32_t compress16(char* s)
{
    uint32_t result = 0;
    uint32_t shift = 0;
    for (int i = 0; i < 16; ++i)
    {
        uint32_t twoBitVal;
        switch(s[i]) 
        {
            case 'A': twoBitVal = 0; break;
            case 'C': twoBitVal = 1; break;
            case 'G': twoBitVal = 2; break;
            default: twoBitVal = 3; break;
        }
        result |= (twoBitVal << shift);
        shift += 2;
    }
    return result;
}

/**
 * @brief Compress input ACGT string each to 2-bit
 * @param dst Destination buffer
 * @param src Source string buffer
 */
__global__
void compressKernel(uint32_t* dst, uint32_t* src)
{
    char* src_char = reinterpret_cast<char*>(src);
    int tx = utils::global_thread_id();
    dst[tx] = compress16(src_char + 16 * tx);
}

__host__
uint32_t* Fasta::toGpuCompressed()
{
    uint32_t* d_buf;
    uint32_t* d_compressed;
    CUDA_CHECK_ERROR(cudaMalloc(&d_buf, this->buffer.size()));
    CUDA_CHECK_ERROR(cudaMalloc(&d_compressed, this->buffer.size() / 4));
    CUDA_CHECK_ERROR(cudaMemcpy(d_buf, &this->buffer[0], this->buffer.size(), cudaMemcpyHostToDevice));

    compressKernel<<<this->buffer.size() / 16, utils::blockSize()>>>
    (d_compressed, d_buf);

    cudaFree(d_buf);
    this->buffer.clear();

    return d_compressed;
}