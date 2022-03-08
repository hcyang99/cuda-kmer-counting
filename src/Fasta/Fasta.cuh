#pragma once
#include <string>

class Fasta
{
    public:
    /**
     * @brief Constructor, reads fasta to CPU memory
     * @param path Path to fasta file
     */
    Fasta(std::string path);

    /**
     * @brief Send string to GPU and compress there
     * @return Pointer to compressed sequence on GPU
     */
    uint32_t* toGpuCompressed();

    uint32_t size() const {return sz;}

    protected:
    std::string buffer;
    uint32_t sz;
};