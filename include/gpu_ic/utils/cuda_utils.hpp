#pragma once

#include <iostream>
#include <stdexcept>

#define CUDA_CHECK_ERROR(err) __cudaSafeCall(err, __FILE__, __LINE__)

inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
    if (cudaSuccess != err) {
        std::ostringstream stringStream;
        stringStream << "cudaSafeCall() failed at " << file << ":" << line << " : "
                     << cudaGetErrorString(err);
        throw(std::runtime_error(stringStream.str()));
    }
}

__device__ uint32_t extract(const uint32_t *in, size_t offset, size_t bit) {
    int      firstBit                = offset;
    int      lastBit                 = firstBit + bit - 1;
    uint32_t packed                  = in[firstBit / 32];
    int      firstBitInPacked        = firstBit % 32;
    uint32_t packedOverflow          = in[lastBit / 32];
    bool     isOverflowing           = lastBit % 32 < firstBitInPacked;
    int      lastBitInPackedOverflow = !isOverflowing ? -1 : lastBit % 32;
    uint32_t outFromPacked =
        ((packed >> firstBitInPacked) & (0xFFFFFFFF >> (32 - (bit - lastBitInPackedOverflow - 1))));
    uint32_t outFromOverflow = (packedOverflow & (0xFFFFFFFF >> (32 - lastBitInPackedOverflow - 1)))
                               << (bit - lastBitInPackedOverflow - 1);
    return outFromPacked | outFromOverflow;
}
