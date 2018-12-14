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
