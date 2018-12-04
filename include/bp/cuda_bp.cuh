/**
 * Copyright 2018-present Antonio Mallia <me@antoniomallia.it>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <utility>
#include <x86intrin.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "bit_stream/bit_istream.hpp"
#include "bit_stream/bit_ostream.hpp"
#include "bp/cuda_common.hpp"

namespace cuda_bp {

namespace details {

static inline size_t bits(const size_t v) { return v == 0 ? 0 : 64 - __builtin_clzll(v); }

size_t size_align8(size_t size) { return (size + 8ULL - 1) & ~(8ULL - 1); }

} // namespace details

/*
 * Format:
 * _______________________________________________________________________________
 * | Payload begin (32-bit) | Payload length (32-bit) | Header (8-bit) | Payload |
 * -------------------------------------------------------------------------------
 * IDEA: fixed 8 bits unary encoded words for the bit-length.
 * popcount 8bit https://stackoverflow.com/questions/17518774/sse-4-popcount-for-16-8-bit-values
 * then lunch 16 kernels
 *
 */
template <size_t block_size = 32>
static size_t encode(uint8_t *out, const uint32_t *in, size_t n) {
    bit_ostream bw(out);

    auto                blocks = std::ceil((double)n / block_size);
    std::vector<size_t> bits(blocks, 0);
    for (size_t i = 0; i < n; ++i) {
        auto value = in[i];
        // std::cerr << value << std::endl;
        auto bit = details::bits(value);
        // std::cerr << value << " " << bit << std::endl;
        auto b  = i / block_size;
        bits[b] = std::max(bit, bits[b]);
    }
    auto   bits_sum = std::accumulate(bits.begin(), std::prev(bits.end()), 0);
    size_t offset   = bits_sum * block_size;
    bits_sum += bits.back();
    offset += bits.back() * ((n / block_size) * block_size + (n % block_size));
    offset = round_up_div(details::size_align8(offset), 8);

    // auto header_len = round_up_div(details::size_align8(bits_sum + bits.size()), 8);
    auto header_len = bits.size();
    // bw.write(offset, 32);
    for (auto b : bits) {
        // bw.write_unary(b);
        bw.write(b, 8);
    }

    // bw.write(0, details::size_align8(bits_sum + bits.size()) - bits_sum - bits.size());
    for (size_t i = 0; i < n; ++i) {
        auto value = in[i];
        auto b     = i / block_size;
        bw.write(value, bits[b]);
    }

    // std::cerr << 8 + header_len + offset << std::endl;
    return  header_len + offset;
}

__host__ __device__ void printBinary(unsigned long long myNumber) {
    int numberOfBits = sizeof(unsigned long long) * 8;
    for (int i = numberOfBits - 1; i >= 0; i--) {
        bool isBitSet = (myNumber & (1ULL << i));
        if (isBitSet) {
            printf("1");
        } else {
            printf("0");
        }
    }
    printf("\n");
}

__device__ uint32_t extract(const uint32_t *in, size_t index, size_t bit) {
    int      firstBit                = bit * index;
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

__global__ void kernel_extract_bits(uint32_t *out, const uint32_t *in, size_t n) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < n) {
      uint32_t bit_size = 8;
      uint32_t offset = blockIdx.x * 8;
      out[index] = extract(in+offset, threadIdx.x, bit_size);
    }
}

__global__ void kernel_decode(uint32_t *out, const uint32_t *in, size_t n,  const uint8_t* bit_sizes, uint32_t* offsets) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < n) {
      uint8_t bit_size = bit_sizes[blockIdx.x];
      uint32_t offset = offsets[blockIdx.x];
      out[index] = extract(in+offset, threadIdx.x, bit_size);
    }
}


static void decode(uint32_t *d_out, const uint8_t *d_in, size_t n) {
    size_t           header_len  = ceil(n/32);


    uint32_t *      d_bits_extr;
    CUDA_CHECK_ERROR(cudaMalloc((void **)&d_bits_extr, ceil(n/32) * sizeof(uint32_t)));

    uint32_t *     d_offsets;
    CUDA_CHECK_ERROR(cudaMalloc((void **)&d_offsets,   ceil(n/32) * sizeof(uint32_t)));

    kernel_extract_bits<<<ceil(n/32/32), 32>>>(d_bits_extr, reinterpret_cast<const uint32_t *>(d_in), ceil(n/32));

    thrust::device_ptr<uint32_t> dp_bit_sizes(d_bits_extr);
    thrust::device_ptr<uint32_t> dp_offsets(d_offsets);
    thrust::exclusive_scan(dp_bit_sizes, dp_bit_sizes+ceil(n/32), dp_offsets);

    const uint8_t *      d_payload = d_in + header_len;
    kernel_decode<<<ceil(n/32), 32>>>(d_out, reinterpret_cast<const uint32_t *>(d_payload), n, d_in, d_offsets);
    cudaFree(d_bits_extr);
    cudaFree(d_offsets);

}

} // namespace cuda_bp