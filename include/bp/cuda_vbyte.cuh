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
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <utility>
#include <x86intrin.h>

#include "bit_stream/bit_istream.hpp"
#include "bit_stream/bit_ostream.hpp"
#include "bp/cuda_common.hpp"

namespace cuda_vbyte {

namespace details {

static inline size_t bits(const size_t v) { return v == 0 ? 0 : 64 - __builtin_clzll(v); }

size_t size_align8(size_t size) { return (size + 8ULL - 1) & ~(8ULL - 1); }

} // namespace details


template <size_t block_size = 16>
static size_t encode(uint8_t *out, const uint32_t *in, size_t n) {

    bit_ostream bw_offset(out);
    size_t block_num = ceil(n/block_size);
    size_t offset_len = 4* block_num;
    bit_ostream bw_header(out+offset_len);
    bit_ostream bw_payload(out +offset_len+ 2*block_size * block_num);
    size_t size = 0;
    for (size_t i = 0; i < n; i+=block_size) {
        bw_offset.write(size, 32);
        for (int j = i; j < i+block_size; ++j)
        {
            const auto value = in[j];
            if (value < (1U << 8)) {
                // std::cerr << value << " 0" << std::endl;
                bw_header.write(0, 2);
                bw_payload.write(value, 8);
                size+=10;
            } else if (value < (1U << 16)) {
                // std::cerr << value << " 1" << std::endl;
                bw_header.write(1, 2);
                bw_payload.write(value, 16);
                size+=18;
            } else if (value < (1U << 24)) {
                // std::cerr << value << " 2" << std::endl;
                bw_header.write(2, 2);
                bw_payload.write(value, 24);
                size+=26;
            } else {
                // std::cerr << value << " 3" << std::endl;
                bw_header.write(3, 2);
                bw_payload.write(value, 32);
                size+=34;
            }
        }
    }

    return offset_len + ceil(size/8);
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

__device__ uint32_t extract2(const uint32_t *in, size_t index, uint32_t *offsets) {
    uint32_t offset = offsets[index];
    uint32_t bit =    offsets[index+1] - offsets[index];

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


__global__ void kernel_decode(
    uint32_t *out, const uint32_t *in, size_t n, const uint32_t* offsets) {

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t offset = offsets[blockIdx.x];
    __shared__ uint32_t min_offsets[16+1];

    min_offsets[index+1] = (extract(in + offset, threadIdx.x, 2)+1)*8;
     __syncthreads();

    if(threadIdx.x == 0) {
        for (int i = 1; i < 16+1; ++i)
        {
            min_offsets[i] += min_offsets[i-1];
        }
    }
     __syncthreads();

    // // prefix_sum(min_offsets, min_offsets, blockIdx.x);
    if (index < n) {
        out[index] = extract2(in + offset + 32, threadIdx.x, min_offsets);
    }
}
template <size_t block_size = 16>
static void decode(uint32_t *d_out, const uint8_t *d_in, size_t n) {
    size_t block_num = ceil(n / block_size);
    size_t offset_len = 4*block_num;
    const uint8_t *d_payload = d_in + offset_len;
    kernel_decode<<<ceil(n / block_size), block_size>>>(d_out, reinterpret_cast<const uint32_t *>(d_payload), n, reinterpret_cast<const uint32_t *>(d_in));
}

} // namespace cuda_vbyte