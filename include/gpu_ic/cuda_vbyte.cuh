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

#include <cuda.h>
#include <numeric>

#include "cub/cub.cuh"

#include "utils/bit_ostream.hpp"
#include "utils/cuda_utils.hpp"

namespace cuda_vbyte {

template <size_t block_size = 128>
static size_t encode(uint8_t *out, const uint32_t *in, size_t n) {

    bit_ostream bw_offset(out);

    size_t block_num  = ceil((double)n / block_size);
    size_t offset_len = 4 * block_num + 4;
    size_t size       = 0;

    bw_offset.write(0, 32);
    size_t i;
    for (i = 0; i + block_size < n; i += block_size) {
        bit_ostream bw_block(out + offset_len);
        for (int j = i; j < i + block_size and j < n; ++j) {
            const auto value = in[j];
            if (value < (1U << 8)) {
                bw_block.write(0, 2);
            } else if (value < (1U << 16)) {
                bw_block.write(1, 2);
            } else if (value < (1U << 24)) {
                bw_block.write(2, 2);
            } else {
                bw_block.write(3, 2);
            }
        }
        for (int j = i; j < i + block_size and j < n; ++j) {
            const auto value = in[j];
            if (value < (1U << 8)) {
                bw_block.write(value, 8);
            } else if (value < (1U << 16)) {
                bw_block.write(value, 16);
            } else if (value < (1U << 24)) {
                bw_block.write(value, 24);
            } else {
                bw_block.write(value, 32);
            }
        }
        auto padding = 32 - (bw_block.size() % 32);
        bw_block.write(0, padding);
        size += ceil((double)bw_block.size() / 8);
        bw_offset.write(size, 32);
        offset_len += ceil((double)(bw_block.size()) / 8);
    }
    bit_ostream bw_block(out + offset_len);
    auto s = i;
    size_t bit   = 0;
    while(s<n) {
        const auto value = in[s];
        size_t b = utils::bits(value);
        bit= std::max(bit, b);
        s+=1;
    }
    bw_block.write(bit, 32);
    while(i<n) {
        const auto value = in[i];
        bw_block.write(value, bit);
        i+=1;
    }
    // auto padding = 32 - (bw_block.size() % 32);
    // bw_block.write(0, padding);
    size += ceil((double)bw_block.size() / 8);
    offset_len += ceil((double)(bw_block.size()) / 8);

    return offset_len;
}
template <size_t block_size = 128>
__global__ void kernel_decode_vbyte(uint32_t *      out,
                              const uint32_t *in,
                              size_t          n,
                              const uint32_t *offsets) {

    size_t     index  = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t   offset = offsets[blockIdx.x] / 4;
    if ((blockIdx.x +1) * block_size  < n) {
        __shared__ uint32_t min_offsets[block_size + 1];
        min_offsets[0] = 0;
        min_offsets[threadIdx.x + 1] = (extract(in + offset, threadIdx.x * 2, 2) + 1) * 8;
        __syncthreads();

        typedef cub::BlockScan<uint32_t, block_size>       BlockScan;
        __shared__ typename BlockScan::TempStorage temp_storage;
        BlockScan(temp_storage)
            .InclusiveSum(min_offsets[threadIdx.x + 1], min_offsets[threadIdx.x + 1]);
        __syncthreads();
        uint32_t bit = min_offsets[threadIdx.x + 1] - min_offsets[threadIdx.x];
        uint32_t header_len = 2 * (block_size/32);
        out[index]   = extract(in + offset + header_len, min_offsets[threadIdx.x], bit);
    } else {
        uint8_t  bit_size = *(in + offset);
        out[index]        = extract(in + offset+1, threadIdx.x * bit_size, bit_size);
    }
}
template <size_t block_size = 128>
static void decode(uint32_t *d_out, const uint8_t *d_in, size_t n) {
    size_t         block_num  = ceil((double)n / block_size);
    size_t         offset_len = 4 * block_num + 4;
    const uint8_t *d_payload  = d_in + offset_len;
    kernel_decode_vbyte<block_size><<<ceil((double)n / block_size), block_size>>>(
        d_out,
        reinterpret_cast<const uint32_t *>(d_payload),
        n,
        reinterpret_cast<const uint32_t *>(d_in));
}

} // namespace cuda_vbyte
