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

#include "cub/cub.cuh"

#include "utils/bit_ostream.hpp"
#include "utils/cuda_utils.hpp"

namespace cuda_vbyte {

template <size_t block_size = 32>
static size_t encode(uint8_t *out, const uint32_t *in, size_t n) {

    bit_ostream bw_offset(out);

    size_t block_num  = ceil(n / block_size);
    size_t header_len = ceil((2 * block_size) / 8);
    size_t offset_len = 4 * block_num + 4;
    size_t size       = 0;

    bw_offset.write(0, 32);
    for (size_t i = 0; i < n; i += block_size) {
        bit_ostream bw_block(out + offset_len);
        bit_ostream bw_payload(out + offset_len + header_len);
        auto        block_len = 0;
        for (int j = i; j < i + block_size; ++j) {
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
            block_len += 2;
        }
        for (int j = i; j < i + block_size; ++j) {
            const auto value = in[j];
            if (value < (1U << 8)) {
                bw_block.write(value, 8);
                block_len += 8;
            } else if (value < (1U << 16)) {
                bw_block.write(value, 16);
                block_len += 16;
            } else if (value < (1U << 24)) {
                bw_block.write(value, 24);
                block_len += 24;
            } else {
                bw_block.write(value, 32);
                block_len += 32;
            }
        }
        auto padding = 32 - (block_len % 32);
        bw_block.write(0, padding);
        size += block_len + padding;

        bw_offset.write(ceil(size / 8), 32);
        offset_len += ceil((block_len + padding) / 8);
    }

    return offset_len;
}

__global__ void kernel_decode(uint32_t *      out,
                              const uint32_t *in,
                              size_t          n,
                              const uint32_t *offsets) {

    size_t     index  = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t   offset = offsets[blockIdx.x] / 4;
    __shared__ uint32_t min_offsets[32 + 1];

    min_offsets[threadIdx.x + 1] = (extract(in + offset, threadIdx.x * 2, 2) + 1) * 8;
    __syncthreads();

    typedef cub::BlockScan<uint32_t, 32>       BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    BlockScan(temp_storage)
        .InclusiveSum(min_offsets[threadIdx.x + 1], min_offsets[threadIdx.x + 1]);
    __syncthreads();

    if (index < n) {
        uint32_t bit = min_offsets[threadIdx.x + 1] - min_offsets[threadIdx.x];
        out[index]   = extract(in + offset + 2, min_offsets[threadIdx.x], bit);
    }
}
template <size_t block_size = 32>
static void decode(uint32_t *d_out, const uint8_t *d_in, size_t n) {
    size_t         block_num  = ceil(n / block_size);
    size_t         offset_len = 4 * block_num + 4;
    const uint8_t *d_payload  = d_in + offset_len;
    kernel_decode<<<ceil(n / block_size), block_size>>>(
        d_out,
        reinterpret_cast<const uint32_t *>(d_payload),
        n,
        reinterpret_cast<const uint32_t *>(d_in));
}

} // namespace cuda_vbyte