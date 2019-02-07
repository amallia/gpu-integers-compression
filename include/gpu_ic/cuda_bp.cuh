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
#include <cuda.h>
#include <numeric>
#include <utility>

#include "utils/bit_istream.hpp"
#include "utils/bit_ostream.hpp"
#include "utils/cuda_utils.hpp"
#include "utils/utils.hpp"

namespace cuda_bp {

template <size_t block_size = 32>
static size_t encode(uint8_t *out, const uint32_t *in, size_t n) {
    bit_ostream bw(out);

    auto                blocks = std::ceil((double)n / block_size);
    std::vector<size_t> bits(blocks, 0);
    for (size_t i = 0; i < n; ++i) {
        auto   value = in[i];
        size_t bit   = utils::bits(value);
        auto   b     = i / block_size;
        bits[b]      = std::max(bit, bits[b]);
    }
    bw.write(0, 32);
    uint32_t offset = 0;
    for (auto b : bits) {
        offset += b * block_size/32;
        bw.write(offset, 32);
    }
    for (size_t i = 0; i < n; ++i) {
        auto value = in[i];
        auto b     = i / block_size;
        bw.write(value, bits[b]);
    }
    return ceil((double)bw.size() / 8);
}

template <size_t block_size = 32>
__global__ void kernel_decode(uint32_t *      out,
                              const uint32_t *in,
                              size_t          n,
                              const uint32_t *offsets) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        uint8_t  bit_size = (offsets[blockIdx.x + 1] - offsets[blockIdx.x])*32/block_size;
        uint32_t offset   = offsets[blockIdx.x];
        out[index]        = extract(in + offset, threadIdx.x * bit_size, bit_size);
    }
}

template <size_t block_size = 32>
static void decode(uint32_t *d_out, const uint8_t *d_in, size_t n) {
    size_t         header_len = 4 * (ceil((double)n / block_size) + 1);
    const uint8_t *d_payload  = d_in + header_len;
    kernel_decode<block_size><<<ceil((double)n / block_size), block_size>>>(d_out,
                                        reinterpret_cast<const uint32_t *>(d_payload),
                                        n,
                                        reinterpret_cast<const uint32_t *>(d_in));
}

} // namespace cuda_bp
