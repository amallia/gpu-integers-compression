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

#include <iostream>

#include <utility>
#include <numeric>
#include <algorithm>
#include <x86intrin.h>

#include "bit_vector/bit_vector.hpp"

namespace cuda_bp {

namespace details {

template <class T> __attribute__((const)) T *padTo32bits(T *inbyte) {
  return reinterpret_cast<T *>((reinterpret_cast<uintptr_t>(inbyte) + 3) & ~3);
}

bool bsr64(unsigned long *const index, const uint64_t mask) {
#if defined(__GNUC__) || defined(__clang__)
  if (mask) {
    *index = (unsigned long)(63 - __builtin_clzll(mask));
    return true;
  } else {
    return false;
  }
#elif defined(_MSC_VER)
  return _BitScanReverse64(index, mask) != 0;
#else
#error Unsupported platform
#endif
}

inline uint8_t msb(uint64_t x, unsigned long &ret) { return bsr64(&ret, x); }

inline uint8_t msb(uint64_t x) {
  assert(x);
  unsigned long ret = -1U;
  msb(x, ret);
  return (uint8_t)ret;
}

inline size_t ceil_log2(const uint64_t x) {
  assert(x > 0);
  return (x > 1) ? msb(x - 1) + 1 : 0;
}

} // namespace details


/*
 * Format:
 * ________________________________________________________
 * | Offset to payload (unary) | Header (unary) | Payload |
 * --------------------------------------------------------
 *
 */
template<size_t block_size = 32>
static size_t encode(uint8_t *out, const uint32_t *in, size_t n) {
  auto blocks = std::ceil((double)n / block_size);
  std::vector<size_t> bits(blocks,0);
  for (size_t i = 0; i < n; ++i) {
    auto value = in[i];
    auto bit = details::ceil_log2(value);
    auto b = i/block_size;
    bits[b] = std::max(bit, bits[b]);
  }
  size_t offset = std::accumulate(bits.begin(), std::prev(bits.end()), 0) * block_size;
  offset += (n % block_size) * bits.back();
  offset = ((offset + 32ULL - 1) & -32ULL) / 8;
  auto header = out+offset;
  for(auto b : bits) {
    // Unary encode the header?
  }
  return offset;
}

/*
 * Procedure:
 *  - Read offsets
 *  - Transfer payload to GPU
 *  - Read header
 *  - all decode kernel
 */
static void decode(uint32_t *out, const uint8_t *in, size_t n) {
    // auto header_len = read_unary(in);
    // auto payload_len = read_unary(in);

    // auto payload = in + offset;
    // cudaMalloc(d_payload, payload_len);
    //  cudaMemcopy(d_payload, payload, payload_len);

    // cudaMalloc(d_decoded, payload_len);

    // while(skip != payload_len){
    //    auto bit = read_unary(in);
    //    skip += decode<<<block_size, grid_size>>>(payload + skip, len, bit);
    // }

    //  cudaMemcopy(out, d_decoded, payload_len);
}

} // namespace cuda_bp