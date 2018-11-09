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
#include <cassert>
#include <algorithm>
#include <cuda.h>
#include <numeric>
#include <utility>
#include <x86intrin.h>

#include "bit_stream/bit_istream.hpp"
#include "bit_stream/bit_ostream.hpp"

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


size_t size_align8(size_t size) { return (size + 8ULL - 1) & ~(8ULL - 1); }

} // namespace details

/*
 * Format:
 * ________________________________________________________
 * | Offset to payload (unary) | Header (unary) | Payload |
 * --------------------------------------------------------
 * IDEA: fixed 8 bits unary encoded words for the bit-length.
 * popcount 8bit https://stackoverflow.com/questions/17518774/sse-4-popcount-for-16-8-bit-values
 * then lunch 16 kernels
 *
 */
template <size_t block_size = 32>
static size_t encode(uint8_t *out, const uint32_t *in, size_t n) {
  bit_ostream bw(out);

  auto blocks = std::ceil((double)n / block_size);
  std::vector<size_t> bits(blocks, 0);
  for (size_t i = 0; i < n; ++i) {
    auto value = in[i];
    auto bit = details::ceil_log2(value);
    auto b = i / block_size;
    bits[b] = std::max(bit, bits[b]);
  }

  auto bits_sum = std::accumulate(bits.begin(), std::prev(bits.end()), 0);
  size_t offset = bits_sum * block_size;
  bits_sum += bits.back();
  offset += bits.back() * ( (n / block_size) * block_size + (n % block_size));
  offset = round_up_div(details::size_align8(offset), 8);

  // bw.write_unary(details::size_align8(bits_sum) / 8);
  // bw.write_unary(offset);
  bw.write(round_up_div(details::size_align8(bits_sum), 8), 32);
  bw.write(offset, 32);
  for (auto b : bits) {
    bw.write_unary(b);
  }

  bw.write(0, details::size_align8(bits_sum + bits.size()) - bits_sum - bits.size());


  for (size_t i = 0; i < n; ++i) {
    auto value = in[i];
    auto b = i / block_size;
    bw.write(value, bits[b]);
  }

  return offset;
}

__global__
void warmUpGPU()
{
  // do nothing
}


__global__
void kernel_decode(uint32_t *out, const uint64_t *in, size_t bit)
{
   size_t index = blockIdx.x * blockDim.x + threadIdx.x;
   const uint64_t mask = (1UL << bit) - 1;
   out[index] = (*(in+(index * bit/64)) >> index * bit%64) & mask;
}


void decode(uint32_t *out, const uint8_t *in, size_t bit, size_t n)
{
    bit_istream br(in);
    for (size_t i = 0; i < n; ++i)
    {
      out[i] = br.read(bit);
    }
}



/*
 * Procedure:
 *  - Read offsets
 *  - Transfer payload to GPU
 *  - Read header
 *  - all decode kernel
 */
static void decode(uint32_t *out, const uint8_t *in, size_t n) {
  warmUpGPU<<<1, 1>>>();

  bit_istream br(in);
  auto header_len = br.read(32);
  auto payload_len = br.read(32);
  auto payload = in + header_len + 8;
  uint8_t *d_payload;
  cudaMalloc((void **)&d_payload, payload_len * sizeof(uint8_t));
  cudaMemcpy(d_payload, payload, payload_len * sizeof(uint8_t),
             cudaMemcpyHostToDevice);

  uint32_t *d_decoded;
  cudaMalloc((void **)&d_decoded, n * sizeof(uint32_t));

  auto decoded = 0;
  auto skip = 0;
  while (n - decoded >= 32) {
    auto bit = br.read_unary();
    kernel_decode<<<32, 1>>>(d_decoded, reinterpret_cast<const uint64_t*>(d_payload + skip), bit);
    skip += round_up_div((32*bit), 8);
    decoded += 32;
  }

  auto bit = br.read_unary();
  decode(out+decoded, payload + skip, bit, n-decoded);


  cudaMemcpy(out, d_decoded, (n/32 *32) * sizeof(uint32_t), cudaMemcpyDeviceToHost);

  cudaFree(d_payload);
  cudaFree(d_decoded);
}




} // namespace cuda_bp