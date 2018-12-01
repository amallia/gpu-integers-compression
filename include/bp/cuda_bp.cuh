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
#include <algorithm>

#include "bit_stream/bit_istream.hpp"
#include "bit_stream/bit_ostream.hpp"
#include "bp/cuda_common.hpp"

namespace cuda_bp {

namespace details {

template <class T> __attribute__((const)) T *padTo32bits(T *inbyte) {
  return reinterpret_cast<T *>((reinterpret_cast<uintptr_t>(inbyte) + 3) & ~3);
}

static inline size_t bits(const size_t v) {
        return v == 0 ? 0 : 64 - __builtin_clzll(v);
}


size_t size_align8(size_t size) { return (size + 8ULL - 1) & ~(8ULL - 1); }

} // namespace details

/*
 * Format:
 * _______________________________________________________________________________
 * | Payload begin (32-bit) | Payload length (32-bit) | Header (unary) | Payload |
 * -------------------------------------------------------------------------------
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
    // std::cerr << value << std::endl;
    auto bit = details::bits(value);
    // std::cerr << value << " " << bit << std::endl;
    auto b = i / block_size;
    bits[b] = std::max(bit, bits[b]);
  }
  auto bits_sum = std::accumulate(bits.begin(), std::prev(bits.end()), 0);
  size_t offset = bits_sum * block_size;
  bits_sum += bits.back();
  offset += bits.back() * ( (n / block_size) * block_size + (n % block_size));
  offset = round_up_div(details::size_align8(offset), 8);

  // auto header_len = round_up_div(details::size_align8(bits_sum + bits.size()), 8);
  auto header_len = bits.size();
  bw.write(header_len, 32);
  bw.write(offset, 32);
  for (auto b : bits) {
    // bw.write_unary(b);
    bw.write(b, 8);

  }

  // bw.write(0, details::size_align8(bits_sum + bits.size()) - bits_sum - bits.size());
  for (size_t i = 0; i < n; ++i) {
    auto value = in[i];
    auto b = i / block_size;
    bw.write(value, bits[b]);
  }

  // std::cerr << 8 + header_len + offset << std::endl;
  return 8 + header_len + offset;
}

__host__ __device__
void printBinary(unsigned long long myNumber)
{
    int numberOfBits = sizeof(unsigned long long)*8;
    for (int i=numberOfBits-1; i>=0; i--) {
        bool isBitSet = (myNumber & (1ULL<<i));
        if (isBitSet) {
          printf("1");
        } else {
          printf("0");
        }
    }
    printf("\n");
}

__global__
void kernel_decode(uint32_t *out, const uint32_t *in, size_t bit)
{
   size_t index = blockIdx.x * blockDim.x + threadIdx.x;
   const uint32_t* ptr = in+(index * bit/32);
   const uint32_t mask = (1UL << bit) - 1;

    int lastbit = (index * bit%32)+bit;
    int stuff = lastbit - 32;
    size_t overflow = max(0,  stuff) % 32;
    bool isOverflowing = lastbit % 31 < (index * bit%32);
    const uint32_t overbuff = *(ptr + isOverflowing);

    //    if(index == 31){
    // printBinary(*(in));
    // printBinary(*(in+1));
    // printBinary(*(in+2));
    // printBinary(*(in+3));
    // printBinary(*(in+4));
    // printBinary(*(in+5));

//     // printBinary(*ptr);
//     // printBinary(*(ptr+1));

   //  printf("%d\n", stuff);
   // printf("%d\n", lastbit);

   //  printf("%d\n", isOverflowing);
   // //  printf("%d\n", (32 - lastBitInPacked));
   //  printBinary(overbuff);
   //  printf("%d\n", overflow);
   //  printf("%d\n", index * bit);
   //  printf("%d\n", lastbit);
   //  printf("%d\n", (index * bit%32));

   //  printf("%d\n", bit);


   //  printBinary((overbuff % (1U << overflow)));

    // printf("%d\n", (index * bit%32));

    // printBinary(((overbuff & (1U << overflow)) << (bit - overflow - 1)));
// }


    // size_t bits_in_first = firstBit % 32;
    out[index] = (*ptr >> (index * bit%32)) & mask
    |(overbuff % (1U << overflow)) << (bit - overflow);
    // if(index == 7){
    //  printf("%d\n", index);
    //  printf("%d\n", out[index]);

    // }

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
 *  - call decode kernel
 */
static void decode(uint32_t *out, const uint8_t *in, size_t n) {
  bit_istream br(in);
  auto header_len = br.read(32);
  auto payload_len = br.read(32);
  const uint8_t * payload = in + header_len + 8;
  uint8_t *d_payload;
  CUDA_CHECK_ERROR(cudaMalloc((void **)&d_payload, payload_len * sizeof(uint8_t)));
  CUDA_CHECK_ERROR(cudaMemcpy(d_payload, payload, payload_len * sizeof(uint8_t), cudaMemcpyHostToDevice));

  uint32_t *d_decoded;
  CUDA_CHECK_ERROR(cudaMalloc((void **)&d_decoded, n * sizeof(uint32_t)));

  auto decoded = 0;
  auto skip = 0;

  while (n - decoded >= 32) {
    // auto bit = br.read_unary();
    auto bit = br.read(8);
  // std::cerr << bit << std::endl;;

    kernel_decode<<<32, 1>>>(d_decoded+decoded, reinterpret_cast<const uint32_t*>(d_payload + skip), bit);
    skip += round_up_div((32*bit), 8);
    decoded += 32;
  }

  // auto bit = br.read_unary();
  auto bit = br.read(8);
  decode(out+decoded, payload + skip, bit, n-decoded);


  CUDA_CHECK_ERROR(cudaMemcpy(out, d_decoded, (n/32 *32) * sizeof(uint32_t), cudaMemcpyDeviceToHost));

  cudaFree(d_payload);
  cudaFree(d_decoded);
}




} // namespace cuda_bp