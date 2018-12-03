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

#include "bit_stream/bit_istream.hpp"
#include "bit_stream/bit_ostream.hpp"
#include "bp/cuda_common.hpp"

namespace cuda_bp {

namespace details {

template <class T>
__attribute__((const)) T *padTo32bits(T *inbyte) {
    return reinterpret_cast<T *>((reinterpret_cast<uintptr_t>(inbyte) + 3) & ~3);
}

static inline size_t bits(const size_t v) { return v == 0 ? 0 : 64 - __builtin_clzll(v); }


static uint32_t maxbits(const uint32_t *in) {
    auto BlockSize = 32;
    uint32_t accumulator = 0;
    for (uint32_t k = 0; k < BlockSize; ++k) {
        accumulator |= in[k];
    }
    return bits(accumulator);
}

size_t size_align8(size_t size) { return (size + 8ULL - 1) & ~(8ULL - 1); }

} // namespace details

static const uint32_t MiniBlockSize = 32;
static const uint32_t HowManyMiniBlocks = 4;

static size_t encode(uint8_t *out, const uint32_t *in, size_t n) {
    bit_ostream    bw(out);
    bw.write(n, 32);

    uint32_t              Bs[HowManyMiniBlocks];
    const uint32_t *const final = in + n;
    for (; in + HowManyMiniBlocks * MiniBlockSize <= final; in += HowManyMiniBlocks * MiniBlockSize) {
        for (uint32_t i = 0; i < HowManyMiniBlocks; ++i) {
            Bs[i] = details::maxbits(in + i * MiniBlockSize);
            std::cerr << Bs[i] << std::endl;
        }
        bw.write((Bs[0] << 24) | (Bs[1] << 16) | (Bs[2] << 8) | Bs[3], 32);
        for (uint32_t i = 0; i < HowManyMiniBlocks; ++i) {
            bw.write(*(in + i * MiniBlockSize), Bs[i]);
        }
    }
    if (in < final) {
    //     size_t   howmany = ((final - in) + MiniBlockSize - 1) / MiniBlockSize;
    //     uint32_t zeroedIn[HowManyMiniBlocks * MiniBlockSize];
    //     if (!divisibleby(length, BlockSize)) {
    //         // We treat the rest of the block as 0
    //         assert(final < in + HowManyMiniBlocks * MiniBlockSize);
    //         memset(&zeroedIn[0], 0, HowManyMiniBlocks * MiniBlockSize * sizeof(uint32_t));
    //         memcpy(&zeroedIn[0], in, (final - in) * sizeof(uint32_t));
    //         assert(zeroedIn[HowManyMiniBlocks * MiniBlockSize - 1] == 0);
    //         assert(zeroedIn[(final - in)] == 0);
    //         in = zeroedIn;
    //     }
    //     uint32_t tmpinit = init;
    //     memset(&Bs[0], 0, HowManyMiniBlocks * sizeof(uint32_t));
    //     for (uint32_t i = 0; i < howmany; ++i) {
    //         Bs[i] = DeviceNoDeltaBlockPacker::maxbits(in + i * MiniBlockSize, tmpinit);
    //     }
    //     *out++ = (Bs[0] << 24) | (Bs[1] << 16) | (Bs[2] << 8) | Bs[3];
    //     for (uint32_t i = 0; i < howmany; ++i) {
    //         DeviceNoDeltaBlockPacker::packblockwithoutmask(
    //             in + i * MiniBlockSize, out, Bs[i], init);
    //         out += Bs[i];
    //     }
    }
    return ceil(bw.size()/8);
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

__global__ void kernel_decode(uint32_t *out, const uint32_t *in, size_t bit) {
    size_t          index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t *ptr   = in + (index * bit / 32);
    const uint32_t  mask  = (1UL << bit) - 1;

    int            lastbit       = (index * bit % 32) + bit;
    int            stuff         = lastbit - 32;
    size_t         overflow      = max(0, stuff) % 32;
    bool           isOverflowing = lastbit % 31 < (index * bit % 32);
    const uint32_t overbuff      = *(ptr + isOverflowing);

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
    out[index] = (*ptr >> (index * bit % 32)) & mask | (overbuff % (1U << overflow))
                                                           << (bit - overflow);
    // if(index == 7){
    //  printf("%d\n", index);
    //  printf("%d\n", out[index]);

    // }
}

void decode(uint32_t *out, const uint8_t *in, size_t bit, size_t n) {
    bit_istream br(in);
    for (size_t i = 0; i < n; ++i) {
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
    bit_istream    br(in);
    auto           header_len  = br.read(32);
    auto           payload_len = br.read(32);
    const uint8_t *payload     = in + header_len + 8;
    uint8_t *      d_payload;
    CUDA_CHECK_ERROR(cudaMalloc((void **)&d_payload, payload_len * sizeof(uint8_t)));
    CUDA_CHECK_ERROR(
        cudaMemcpy(d_payload, payload, payload_len * sizeof(uint8_t), cudaMemcpyHostToDevice));

    uint32_t *d_decoded;
    CUDA_CHECK_ERROR(cudaMalloc((void **)&d_decoded, n * sizeof(uint32_t)));

    auto decoded = 0;
    auto skip    = 0;

    while (n - decoded >= 32) {
        // auto bit = br.read_unary();
        auto bit = br.read(8);
        // std::cerr << bit << std::endl;;

        kernel_decode<<<32, 1>>>(
            d_decoded + decoded, reinterpret_cast<const uint32_t *>(d_payload + skip), bit);
        skip += round_up_div((32 * bit), 8);
        decoded += 32;
    }

    // auto bit = br.read_unary();
    auto bit = br.read(8);
    decode(out + decoded, payload + skip, bit, n - decoded);

    CUDA_CHECK_ERROR(
        cudaMemcpy(out, d_decoded, (n / 32 * 32) * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    cudaFree(d_payload);
    cudaFree(d_decoded);
}

} // namespace cuda_bp