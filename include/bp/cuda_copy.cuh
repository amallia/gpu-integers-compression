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

#include "cuda_common.hpp"

#include "bit_stream/bit_istream.hpp"
#include "bit_stream/bit_ostream.hpp"

namespace cuda_copy {

__global__
void kernel_decode(uint32_t *out, const uint32_t *in, size_t limit)
{
   size_t index = blockIdx.x * blockDim.x + threadIdx.x;
   if(index < limit)
      out[index] = in[index];
}

static void decode(uint32_t *out, const uint8_t *in, size_t n) {

  uint8_t *d_in;
  CUDA_CHECK_ERROR(cudaMalloc((void **)&d_in, n*4 * sizeof(uint8_t)));
  CUDA_CHECK_ERROR(cudaMemcpy(d_in, in, n*4 * sizeof(uint8_t), cudaMemcpyHostToDevice));

  uint32_t *d_decoded;
  CUDA_CHECK_ERROR(cudaMalloc((void **)&d_decoded, n * sizeof(uint32_t)));

  dim3 block(32);
  dim3 grid(ceil(n/32));

  kernel_decode<<<grid, block>>>(d_decoded, reinterpret_cast<const uint32_t*>(d_in), n);

  CUDA_CHECK_ERROR(cudaMemcpy(out, d_decoded, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK_ERROR(cudaFree(d_in));
  CUDA_CHECK_ERROR(cudaFree(d_decoded));
}




} // namespace cuda_bp