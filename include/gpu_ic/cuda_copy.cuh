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

#include "utils/utils.hpp"

namespace cuda_copy {

__global__
void kernel_decode(uint32_t *out, const uint32_t *in, size_t limit)
{
   size_t index = blockIdx.x * blockDim.x + threadIdx.x;
   if(index < limit)
      out[index] = in[index];
}

static void decode(uint32_t *d_out, const uint8_t *d_in, size_t n) {
  kernel_decode<<<ceil(n/512), 512>>>(d_out, reinterpret_cast<const uint32_t*>(d_in), n);
}

} // namespace cuda_bp
