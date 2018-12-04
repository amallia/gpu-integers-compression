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
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "cuda_common.hpp"

namespace cuda_delta {


static size_t encode(uint8_t *out, const uint32_t *in, size_t n) {
  uint32_t * out_p = reinterpret_cast<uint32_t *>(out);
  out_p[0] = in[0];
  for (size_t i = 1; i < n; ++i) {
    out_p[i] = in[i] - in[i-1];
  }
  return n * sizeof(uint32_t);
}

static void decode(uint32_t *d_out, const uint8_t *d_in, size_t n) {
  thrust::device_ptr<const uint32_t> dp_in(reinterpret_cast<const uint32_t*>(d_in));
  thrust::device_ptr<uint32_t> dp_out(d_out);
  thrust::inclusive_scan(dp_in, dp_in+n, dp_out);
}




} // namespace cuda_bp
