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

static void decode(uint32_t *d_out, const uint8_t *d_in, size_t n) {
  thrust::device_ptr<const uint32_t> dp_in(reinterpret_cast<const uint32_t*>(d_in));
  thrust::device_ptr<uint32_t> dp_out(d_out);
  thrust::inclusive_scan(dp_in, dp_in+n, dp_out);
}


} // namespace cuda_bp
