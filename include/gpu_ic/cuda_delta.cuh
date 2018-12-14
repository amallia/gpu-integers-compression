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
#include <algorithm>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace cuda_delta {

static void decode(uint32_t *d_in, size_t n) {
  thrust::device_ptr<uint32_t> dp_in(d_in);
  thrust::inclusive_scan(dp_in, dp_in+n, dp_in);
}


} // namespace cuda_bp