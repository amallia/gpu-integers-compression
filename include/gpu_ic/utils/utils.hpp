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
#include <stdexcept>

namespace utils {

template <class T>
inline void do_not_optimize_away(T&& datum) {
    asm volatile("" : "+r" (datum));
}

inline void delta_encode(uint32_t *in, size_t n) {
    for (size_t i = n - 1; i > 0; --i) {
        in[i] -= in[i - 1];
    }
}

inline void delta_decode(uint32_t *in, size_t n) {
    for (size_t i = 1; i < n; ++i) {
        in[i] += in[i - 1];
    }
}

/*
 * Computes the number of bits required to store the given integer value.
 */
inline constexpr uint_fast8_t bits(size_t value) {
    return value == 0 ? 1U : (64 - __builtin_clzll(value));
}

} // namespace utils