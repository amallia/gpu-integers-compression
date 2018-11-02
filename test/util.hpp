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
#include <vector>
#include <random>
#include <limits>
#include <algorithm>

std::vector<uint32_t> generate_random_vector(size_t n, uint32_t max_value = std::numeric_limits<uint32_t>::max()) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<uint32_t> values(n);
    std::uniform_int_distribution<> dis(uint32_t(0), max_value);
    std::generate(values.begin(), values.end(), [&](){ return dis(gen); });
    return values;
}

void make_strict(std::vector<uint32_t> &vec){
    for (size_t i = 1; i < vec.size(); ++i)
    {
        vec[i] += vec[i-1];
    }
}

void delta_encode(std::vector<uint32_t> &vec){
    for (size_t i = 1; i < vec.size(); ++i)
    {
        vec[i] -= vec[i-1];
    }
}