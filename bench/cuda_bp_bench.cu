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

#include "bp/cuda_bp.cuh"
#include "synthetic/uniform.hpp"
#include "benchmark/benchmark.h"

std::vector<uint32_t> generate_random_vector(size_t n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<uint32_t> values(n);
    std::uniform_int_distribution<> dis(uint32_t(0));
    std::generate(values.begin(), values.end(), [&](){ return dis(gen); });
    return values;
}

static void decode(benchmark::State &state) {
    while (state.KeepRunning()) {
        state.PauseTiming();
        auto n   = state.range(0);
        auto min = 1;
        auto max = state.range(0)+2;

        std::vector<uint32_t> values = generate_random_vector(state.range(0));
        std::vector<uint8_t>  buffer(values.size() * 8);
        cuda_bp::encode(buffer.data(), values.data(), values.size());
        std::vector<uint32_t> decoded_values(values.size());
        state.ResumeTiming();
        cuda_bp::decode(decoded_values.data(), buffer.data(), values.size());
    }
}

BENCHMARK(decode)->Range(1ULL<<5, 1ULL<<20);


BENCHMARK_MAIN();
