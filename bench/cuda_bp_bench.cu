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

#include <cuda.h>

__global__
void warmUpGPU()
{
  // do nothing
}

class RandomValuesFixture : public ::benchmark::Fixture {

    static std::vector<uint32_t> generate_random_vector(size_t n) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::vector<uint32_t> values(n);
        std::uniform_int_distribution<> dis(uint32_t(0));
        std::generate(values.begin(), values.end(), [&](){ return dis(gen); });
        return values;
    }

public:
    using ::benchmark::Fixture::SetUp;
    using ::benchmark::Fixture::TearDown;

    virtual void SetUp(::benchmark::State& st) {
        std::vector<uint32_t> values = generate_random_vector(st.range(0));
        encoded_values.resize(values.size() * 8);
        cuda_bp::encode(encoded_values.data(), values.data(), values.size());
        decoded_values.resize(values.size());
        warmUpGPU<<<1, 1>>>();

    }

    virtual void TearDown(::benchmark::State&) {
        encoded_values.clear();
        decoded_values.clear();
    }

    std::vector<uint8_t>  encoded_values;
    std::vector<uint32_t> decoded_values;
};


BENCHMARK_DEFINE_F(RandomValuesFixture, decode)(benchmark::State& state) {
    while (state.KeepRunning()) {
        cuda_bp::decode(decoded_values.data(), encoded_values.data(), decoded_values.size());
    }
}
BENCHMARK_REGISTER_F(RandomValuesFixture, decode)->Range(1ULL<<14, 1ULL<<28);

// static void decode(benchmark::State &state) {
//     while (state.KeepRunning()) {
//         // state.PauseTiming();
//         // auto n   = state.range(0);
//         // auto min = 1;
//         // auto max = state.range(0)+2;

//         // std::vector<uint32_t> values = generate_random_vector(state.range(0));
//         // std::vector<uint8_t>  buffer(values.size() * 8);
//         // cuda_bp::encode(buffer.data(), values.data(), values.size());
//         // std::vector<uint32_t> decoded_values(values.size());
//         // state.ResumeTiming();
//         // cuda_bp::decode(decoded_values.data(), buffer.data(), values.size());
//     }
// }

// BENCHMARK(decode)->Range(1ULL<<28, 1ULL<<30);


BENCHMARK_MAIN();
