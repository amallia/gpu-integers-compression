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

#include <cuda.h>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "benchmark/benchmark.h"

#include "bp/utils/utils.hpp"
#include "bp/cuda_delta.cuh"
#include "bp/utils/cuda_utils.hpp"


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
        values = generate_random_vector(st.range(0));
        std::sort(values.begin(), values.end());
        encoded_values = values;
        utils::delta_encode(encoded_values.data(), encoded_values.size());

        decoded_values.resize(values.size());
        CUDA_CHECK_ERROR(cudaSetDevice(0));
        CUDA_CHECK_ERROR(cudaMalloc((void **)&d_encoded, values.size() * sizeof(uint32_t)));
        CUDA_CHECK_ERROR(cudaMemcpy(d_encoded, encoded_values.data(), values.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

        CUDA_CHECK_ERROR(cudaMalloc((void **)&d_decoded, values.size() * sizeof(uint32_t)));
        warmUpGPU<<<1, 1>>>();
    }

    virtual void TearDown(::benchmark::State&) {
        CUDA_CHECK_ERROR(cudaMemcpy(decoded_values.data(), d_decoded, values.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        ASSERT_EQ(decoded_values.size(), values.size());
        for (size_t i = 0; i < values.size(); ++i)
        {
            ASSERT_EQ(decoded_values[i], values[i]);
        }
        CUDA_CHECK_ERROR(cudaFree(d_encoded));
        CUDA_CHECK_ERROR(cudaFree(d_decoded));
        values.clear();
        encoded_values.clear();
        decoded_values.clear();
    }
    std::vector<uint32_t> values;
    std::vector<uint32_t> encoded_values;
    std::vector<uint32_t> decoded_values;
    uint8_t *  d_encoded;
    uint32_t * d_decoded;

};


BENCHMARK_DEFINE_F(RandomValuesFixture, decode)(benchmark::State& state) {
    while (state.KeepRunning()) {
        cuda_delta::decode(d_decoded, reinterpret_cast<uint8_t*>(d_encoded), decoded_values.size());
    }
    auto bpi = 32;
    state.counters["bpi"] = benchmark::Counter(bpi, benchmark::Counter::kAvgThreads);
}
BENCHMARK_REGISTER_F(RandomValuesFixture, decode)->Range(1ULL<<14, 1ULL<<28);

BENCHMARK_MAIN();
