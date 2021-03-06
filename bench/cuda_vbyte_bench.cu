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

#include "gpu_ic/cuda_vbyte.cuh"
#include "gpu_ic/utils/utils.hpp"
#include "gpu_ic/utils/cuda_utils.hpp"

#include "synthetic.hpp"

template <typename Generator, size_t block_size>
class ValuesFixture : public ::benchmark::Fixture {

public:
    using ::benchmark::Fixture::SetUp;
    using ::benchmark::Fixture::TearDown;

    virtual void SetUp(::benchmark::State& st) {
        using namespace gpu_ic;

        Generator clu(1);
        values = clu.generate(st.range(0), 1U << 29);
        utils::delta_encode(values.data(), values.size());

        encoded_values.resize(values.size() * 8);
        auto compressedsize = cuda_vbyte::encode<block_size>(encoded_values.data(), values.data(), values.size());
        encoded_values.resize(compressedsize);
        encoded_values.shrink_to_fit();

        decoded_values.resize(values.size());
        CUDA_CHECK_ERROR(cudaSetDevice(0));
        warmUpGPU<<<1, 1>>>();
        CUDA_CHECK_ERROR(cudaMalloc((void **)&d_encoded, encoded_values.size() * sizeof(uint8_t)));
        CUDA_CHECK_ERROR(cudaMemcpy(d_encoded, encoded_values.data(), encoded_values.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));

        CUDA_CHECK_ERROR(cudaMalloc((void **)&d_decoded, values.size() * sizeof(uint32_t)));
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }

    virtual void TearDown(::benchmark::State&) {
        CUDA_CHECK_ERROR(cudaMemcpy(decoded_values.data(), d_decoded, values.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        ASSERT_EQ(decoded_values.size(), values.size());
        for (size_t i = 0; i < values.size(); ++i)
        {
            ASSERT_EQ(decoded_values[i], values[i]);
        }

        cudaFree(d_encoded);
        cudaFree(d_decoded);
        values.clear();
        encoded_values.clear();
        decoded_values.clear();
    }
    std::vector<uint32_t> values;
    std::vector<uint8_t> encoded_values;
    std::vector<uint32_t> decoded_values;
    uint8_t *  d_encoded;
    uint32_t * d_decoded;
};

BENCHMARK_TEMPLATE_DEFINE_F(ValuesFixture, decodeUniform128, gpu_ic::UniformDataGenerator, 128)(benchmark::State& state) {
    while (state.KeepRunning()) {
        cuda_vbyte::decode<128>(d_decoded, d_encoded, decoded_values.size());
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }
    auto bpi = double(8*encoded_values.size())/decoded_values.size();
    state.counters["bpi"] = benchmark::Counter(bpi, benchmark::Counter::kAvgThreads);
}
BENCHMARK_REGISTER_F(ValuesFixture, decodeUniform128)->RangeMultiplier(2)->Range((1ULL << 15), (1ULL<<25));

BENCHMARK_TEMPLATE_DEFINE_F(ValuesFixture, decodeUniform1024, gpu_ic::UniformDataGenerator, 1024)(benchmark::State& state) {
    while (state.KeepRunning()) {
        cuda_vbyte::decode<1024>(d_decoded, d_encoded, decoded_values.size());
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }
    auto bpi = double(8*encoded_values.size())/decoded_values.size();
    state.counters["bpi"] = benchmark::Counter(bpi, benchmark::Counter::kAvgThreads);
}
BENCHMARK_REGISTER_F(ValuesFixture, decodeUniform1024)->RangeMultiplier(2)->Range((1ULL << 15), (1ULL<<25));

BENCHMARK_TEMPLATE_DEFINE_F(ValuesFixture, decodeClustered128, gpu_ic::ClusteredDataGenerator, 128)(benchmark::State& state) {
    while (state.KeepRunning()) {
        cuda_vbyte::decode<128>(d_decoded, d_encoded, decoded_values.size());
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }
    auto bpi = double(8*encoded_values.size())/decoded_values.size();
    state.counters["bpi"] = benchmark::Counter(bpi, benchmark::Counter::kAvgThreads);
}
BENCHMARK_REGISTER_F(ValuesFixture, decodeClustered128)->RangeMultiplier(2)->Range((1ULL << 15), (1ULL<<25));

BENCHMARK_TEMPLATE_DEFINE_F(ValuesFixture, decodeClustered1024, gpu_ic::ClusteredDataGenerator, 1024)(benchmark::State& state) {
    while (state.KeepRunning()) {
        cuda_vbyte::decode<1024>(d_decoded, d_encoded, decoded_values.size());
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }
    auto bpi = double(8*encoded_values.size())/decoded_values.size();
    state.counters["bpi"] = benchmark::Counter(bpi, benchmark::Counter::kAvgThreads);
}
BENCHMARK_REGISTER_F(ValuesFixture, decodeClustered1024)->RangeMultiplier(2)->Range((1ULL << 15), (1ULL<<25));

BENCHMARK_MAIN();
