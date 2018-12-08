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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cuda.h>
#include "bp/utils.hpp"

#include "synthetic/uniform.hpp"
#include "benchmark/benchmark.h"

#include "bp/cuda_vbyte.cuh"
#include "bp/cuda_common.hpp"

__global__
void warmUpGPU()
{
  // do nothing
}

__host__ __device__ void printBinary(unsigned long long myNumber) {
    int numberOfBits = sizeof(unsigned long long) * 8;
    for (int i = numberOfBits - 1; i >= 0; i--) {
        bool isBitSet = (myNumber & (1ULL << i));
        if (isBitSet) {
            printf("1");
        } else {
            printf("0");
        }
    }
    printf("\n");
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
        utils::delta_encode(values.data(), values.size());

        encoded_values.resize(values.size() * 8);
        auto compressedsize = cuda_vbyte::encode(encoded_values.data(), values.data(), values.size());

        bit_istream br(encoded_values.data());

        // std::cerr << br.read(32) << std::endl;
        // auto b = br.read(2);
        // std::cerr << b << std::endl;
        // std::cerr << br.read(2) << std::endl;
        // std::cerr << br.read(2) << std::endl;
        // std::cerr << br.read(2) << std::endl;
        // std::cerr << br.read(2) << std::endl;
        // std::cerr << br.read(2) << std::endl;
        // std::cerr << br.read(2) << std::endl;
        // std::cerr << br.read(2) << std::endl;
        // std::cerr << br.read(2) << std::endl;
        // std::cerr << br.read(2) << std::endl;
        // std::cerr << br.read(2) << std::endl;
        // std::cerr << br.read(2) << std::endl;
        // std::cerr << br.read(2) << std::endl;
        // std::cerr << br.read(2) << std::endl;
        // std::cerr << br.read(2) << std::endl;
        // std::cerr << br.read(2) << std::endl;

        // std::cerr << (b+1)*8 << std::endl;
        // std::cerr << *(reinterpret_cast<uint32_t*>(encoded_values.data())) << std::endl;

        encoded_values.resize(compressedsize);
        encoded_values.shrink_to_fit();

        decoded_values.resize(values.size());
        CUDA_CHECK_ERROR(cudaSetDevice(0));
        warmUpGPU<<<1, 1>>>();
        CUDA_CHECK_ERROR(cudaMalloc((void **)&d_encoded, encoded_values.size() * sizeof(uint8_t)));
        CUDA_CHECK_ERROR(cudaMemcpy(d_encoded, encoded_values.data(), encoded_values.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));

        CUDA_CHECK_ERROR(cudaMalloc((void **)&d_decoded, values.size() * sizeof(uint32_t)));

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


BENCHMARK_DEFINE_F(RandomValuesFixture, decode)(benchmark::State& state) {
    while (state.KeepRunning()) {
        cuda_vbyte::decode(d_decoded, d_encoded, decoded_values.size());
    }
    auto bpi = double(8*encoded_values.size())/decoded_values.size();
    state.counters["bpi"] = benchmark::Counter(bpi, benchmark::Counter::kAvgThreads);
}
BENCHMARK_REGISTER_F(RandomValuesFixture, decode)->Range(1ULL<<4, 1ULL<<4);

BENCHMARK_MAIN();
