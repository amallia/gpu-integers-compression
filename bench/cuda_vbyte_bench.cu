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


__global__
void warmUpGPU()
{
  // do nothing
}

__host__ __device__ void printBinary(unsigned myNumber) {
    int numberOfBits = sizeof(unsigned) * 8;
    for (int i = numberOfBits - 1; i >= 0; i--) {
        bool isBitSet = (myNumber & (1U << i));
        if (isBitSet) {
            printf("1");
        } else {
            printf("0");
        }
    }
    printf("\n");
}

class UniformValuesFixture : public ::benchmark::Fixture {

public:
    using ::benchmark::Fixture::SetUp;
    using ::benchmark::Fixture::TearDown;

    virtual void SetUp(::benchmark::State& st) {
        using namespace gpu_ic;

        UniformDataGenerator clu(1);
        values = clu.generate(st.range(0), 1U << 29);
        utils::delta_encode(values.data(), values.size());

        encoded_values.resize(values.size() * 8);
        auto compressedsize = cuda_vbyte::encode(encoded_values.data(), values.data(), values.size());
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


class ClusteredValuesFixture : public ::benchmark::Fixture {

public:
    using ::benchmark::Fixture::SetUp;
    using ::benchmark::Fixture::TearDown;

    virtual void SetUp(::benchmark::State& st) {
        using namespace gpu_ic;

        ClusteredDataGenerator clu(1);
        values = clu.generate(st.range(0), 1U << 29);
        utils::delta_encode(values.data(), values.size());

        encoded_values.resize(values.size() * 8);
        auto compressedsize = cuda_vbyte::encode(encoded_values.data(), values.data(), values.size());
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



BENCHMARK_DEFINE_F(UniformValuesFixture, decode)(benchmark::State& state) {
    while (state.KeepRunning()) {
        cuda_vbyte::decode(d_decoded, d_encoded, decoded_values.size());
    }
    auto bpi = double(8*encoded_values.size())/decoded_values.size();
    state.counters["bpi"] = benchmark::Counter(bpi, benchmark::Counter::kAvgThreads);
}
BENCHMARK_REGISTER_F(UniformValuesFixture, decode)->RangeMultiplier(2)->Range((1ULL << 15), (1ULL<<25));


BENCHMARK_DEFINE_F(ClusteredValuesFixture, decode)(benchmark::State& state) {
    while (state.KeepRunning()) {
        cuda_vbyte::decode(d_decoded, d_encoded, decoded_values.size());
    }
    auto bpi = double(8*encoded_values.size())/decoded_values.size();
    state.counters["bpi"] = benchmark::Counter(bpi, benchmark::Counter::kAvgThreads);
}
BENCHMARK_REGISTER_F(ClusteredValuesFixture, decode)->RangeMultiplier(2)->Range((1ULL << 15), (1ULL<<25));

BENCHMARK_MAIN();


// 2018-12-14 16:00:30
// Running ./bench/cuda_vbyte_bench
// Run on (28 X 3500 MHz CPU s)
// CPU Caches:
//   L1 Data 32K (x28)
//   L1 Instruction 32K (x28)
//   L2 Unified 256K (x28)
//   L3 Unified 35840K (x2)
// ***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
// -----------------------------------------------------------------------------------------------
// Benchmark                                        Time           CPU Iterations UserCounters...
// -----------------------------------------------------------------------------------------------
// UniformValuesFixture/decode/32768             4392 ns       4391 ns     160119 bpi=19.7529
// UniformValuesFixture/decode/65536             6084 ns       6083 ns     127017 bpi=19.3794
// UniformValuesFixture/decode/131072            9469 ns       9469 ns      81282 bpi=19.0952
// UniformValuesFixture/decode/262144           16409 ns      16409 ns      46457 bpi=18.6882
// UniformValuesFixture/decode/524288           29385 ns      29383 ns      25466 bpi=17.8624
// UniformValuesFixture/decode/1048576          53848 ns      53845 ns      10000 bpi=16.4886
// UniformValuesFixture/decode/2097152         103242 ns     103237 ns      10000 bpi=14.5715
// UniformValuesFixture/decode/4194304         203382 ns     203372 ns      10000 bpi=12.7031
// UniformValuesFixture/decode/8388608         405533 ns     405517 ns      10000 bpi=12.0025
// UniformValuesFixture/decode/16777216        807971 ns     807945 ns      10000 bpi=12
// UniformValuesFixture/decode/33554432       1608733 ns    1608607 ns      10000 bpi=12
// UniformValuesFixture/decode/67108864       3216589 ns    3216433 ns      10000 bpi=12
// UniformValuesFixture/decode/134217728      6427645 ns    6427269 ns      10000 bpi=12
// ClusteredValuesFixture/decode/32768           4388 ns       4388 ns     160204 bpi=19.0527
// ClusteredValuesFixture/decode/65536           6080 ns       6079 ns     127209 bpi=17.957
// ClusteredValuesFixture/decode/131072          9463 ns       9463 ns      81341 bpi=17.623
// ClusteredValuesFixture/decode/262144         16421 ns      16409 ns      46430 bpi=16.4943
// ClusteredValuesFixture/decode/524288         29384 ns      29383 ns      25469 bpi=16.1244
// ClusteredValuesFixture/decode/1048576        53519 ns      53517 ns      10000 bpi=14.3243
// ClusteredValuesFixture/decode/2097152       103623 ns     103618 ns      10000 bpi=13.4584
// ClusteredValuesFixture/decode/4194304       203156 ns     203146 ns      10000 bpi=12.7371
// ClusteredValuesFixture/decode/8388608       403868 ns     403852 ns      10000 bpi=12.2113
// ClusteredValuesFixture/decode/16777216      807920 ns     807892 ns      10000 bpi=12.0574
// ClusteredValuesFixture/decode/33554432     1608718 ns    1608661 ns      10000 bpi=12.006
// ClusteredValuesFixture/decode/67108864     3216494 ns    3216368 ns      10000 bpi=12
// ClusteredValuesFixture/decode/134217728    6427839 ns    6427591 ns      10000 bpi=12
//