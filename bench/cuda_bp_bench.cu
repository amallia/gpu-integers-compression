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

#include "benchmark/benchmark.h"

#include "synthetic.hpp"
#include "bp/cuda_bp.cuh"
#include "bp/cuda_common.hpp"
#include "bp/utils.hpp"

__global__
void warmUpGPU()
{
  // do nothing
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
        auto compressedsize = cuda_bp::encode(encoded_values.data(), values.data(), values.size());
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
        auto compressedsize = cuda_bp::encode(encoded_values.data(), values.data(), values.size());
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
        cuda_bp::decode(d_decoded, d_encoded, decoded_values.size());
    }
    auto bpi = double(8*encoded_values.size())/decoded_values.size();
    state.counters["bpi"] = benchmark::Counter(bpi, benchmark::Counter::kAvgThreads);
}
BENCHMARK_REGISTER_F(UniformValuesFixture, decode)->RangeMultiplier(2)->Range((1ULL << 15), (1ULL<<27));

BENCHMARK_DEFINE_F(ClusteredValuesFixture, decode)(benchmark::State& state) {
    while (state.KeepRunning()) {
        cuda_bp::decode(d_decoded, d_encoded, decoded_values.size());
    }
    auto bpi = double(8*encoded_values.size())/decoded_values.size();
    state.counters["bpi"] = benchmark::Counter(bpi, benchmark::Counter::kAvgThreads);
}
BENCHMARK_REGISTER_F(ClusteredValuesFixture, decode)->RangeMultiplier(2)->Range((1ULL << 15), (1ULL<<27));

BENCHMARK_MAIN();


// 2018-12-13 16:55:50
// Running ./bench/bp_bench
// Run on (20 X 3300 MHz CPU s)
// CPU Caches:
//   L1 Data 32K (x20)
//   L1 Instruction 32K (x20)
//   L2 Unified 256K (x20)
//   L3 Unified 25600K (x2)
// ***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
// -----------------------------------------------------------------------------------------------
// Benchmark                                        Time           CPU Iterations UserCounters...
// -----------------------------------------------------------------------------------------------
// UniformValuesFixture/decode/32768            17409 ns      17385 ns      40108 bpi=16.6943
// UniformValuesFixture/decode/65536            35655 ns      35617 ns      19475 bpi=15.7017
// UniformValuesFixture/decode/131072           87983 ns      87858 ns       8024 bpi=14.7017
// UniformValuesFixture/decode/262144          174795 ns     174617 ns       4014 bpi=13.7122
// UniformValuesFixture/decode/524288          344581 ns     344472 ns       2017 bpi=12.6995
// UniformValuesFixture/decode/1048576         673893 ns     673677 ns       1035 bpi=11.6968
// UniformValuesFixture/decode/2097152        1454940 ns    1454459 ns        487 bpi=10.6989
// UniformValuesFixture/decode/4194304        3159596 ns    3154739 ns        218 bpi=9.69363
// UniformValuesFixture/decode/8388608        6883850 ns    6876522 ns        101 bpi=8.69091
// UniformValuesFixture/decode/16777216      13062984 ns   13047336 ns         53 bpi=7.68572
// UniformValuesFixture/decode/33554432      27467047 ns   27458426 ns         26 bpi=6.67464
// UniformValuesFixture/decode/67108864      53611274 ns   53553763 ns         13 bpi=5.64925
// UniformValuesFixture/decode/134217728     92672544 ns   92576065 ns          8 bpi=4.59395
// ClusteredValuesFixture/decode/32768          18784 ns      18778 ns      35470 bpi=15.3086
// ClusteredValuesFixture/decode/65536          40722 ns      40710 ns      18460 bpi=15.3008
// ClusteredValuesFixture/decode/131072         74047 ns      74022 ns       8457 bpi=12.5525
// ClusteredValuesFixture/decode/262144        167060 ns     166894 ns       4217 bpi=12.9794
// ClusteredValuesFixture/decode/524288        322968 ns     322624 ns       2215 bpi=10.9802
// ClusteredValuesFixture/decode/1048576       665217 ns     664979 ns       1022 bpi=10.1548
// ClusteredValuesFixture/decode/2097152      1243538 ns    1241797 ns        549 bpi=8.51848
// ClusteredValuesFixture/decode/4194304      2919127 ns    2911155 ns        244 bpi=7.27275
// ClusteredValuesFixture/decode/8388608      6502194 ns    6494839 ns        107 bpi=7.16933
// ClusteredValuesFixture/decode/16777216    12394977 ns   12377570 ns         58 bpi=6.55128
// ClusteredValuesFixture/decode/33554432    26618851 ns   26582725 ns         27 bpi=6.07258
// ClusteredValuesFixture/decode/67108864    48767993 ns   48714277 ns         14 bpi=4.03127
// ClusteredValuesFixture/decode/134217728   98318502 ns   98210648 ns          7 bpi=4.08377
//
