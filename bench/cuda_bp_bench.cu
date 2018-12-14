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


#include "synthetic.hpp"
#include "bp/cuda_bp.cuh"
#include "bp/utils/utils.hpp"
#include "bp/utils/cuda_utils.hpp"

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


// 2018-12-13 20:20:04
// Running ./bench/cuda_bp_bench
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
// UniformValuesFixture/decode/32768             4015 ns       4015 ns     174989 bpi=17.4648
// UniformValuesFixture/decode/65536             5913 ns       5912 ns     130802 bpi=16.458
// UniformValuesFixture/decode/131072            9213 ns       9212 ns      83587 bpi=15.4509
// UniformValuesFixture/decode/262144           15922 ns      15921 ns      47925 bpi=14.451
// UniformValuesFixture/decode/524288           28889 ns      28887 ns      25928 bpi=13.4471
// UniformValuesFixture/decode/1048576          52090 ns      52061 ns      10000 bpi=12.4435
// UniformValuesFixture/decode/2097152         101568 ns     101561 ns      10000 bpi=11.4459
// UniformValuesFixture/decode/4194304         200545 ns     200480 ns      10000 bpi=10.4459
// UniformValuesFixture/decode/8388608         398404 ns     398387 ns      10000 bpi=9.44149
// UniformValuesFixture/decode/16777216        795832 ns     795792 ns      10000 bpi=8.43567
// UniformValuesFixture/decode/33554432       1587140 ns    1587061 ns      10000 bpi=7.42388
// UniformValuesFixture/decode/67108864       3168809 ns    3168674 ns      10000 bpi=6.39956
// UniformValuesFixture/decode/134217728      6335958 ns    6335701 ns      10000 bpi=5.34388
// ClusteredValuesFixture/decode/32768           4026 ns       4026 ns     174544 bpi=16.1348
// ClusteredValuesFixture/decode/65536           5906 ns       5905 ns     130792 bpi=14.6621
// ClusteredValuesFixture/decode/131072          9217 ns       9217 ns      83540 bpi=14.0693
// ClusteredValuesFixture/decode/262144         15931 ns      15930 ns      47895 bpi=12.527
// ClusteredValuesFixture/decode/524288         28890 ns      28888 ns      25926 bpi=12.0353
// ClusteredValuesFixture/decode/1048576        52056 ns      52054 ns      10000 bpi=10.7443
// ClusteredValuesFixture/decode/2097152       101566 ns     101560 ns      10000 bpi=9.07336
// ClusteredValuesFixture/decode/4194304       200534 ns     200501 ns      10000 bpi=8.28108
// ClusteredValuesFixture/decode/8388608       398396 ns     398366 ns      10000 bpi=8.16462
// ClusteredValuesFixture/decode/16777216      794791 ns     794760 ns      10000 bpi=6.79579
// ClusteredValuesFixture/decode/33554432     1587420 ns    1587330 ns      10000 bpi=6.14845
// ClusteredValuesFixture/decode/67108864     3170195 ns    3170014 ns      10000 bpi=5.5435
// ClusteredValuesFixture/decode/134217728    6335425 ns    6335043 ns      10000 bpi=4.88561
//