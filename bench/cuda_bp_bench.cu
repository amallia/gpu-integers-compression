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
#include "gpu_ic/cuda_bp.cuh"
 #include "gpu_ic/cuda_delta.cuh"
#include "gpu_ic/utils/utils.hpp"
#include "gpu_ic/utils/cuda_utils.hpp"

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
BENCHMARK_REGISTER_F(UniformValuesFixture, decode)->RangeMultiplier(2)->Range((1ULL << 15), (1ULL<<25));

BENCHMARK_DEFINE_F(UniformValuesFixture, decodeDelta)(benchmark::State& state) {
    while (state.KeepRunning()) {
        cuda_bp::decode(d_decoded, d_encoded, decoded_values.size());
        cuda_delta::decode(d_decoded, decoded_values.size());
    }
    utils::delta_decode(values.data(), values.size());
    auto bpi = double(8*encoded_values.size())/decoded_values.size();
    state.counters["bpi"] = benchmark::Counter(bpi, benchmark::Counter::kAvgThreads);
}
BENCHMARK_REGISTER_F(UniformValuesFixture, decodeDelta)->RangeMultiplier(2)->Range((1ULL << 15), (1ULL<<25));


BENCHMARK_DEFINE_F(ClusteredValuesFixture, decode)(benchmark::State& state) {
    while (state.KeepRunning()) {
        cuda_bp::decode(d_decoded, d_encoded, decoded_values.size());
    }
    auto bpi = double(8*encoded_values.size())/decoded_values.size();
    state.counters["bpi"] = benchmark::Counter(bpi, benchmark::Counter::kAvgThreads);
}
BENCHMARK_REGISTER_F(ClusteredValuesFixture, decode)->RangeMultiplier(2)->Range((1ULL << 15), (1ULL<<25));


BENCHMARK_DEFINE_F(ClusteredValuesFixture, decodeDelta)(benchmark::State& state) {
    while (state.KeepRunning()) {
        cuda_bp::decode(d_decoded, d_encoded, decoded_values.size());
        cuda_delta::decode(d_decoded, decoded_values.size());
    }
    utils::delta_decode(values.data(), values.size());
    auto bpi = double(8*encoded_values.size())/decoded_values.size();
    state.counters["bpi"] = benchmark::Counter(bpi, benchmark::Counter::kAvgThreads);
}
BENCHMARK_REGISTER_F(ClusteredValuesFixture, decodeDelta)->RangeMultiplier(2)->Range((1ULL << 15), (1ULL<<25));

BENCHMARK_MAIN();


// 2018-12-14 14:40:47
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
// UniformValuesFixture/decode/32768             4006 ns       4006 ns     167949 bpi=17.4648
// UniformValuesFixture/decode/65536             5899 ns       5898 ns     131200 bpi=16.458
// UniformValuesFixture/decode/131072            9222 ns       9222 ns      83495 bpi=15.4509
// UniformValuesFixture/decode/262144           15920 ns      15920 ns      47915 bpi=14.451
// UniformValuesFixture/decode/524288           28883 ns      28882 ns      25930 bpi=13.4471
// UniformValuesFixture/decode/1048576          53991 ns      53988 ns      10000 bpi=12.4435
// UniformValuesFixture/decode/2097152         101572 ns     101567 ns      10000 bpi=11.4459
// UniformValuesFixture/decode/4194304         200566 ns     200556 ns      10000 bpi=10.4459
// UniformValuesFixture/decode/8388608         399358 ns     399308 ns      10000 bpi=9.44149
// UniformValuesFixture/decode/16777216        795392 ns     795292 ns      10000 bpi=8.43567
// UniformValuesFixture/decode/33554432       1586824 ns    1586774 ns      10000 bpi=7.42388
// UniformValuesFixture/decode/67108864       3170007 ns    3169839 ns      10000 bpi=6.39956
// UniformValuesFixture/decode/134217728      6334854 ns    6334586 ns      10000 bpi=5.34388
// ClusteredValuesFixture/decode/32768           4013 ns       4013 ns     174890 bpi=16.1348
// ClusteredValuesFixture/decode/65536           5898 ns       5897 ns     131204 bpi=14.6621
// ClusteredValuesFixture/decode/131072          9223 ns       9223 ns      83479 bpi=14.0693
// ClusteredValuesFixture/decode/262144         15918 ns      15918 ns      47934 bpi=12.527
// ClusteredValuesFixture/decode/524288         28881 ns      28880 ns      25932 bpi=12.0353
// ClusteredValuesFixture/decode/1048576        52046 ns      52044 ns      10000 bpi=10.7443
// ClusteredValuesFixture/decode/2097152       101541 ns     101536 ns      10000 bpi=9.07336
// ClusteredValuesFixture/decode/4194304       200581 ns     200572 ns      10000 bpi=8.28108
// ClusteredValuesFixture/decode/8388608       398397 ns     398383 ns      10000 bpi=8.16462
// ClusteredValuesFixture/decode/16777216      795931 ns     795909 ns      10000 bpi=6.79579
// ClusteredValuesFixture/decode/33554432     1586670 ns    1586586 ns      10000 bpi=6.14845
// ClusteredValuesFixture/decode/67108864     3169116 ns    3169005 ns      10000 bpi=5.5435
// ClusteredValuesFixture/decode/134217728    6335305 ns    6335035 ns      10000 bpi=4.88561
//