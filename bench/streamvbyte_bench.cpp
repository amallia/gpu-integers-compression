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
#include "benchmark/benchmark.h"
#include "../external/FastPFor/headers/codecfactory.h"
#include "synthetic.hpp"
#include "bp/utils.hpp"


class UniformValuesFixture : public ::benchmark::Fixture {

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
        using namespace FastPForLib;
        using namespace gpu_ic;

        IntegerCODEC &codec = *CODECFactory::getFromName("streamvbyte");

        UniformDataGenerator clu(1);
        values = clu.generate(st.range(0), 1U << 29);
        utils::delta_encode(values.data(), values.size());

        encoded_values.resize(values.size() * 8);
        size_t compressedsize = 0;
        codec.encodeArray(values.data(), values.size(), encoded_values.data(),
                compressedsize);
        encoded_values.resize(compressedsize);
        encoded_values.shrink_to_fit();

        decoded_values.resize(values.size());
    }

    virtual void TearDown(::benchmark::State&) {
        ASSERT_EQ(decoded_values.size(), values.size());
        for (size_t i = 0; i < values.size(); ++i)
        {
            ASSERT_EQ(decoded_values[i], values[i]);
        }
        values.clear();
        encoded_values.clear();
        decoded_values.clear();
    }
    std::vector<uint32_t> values;
    std::vector<uint32_t> encoded_values;
    std::vector<uint32_t> decoded_values;
};

class ClusteredValuesFixture : public ::benchmark::Fixture {

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
        using namespace FastPForLib;
        using namespace gpu_ic;

        IntegerCODEC &codec = *CODECFactory::getFromName("streamvbyte");

        ClusteredDataGenerator clu(1);
        values = clu.generate(st.range(0), 1U << 29);
        utils::delta_encode(values.data(), values.size());

        encoded_values.resize(values.size() * 8);
        size_t compressedsize = 0;
        codec.encodeArray(values.data(), values.size(), encoded_values.data(),
                compressedsize);
        encoded_values.resize(compressedsize);
        encoded_values.shrink_to_fit();

        decoded_values.resize(values.size());
    }

    virtual void TearDown(::benchmark::State&) {
        ASSERT_EQ(decoded_values.size(), values.size());
        for (size_t i = 0; i < values.size(); ++i)
        {
            ASSERT_EQ(decoded_values[i], values[i]);
        }
        values.clear();
        encoded_values.clear();
        decoded_values.clear();
    }
    std::vector<uint32_t> values;
    std::vector<uint32_t> encoded_values;
    std::vector<uint32_t> decoded_values;
};

BENCHMARK_DEFINE_F(UniformValuesFixture, decode)(benchmark::State& state) {
    using namespace FastPForLib;
    IntegerCODEC &codec = *CODECFactory::getFromName("streamvbyte");

    while (state.KeepRunning()) {
          size_t recoveredsize = 0;
          codec.decodeArray(encoded_values.data(), encoded_values.size(),
                    decoded_values.data(), recoveredsize);
    }
    auto bpi = double(32*encoded_values.size())/decoded_values.size();
    state.counters["bpi"] = benchmark::Counter(bpi, benchmark::Counter::kAvgThreads);

}
BENCHMARK_REGISTER_F(UniformValuesFixture, decode)->RangeMultiplier(2)->Range((1ULL << 15), (1ULL<<27));


BENCHMARK_DEFINE_F(ClusteredValuesFixture, decode)(benchmark::State& state) {
    using namespace FastPForLib;
    IntegerCODEC &codec = *CODECFactory::getFromName("streamvbyte");

    while (state.KeepRunning()) {
          size_t recoveredsize = 0;
          codec.decodeArray(encoded_values.data(), encoded_values.size(),
                    decoded_values.data(), recoveredsize);
    }
    auto bpi = double(32*encoded_values.size())/decoded_values.size();
    state.counters["bpi"] = benchmark::Counter(bpi, benchmark::Counter::kAvgThreads);

}
BENCHMARK_REGISTER_F(ClusteredValuesFixture, decode)->RangeMultiplier(2)->Range((1ULL << 15), (1ULL<<27));

BENCHMARK_MAIN();

// 2018-12-13 17:35:15
// Running ./bench/streamvbyte_bench
// Run on (20 X 3600 MHz CPU s)
// CPU Caches:
//   L1 Data 32K (x20)
//   L1 Instruction 32K (x20)
//   L2 Unified 256K (x20)
//   L3 Unified 25600K (x2)
// ***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
// -----------------------------------------------------------------------------------------------
// Benchmark                                        Time           CPU Iterations UserCounters...
// -----------------------------------------------------------------------------------------------
// UniformValuesFixture/decode/32768             7441 ns       7441 ns      78255 bpi=18.0234
// UniformValuesFixture/decode/65536            17256 ns      17257 ns      40340 bpi=17.7554
// UniformValuesFixture/decode/131072           35392 ns      35394 ns      19781 bpi=17.512
// UniformValuesFixture/decode/262144           70221 ns      70224 ns       9981 bpi=17.0621
// UniformValuesFixture/decode/524288          139472 ns     139479 ns       5027 bpi=16.236
// UniformValuesFixture/decode/1048576         272345 ns     272352 ns       2570 bpi=14.8615
// UniformValuesFixture/decode/2097152         525455 ns     525474 ns       1331 bpi=12.9461
// UniformValuesFixture/decode/4194304        1233530 ns    1233558 ns        567 bpi=11.0822
// UniformValuesFixture/decode/8388608        3648455 ns    3648568 ns        191 bpi=10.145
// UniformValuesFixture/decode/16777216       7427241 ns    7427527 ns         94 bpi=10.0025
// UniformValuesFixture/decode/33554432      15115604 ns   15116007 ns         47 bpi=10
// UniformValuesFixture/decode/67108864      30564249 ns   30564557 ns         24 bpi=10
// UniformValuesFixture/decode/134217728     59201439 ns   59202565 ns         12 bpi=10
// ClusteredValuesFixture/decode/32768           6812 ns       6812 ns     101663 bpi=17.7148
// ClusteredValuesFixture/decode/65536          16619 ns      16619 ns      41968 bpi=17.0273
// ClusteredValuesFixture/decode/131072         34465 ns      34467 ns      20519 bpi=16.0706
// ClusteredValuesFixture/decode/262144         69109 ns      69114 ns      10136 bpi=15.9933
// ClusteredValuesFixture/decode/524288        129856 ns     129861 ns       5425 bpi=13.6104
// ClusteredValuesFixture/decode/1048576       262295 ns     262299 ns       2720 bpi=13.0646
// ClusteredValuesFixture/decode/2097152       515444 ns     515461 ns       1415 bpi=12.1334
// ClusteredValuesFixture/decode/4194304      1221010 ns    1221066 ns        577 bpi=11.4315
// ClusteredValuesFixture/decode/8388608      3707675 ns    3707819 ns        190 bpi=10.535
// ClusteredValuesFixture/decode/16777216     7402449 ns    7402553 ns         93 bpi=10.0479
// ClusteredValuesFixture/decode/33554432    14775928 ns   14776484 ns         47 bpi=10.0008
// ClusteredValuesFixture/decode/67108864    29534097 ns   29534518 ns         24 bpi=10.0041
// ClusteredValuesFixture/decode/134217728   58863184 ns   58863812 ns         12 bpi=10.0089
//
