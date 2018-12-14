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

        IntegerCODEC &codec = *CODECFactory::getFromName("simdbinarypacking");

        UniformDataGenerator clu(1);
        auto tmp = clu.generate(st.range(0), 1U << 29);
        values = std::vector<uint32_t>(tmp.begin(), tmp.end());
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

        IntegerCODEC &codec = *CODECFactory::getFromName("simdbinarypacking");

        ClusteredDataGenerator clu(1);
        auto tmp = clu.generate(st.range(0), 1U << 29);
        values = std::vector<uint32_t>(tmp.begin(), tmp.end());
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
    IntegerCODEC &codec = *CODECFactory::getFromName("simdbinarypacking");

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
    IntegerCODEC &codec = *CODECFactory::getFromName("simdbinarypacking");

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

// 2018-12-13 17:10:19
// Running ./bench/simdbp_bench
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
// UniformValuesFixture/decode/32768             5726 ns       5725 ns     123239 bpi=17.0117
// UniformValuesFixture/decode/65536            15691 ns      15687 ns      45080 bpi=16.0332
// UniformValuesFixture/decode/131072           34484 ns      34470 ns      20280 bpi=15.002
// UniformValuesFixture/decode/262144           68037 ns      68034 ns      10276 bpi=13.999
// UniformValuesFixture/decode/524288          136875 ns     136851 ns       5109 bpi=13.0107
// UniformValuesFixture/decode/1048576         278092 ns     278072 ns       2548 bpi=12.0148
// UniformValuesFixture/decode/2097152         846230 ns     846154 ns        716 bpi=11.0112
// UniformValuesFixture/decode/4194304        2490323 ns    2490096 ns        287 bpi=10.0119
// UniformValuesFixture/decode/8388608        5115960 ns    5115616 ns        137 bpi=9.00362
// UniformValuesFixture/decode/16777216       9607209 ns    9606369 ns         74 bpi=7.99929
// UniformValuesFixture/decode/33554432      17166027 ns   17164597 ns         38 bpi=6.98763
// UniformValuesFixture/decode/67108864      34623921 ns   34620815 ns         20 bpi=5.96243
// UniformValuesFixture/decode/134217728     68372372 ns   68366372 ns         10 bpi=4.90066
// ClusteredValuesFixture/decode/32768           5852 ns       5852 ns     122222 bpi=14.5039
// ClusteredValuesFixture/decode/65536          15019 ns      15019 ns      46574 bpi=14.166
// ClusteredValuesFixture/decode/131072         34434 ns      34431 ns      20140 bpi=14.1201
// ClusteredValuesFixture/decode/262144         67853 ns      67849 ns      10288 bpi=12.4434
// ClusteredValuesFixture/decode/524288        134160 ns     134153 ns       5135 bpi=10.9097
// ClusteredValuesFixture/decode/1048576       267771 ns     267754 ns       2606 bpi=10.1447
// ClusteredValuesFixture/decode/2097152       533289 ns     533254 ns       1312 bpi=7.52313
// ClusteredValuesFixture/decode/4194304      1690085 ns    1689953 ns        419 bpi=9.18613
// ClusteredValuesFixture/decode/8388608      4139615 ns    4139418 ns        171 bpi=7.47873
// ClusteredValuesFixture/decode/16777216     8160616 ns    8160203 ns         87 bpi=7.72549
// ClusteredValuesFixture/decode/33554432    16158062 ns   16156173 ns         43 bpi=5.9601
// ClusteredValuesFixture/decode/67108864    32125292 ns   32122824 ns         22 bpi=5.05499
// ClusteredValuesFixture/decode/134217728   64964340 ns   64959360 ns         11 bpi=4.07462
//
