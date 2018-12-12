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

public:
    using ::benchmark::Fixture::SetUp;
    using ::benchmark::Fixture::TearDown;

    virtual void SetUp(::benchmark::State& st) {
        using namespace FastPForLib;
        using namespace gpu_ic;

        IntegerCODEC &codec = *CODECFactory::getFromName("BP32");
        UniformDataGenerator clu;
        values = clu.generate(st.range(0), 1U << 29);
        utils::delta_encode(values.data(), values.size());

        // values = generate_random_vector(st.range(0));
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
    std::vector<uint32_t>  encoded_values;
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

        IntegerCODEC &codec = *CODECFactory::getFromName("BP32");
        ClusteredDataGenerator clu;
        values = clu.generate(st.range(0), 1U << 29);
        utils::delta_encode(values.data(), values.size());

        // values = generate_random_vector(st.range(0));
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
    std::vector<uint32_t>  encoded_values;
    std::vector<uint32_t> decoded_values;
};

BENCHMARK_DEFINE_F(UniformValuesFixture, decode)(benchmark::State& state) {
    using namespace FastPForLib;
    IntegerCODEC &codec = *CODECFactory::getFromName("BP32");

    while (state.KeepRunning()) {
          size_t recoveredsize = 0;
          codec.decodeArray(encoded_values.data(), encoded_values.size(),
                    decoded_values.data(), recoveredsize);
    }
    auto bpi = double(32*encoded_values.size())/decoded_values.size();
    state.counters["bpi"] = benchmark::Counter(bpi, benchmark::Counter::kAvgThreads);
}
BENCHMARK_REGISTER_F(UniformValuesFixture, decode)->RangeMultiplier(2)->Range((1ULL << 15), (1ULL<<25));

BENCHMARK_DEFINE_F(ClusteredValuesFixture, decode)(benchmark::State& state) {
    using namespace FastPForLib;
    IntegerCODEC &codec = *CODECFactory::getFromName("BP32");

    while (state.KeepRunning()) {
          size_t recoveredsize = 0;
          codec.decodeArray(encoded_values.data(), encoded_values.size(),
                    decoded_values.data(), recoveredsize);
    }
    auto bpi = double(32*encoded_values.size())/decoded_values.size();
    state.counters["bpi"] = benchmark::Counter(bpi, benchmark::Counter::kAvgThreads);
}
BENCHMARK_REGISTER_F(ClusteredValuesFixture, decode)->RangeMultiplier(2)->Range((1ULL << 15), (1ULL<<25));

BENCHMARK_MAIN();


// 2018-12-06 17:23:26
// Running ./bench/bp_bench
// Run on (20 X 3300 MHz CPU s)
// CPU Caches:
//   L1 Data 32K (x20)
//   L1 Instruction 32K (x20)
//   L2 Unified 256K (x20)
//   L3 Unified 25600K (x2)
// ***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
// -------------------------------------------------------------------------------------------
// Benchmark                                    Time           CPU Iterations UserCounters...
// -------------------------------------------------------------------------------------------
// UniformValuesFixture/decode/32768         16877 ns      16830 ns      41172 bpi=16.7168
// UniformValuesFixture/decode/65536         35882 ns      35786 ns      19492 bpi=15.6792
// UniformValuesFixture/decode/131072        87954 ns      87741 ns       7994 bpi=14.6921
// UniformValuesFixture/decode/262144       175259 ns     174845 ns       4021 bpi=13.6907
// UniformValuesFixture/decode/524288       341369 ns     340466 ns       2042 bpi=12.6934
// UniformValuesFixture/decode/1048576      667551 ns     666325 ns       1044 bpi=11.6936
// UniformValuesFixture/decode/2097152     1353950 ns    1352086 ns        517 bpi=10.6969
// UniformValuesFixture/decode/4194304     3030158 ns    3021661 ns        240 bpi=9.69523
// UniformValuesFixture/decode/8388608     6849383 ns    6832816 ns        102 bpi=8.69191
// UniformValuesFixture/decode/16777216   13706952 ns   13541625 ns         52 bpi=7.68595
// UniformValuesFixture/decode/33554432   27643785 ns   27562744 ns         26 bpi=6.67392
// ClusteredValuesFixture/decode/32768         15636 ns      15521 ns      45112 bpi=16.5576
// ClusteredValuesFixture/decode/65536         42432 ns      41821 ns      16692 bpi=14.2466
// ClusteredValuesFixture/decode/131072        79689 ns      78986 ns       8815 bpi=13.696
// ClusteredValuesFixture/decode/262144       173244 ns     171493 ns       4077 bpi=11.8008
// ClusteredValuesFixture/decode/524288       339450 ns     337114 ns       2069 bpi=9.84265
// ClusteredValuesFixture/decode/1048576      666677 ns     658358 ns       1087 bpi=10.7903
// ClusteredValuesFixture/decode/2097152     1272503 ns    1258142 ns        539 bpi=9.1908
// ClusteredValuesFixture/decode/4194304     3007479 ns    2962997 ns        241 bpi=7.92332
// ClusteredValuesFixture/decode/8388608     5882221 ns    5792226 ns        113 bpi=8.57848
// ClusteredValuesFixture/decode/16777216   13562463 ns   13389704 ns         52 bpi=5.73607
// ClusteredValuesFixture/decode/33554432   26640289 ns   26560698 ns         27 bpi=5.22692