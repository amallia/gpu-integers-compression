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
#include "bp/utils.hpp"

#include "synthetic.hpp"


class UniformValuesFixture : public ::benchmark::Fixture {

public:
    using ::benchmark::Fixture::SetUp;
    using ::benchmark::Fixture::TearDown;

    virtual void SetUp(::benchmark::State& st) {
        using namespace FastPForLib;
        using namespace gpu_ic;

        IntegerCODEC &codec = *CODECFactory::getFromName("varintgb");

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

public:
    using ::benchmark::Fixture::SetUp;
    using ::benchmark::Fixture::TearDown;

    virtual void SetUp(::benchmark::State& st) {
        using namespace FastPForLib;
        using namespace gpu_ic;
        IntegerCODEC &codec = *CODECFactory::getFromName("varintgb");

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
    IntegerCODEC &codec = *CODECFactory::getFromName("varintgb");

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
    IntegerCODEC &codec = *CODECFactory::getFromName("varintgb");

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

// 2018-12-13 17:22:53
// Running ./bench/varintgb_bench
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
// UniformValuesFixture/decode/32768            36735 ns      36735 ns      19064 bpi=18.0205
// UniformValuesFixture/decode/65536            73480 ns      73483 ns       9525 bpi=17.7593
// UniformValuesFixture/decode/131072          147085 ns     147088 ns       4762 bpi=17.5115
// UniformValuesFixture/decode/262144          294363 ns     294355 ns       2376 bpi=17.0621
// UniformValuesFixture/decode/524288          592717 ns     592738 ns       1181 bpi=16.2365
// UniformValuesFixture/decode/1048576        1223331 ns    1223313 ns        573 bpi=14.8548
// UniformValuesFixture/decode/2097152        2714597 ns    2714589 ns        258 bpi=12.9496
// UniformValuesFixture/decode/4194304        5769699 ns    5769867 ns        121 bpi=11.0831
// UniformValuesFixture/decode/8388608        4805327 ns    4805371 ns        146 bpi=10.1444
// UniformValuesFixture/decode/16777216       7874507 ns    7874597 ns         88 bpi=10.0024
// UniformValuesFixture/decode/33554432      15657883 ns   15657944 ns         45 bpi=10
// UniformValuesFixture/decode/67108864      33114124 ns   33114029 ns         22 bpi=10
// UniformValuesFixture/decode/134217728     63018521 ns   63017599 ns         11 bpi=10
// ClusteredValuesFixture/decode/32768          36689 ns      36690 ns      19361 bpi=17.8379
// ClusteredValuesFixture/decode/65536          72100 ns      72102 ns       9614 bpi=17.165
// ClusteredValuesFixture/decode/131072        144247 ns     144249 ns       4973 bpi=16.676
// ClusteredValuesFixture/decode/262144        275069 ns     275064 ns       2385 bpi=14.6105
// ClusteredValuesFixture/decode/524288        565042 ns     565067 ns       1312 bpi=15.2261
// ClusteredValuesFixture/decode/1048576      1019963 ns    1019982 ns       1012 bpi=12.4913
// ClusteredValuesFixture/decode/2097152      1948842 ns    1948869 ns        459 bpi=11.7296
// ClusteredValuesFixture/decode/4194304      4173419 ns    4173392 ns        157 bpi=11.3336
// ClusteredValuesFixture/decode/8388608      6839816 ns    6839821 ns        100 bpi=10.419
// ClusteredValuesFixture/decode/16777216     8954758 ns    8954437 ns         87 bpi=10.225
// ClusteredValuesFixture/decode/33554432    16006026 ns   16005913 ns         39 bpi=10.0088
// ClusteredValuesFixture/decode/67108864    32279707 ns   32280206 ns         22 bpi=10.0042
// ClusteredValuesFixture/decode/134217728   62708651 ns   62710260 ns         11 bpi=10.0001
//
