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
#include <vector>

#include "bp/cuda_bp.cuh"

#include "synthetic/uniform.hpp"

using namespace ::testing;

TEST(cuda_bp, random) {
    auto n   = 32;
    auto min = 1;
    auto max = 34;

    std::vector<uint32_t> values = synthetic::uniform(n, min, max);
    // std::vector<uint32_t> values = {1, 3, 5, 6};
    for (auto&&v:values)
    {
        // std::cerr<<v<<std::endl;
    }
    std::vector<uint8_t>  buffer(values.size() * 8, 0);

    cuda_bp::encode(buffer.data(), values.data(), values.size());

    std::vector<uint32_t> decoded_values(values.size());
    cuda_bp::decode(decoded_values.data(), buffer.data(), values.size());

    EXPECT_EQ(decoded_values.size(), values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        // std::cerr<<decoded_values[i] << " aa "<<values[i]<<std::endl;
        EXPECT_EQ(decoded_values[i], values[i]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
