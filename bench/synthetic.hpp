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

#pragma once

#include <random>
#include <set>
#include <vector>

namespace gpu_ic {

class UniformDataGenerator {
   public:
    UniformDataGenerator(uint32_t seed = std::random_device{}()) : rand(seed) {}

    std::vector<uint32_t> generate(uint32_t N, uint32_t Max) {
        if (Max < N)
            throw std::runtime_error("can't generate enough distinct elements in small interval");

        std::uniform_int_distribution<uint32_t> dis(1, Max - 1);
        std::vector<uint32_t>                   ans;
        if (N == 0)
            return ans; // nothing to do
        ans.reserve(N);
        assert(Max >= 1);

        if (2 * N > Max) {
            std::set<uint32_t> s;
            while (s.size() < Max - N)
                s.insert(dis(rand));
            s.insert(Max);
            ans.resize(N);
            uint32_t i = 0;
            size_t   c = 0;
            for (uint32_t v : s) {
                for (; i < v; ++i)
                    ans[c++] = i;
                ++i;
            }
            assert(c == ans.size());
        } else {
            std::set<uint32_t> s;
            while (s.size() < N)
                s.insert(dis(rand));
            ans.assign(s.begin(), s.end());
            assert(N == ans.size());
        }
        return ans;
    }
    std::mt19937 rand;
};

class ClusteredDataGenerator {
   public:
    UniformDataGenerator unidg;
    ClusteredDataGenerator(uint32_t seed = std::random_device{}()) : unidg(seed) {}

    template <class iterator>
    void fillUniform(iterator begin, iterator end, uint32_t Min, uint32_t Max) {
        std::vector<uint32_t> v = unidg.generate(static_cast<uint32_t>(end - begin), Max - Min);
        for (size_t k = 0; k < v.size(); ++k)
            *(begin + k) = Min + v[k];
    }
    template <class iterator>
    void fillClustered(iterator begin, iterator end, uint32_t Min, uint32_t Max) {
        const uint32_t N     = static_cast<uint32_t>(end - begin);
        const uint32_t range = Max - Min;
        if (range < N)
            throw std::runtime_error("can't generate that many in small interval.");
        assert(range >= N);
        if ((range == N) || (N < 10)) {
            fillUniform(begin, end, Min, Max);
            return;
        }
        std::uniform_int_distribution<uint32_t> dis(1, range - N);
        const uint32_t                          cut = N / 2 + dis(unidg.rand);
        assert(cut >= N / 2);
        assert(Max - Min - cut >= N - N / 2);

        std::uniform_real_distribution<double> urd_dis;
        const double                           p = urd_dis(unidg.rand);
        assert(p <= 1);
        assert(p >= 0);
        if (p <= 0.25) {
            fillUniform(begin, begin + N / 2, Min, Min + cut);
            fillClustered(begin + N / 2, end, Min + cut, Max);
        } else if (p <= 0.5) {
            fillClustered(begin, begin + N / 2, Min, Min + cut);
            fillUniform(begin + N / 2, end, Min + cut, Max);
        } else {
            fillClustered(begin, begin + N / 2, Min, Min + cut);
            fillClustered(begin + N / 2, end, Min + cut, Max);
        }
    }

    std::vector<uint32_t> generate(uint32_t N, uint32_t Max) {
        std::vector<uint32_t> ans(N);
        fillClustered(ans.begin(), ans.end(), 0, Max);
        return ans;
    }
};

} // namespace gpu_ic
