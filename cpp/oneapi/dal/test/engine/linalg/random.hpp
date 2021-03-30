/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <random>
#include "oneapi/dal/test/engine/linalg/loops.hpp"

namespace oneapi::dal::test::engine::linalg {

template <typename Float>
matrix<Float> generate_uniform(const shape& s, Float a, Float b, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<Float> uniform(a, b);

    auto m = matrix<Float>::empty(s);
    enumerate_linear_mutable(m, [&](std::int64_t i, Float& x) {
        x = uniform(rng);
    });

    return m;
}

} // namespace oneapi::dal::test::engine::linalg
