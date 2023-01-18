/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/algo/basic_statistics/backend/cpu/apply_weights.hpp"

namespace oneapi::dal::basic_statistics::backend {

template<typename Cpu, typename Float> 
void apply_weights_single_thread(const pr::ndview<Float, 1>& weights,
                                 pr::ndview<Float, 2>& samples) {
    ONEDAL_ASSERT(weights.has_data());
    ONEDAL_ASSERT(samples.has_mutable_data());

    const auto r_count = samples.get_dimension(0);
    const auto c_count = samples.get_dimension(1);
    ONEDAL_ASSERT(samples.get_count() == r_count);

    const auto* const weights_ptr = weights.get_data();
    auto* const samples_ptr = samples.get_mutable_data();
    const auto samples_str = samples.get_leading_stride();

    for(std::int64_t r = 0; r < r_count; ++r) {
        const auto weight = weights_ptr[r];
        auto* const row = samples_ptr + r * samples_str;

        for(std::int64_t c = 0; c < c_count; ++c) {
            row[c] *= weight;
        }
    }
}

} // namespace oneapi::dal::basic_statistics::backend
