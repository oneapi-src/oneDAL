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

#include "oneapi/dal/test/engine/common.hpp"

#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::test::engine {

template <typename Float = double>
inline auto mse_score(const table& groundtruth, const table& prediction) {
    INFO("check if response shape is expected to be equal")
    REQUIRE(prediction.get_row_count() == groundtruth.get_row_count());
    REQUIRE(prediction.get_column_count() == groundtruth.get_column_count());

    const auto n_responses = groundtruth.get_column_count();
    const auto n_samples = groundtruth.get_row_count();

    auto result = array<Float>::zeros(n_responses);
    auto* res_ptr = result.get_mutable_data();
    for (std::int64_t j = 0; j < n_samples; ++j) {
        const auto gt_row = row_accessor<const Float>(groundtruth).pull({ j, j + 1 });
        const auto pr_row = row_accessor<const Float>(prediction).pull({ j, j + 1 });
        for (std::int64_t i = 0; i < n_responses; ++i) {
            const auto diff = gt_row[i] - pr_row[i];
            res_ptr[i] += Float(diff * diff);
        }
    }
    for (std::int64_t i = 0; i < n_responses; ++i) {
        res_ptr[i] /= Float(n_samples);
    }
    using oneapi::dal::detail::homogen_table_builder;
    return homogen_table_builder{}.reset(result, 1, n_responses).build();
}

} // namespace oneapi::dal::test::engine
