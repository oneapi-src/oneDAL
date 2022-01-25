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

#include <array>

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/metrics/regression.hpp"

#include "oneapi/dal/table/detail/table_builder.hpp"

namespace de = oneapi::dal::detail;

namespace oneapi::dal::test::engine::test {

TEST("Simple accuracy check", "[regression][accuracy]") {
    using Float = double;

    constexpr Float tol(1e-7);

    constexpr std::int64_t row_count = 5;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t element_count = row_count * column_count;

    constexpr std::array<Float, element_count> groundtruth = { 3.5,  -1.0, 7.0,  8.0,   -2.0,
                                                               -8.0, -9.0, 13.0, 128.0, 0.0 };
    const auto gt_table = homogen_table::wrap(groundtruth.data(), row_count, column_count);

    constexpr std::array<Float, element_count> prediction = { 3.7,  -2.0, 0.0,  8.0,   -2.0,
                                                              -8.0, -9.0, 13.0, 128.0, 3.0 };
    const auto pr_table = homogen_table::wrap(prediction.data(), row_count, column_count);

    constexpr std::array<Float, column_count> gt_res = { 9.808, 2.0 };

    const auto res = mse_score(gt_table, pr_table);
    const auto res_ptr = row_accessor<const Float>(res).pull({ 0, -1 });
    for (std::int64_t i = 0; i < column_count; ++i) {
        CAPTURE(i, res_ptr[i], gt_res[i]);
        const auto diff = res_ptr[i] - gt_res[i];
        REQUIRE(-tol <= diff);
        REQUIRE(diff <= tol);
    }
}

} // namespace oneapi::dal::test::engine::test
