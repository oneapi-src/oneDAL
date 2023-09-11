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
#include <cmath>

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/metrics/clustering.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::test::engine::test {

TEST("Simple accuracy check", "[clustering][accuracy]") {
    using Float = double;

    const Float tol(1.e-5);

    constexpr std::int64_t cluster_count = 3;
    constexpr std::int64_t row_count = 5;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t element_count = row_count * column_count;

    constexpr std::array<Float, element_count> centroids = { -2.f, -2.f, 0.f, 0.f, 1.f, 1.f };
    const auto centroids_table = homogen_table::wrap(centroids.data(), cluster_count, column_count);

    constexpr std::array<Float, element_count> data = { -3.f, -3.f, -4.f, -4.f, 0.f,
                                                        0.f,  1.5f, 1.5f, 5.f,  5.f };
    const auto data_table = homogen_table::wrap(data.data(), row_count, column_count);

    constexpr std::array<Float, element_count> assignments = { 0.f, 0.f, 1.f, 2.f, 2.f };
    const auto assignments_table = homogen_table::wrap(assignments.data(), row_count, 1);

    const Float expected = (1.25f + 2.25f + 2.25f) / 3.f;

    const auto res = davies_bouldin_index(data_table, centroids_table, assignments_table);
    const auto diff = expected - res;
    CAPTURE(expected, res);
    if (res == 0.0f || expected == 0.f) {
        REQUIRE(std::fabs(diff) <= tol);
    }
    else {
        REQUIRE(std::fabs(diff) / (std::max(std::fabs(expected), std::fabs(res))) <= tol);
    }
}

} // namespace oneapi::dal::test::engine::test
