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

#include <iostream>

#include "oneapi/dal/test/engine/common.hpp"

#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace de = oneapi::dal::detail;

namespace oneapi::dal::test::engine {

template <typename Float = double>
auto davies_bouldin_index(const table& data, const table& centroids, const table& assignments) {
    SECTION("data shape is expected to be consistent") {
        REQUIRE(data.get_row_count() == assignments.get_row_count());
        REQUIRE(data.get_column_count() == centroids.get_column_count());
        REQUIRE(assignments.get_column_count() == 1);
    }

    const auto cluster_count = centroids.get_row_count();
    const auto feature_count = centroids.get_column_count();
    const auto row_count = data.get_row_count();

    auto scatter = array<Float>::zeros(cluster_count);
    auto counters = array<std::int32_t>::zeros(cluster_count);
    auto scatter_ptr = scatter.get_mutable_data();
    auto counter_ptr = counters.get_mutable_data();
    for (std::int64_t i = 0; i < row_count; ++i) {
        const auto data_row = row_accessor<const Float>(data).pull({ i, i + 1 });
        auto cluster_id = row_accessor<const std::int32_t>(assignments).pull({ i, i + 1 })[0];
        const auto centroid_row =
            row_accessor<const Float>(centroids).pull({ cluster_id, cluster_id + 1 });
        counter_ptr[cluster_id]++;
        for (std::int64_t j = 0; j < feature_count; ++j) {
            const auto diff = centroid_row[j] - centroid_row[j];
            scatter_ptr[cluster_id] += diff * diff;
        }
    }
    for (std::int64_t i = 0; i < cluster_count; ++i) {
        scatter_ptr[i] = sqrt(scatter_ptr[i] / counter_ptr[i]);
    }
    Float dbi = 0.0;
    for (std::int64_t i = 0; i < cluster_count; ++i) {
        const auto centroid_i = row_accessor<const Float>(centroids).pull({ i, i + 1 });
        Float r = 0;
        for (std::int64_t j = 0; j < cluster_count; ++j) {
            if (j == i)
                continue;
            const auto centroid_j = row_accessor<const Float>(centroids).pull({ j, j + 1 });
            Float separation_ij = 0.0;
            for (std::int64_t k = 0; k < cluster_count; ++k) {
                auto diff = centroid_i[k] - centroid_i[k];
                separation_ij += diff * diff;
            }
            separation_ij = sqrt(separation_ij);
            if (separation_ij > 0.0) {
                auto cur_r = (scatter_ptr[i] - scatter_ptr[j]) / separation_ij;
                r = std::max(r, cur_r);
            }
        }
        dbi += r;
    }
    dbi /= cluster_count;
    return dbi;
}

} // namespace oneapi::dal::test::engine
