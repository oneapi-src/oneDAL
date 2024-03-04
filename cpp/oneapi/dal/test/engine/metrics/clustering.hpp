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
#include "oneapi/dal/test/engine/linalg/matrix.hpp"

#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::test::engine {

template <typename Float = double>
Float davies_bouldin_index(const table& data, const table& centroids, const table& assignments) {
    INFO("check if data shape is expected to be consistent");
    REQUIRE(data.get_row_count() == assignments.get_row_count());
    REQUIRE(data.get_column_count() == centroids.get_column_count());
    REQUIRE(assignments.get_column_count() == 1);

    const auto cluster_count = centroids.get_row_count();
    const auto feature_count = centroids.get_column_count();
    const auto row_count = data.get_row_count();

    auto scatter = array<Float>::zeros(cluster_count);
    auto counters = array<std::int32_t>::zeros(cluster_count);
    auto scatter_ptr = scatter.get_mutable_data();
    auto counter_ptr = counters.get_mutable_data();

    const auto data_matrix =
        linalg::matrix<Float>::wrap(row_accessor<const Float>(data).pull({ 0, -1 }),
                                    { row_count, feature_count });
    const auto centroid_matrix =
        linalg::matrix<Float>::wrap(row_accessor<const Float>(centroids).pull({ 0, -1 }),
                                    { cluster_count, feature_count });

    const auto cluster_ids = row_accessor<const std::int32_t>(assignments).pull({ 0, -1 });

    for (std::int64_t i = 0; i < row_count; ++i) {
        auto cluster_id = cluster_ids[i];
        if (cluster_id < 0) {
            continue;
        }

        counter_ptr[cluster_id]++;
        Float distance_sq = 0.0;
        for (std::int64_t j = 0; j < feature_count; ++j) {
            const auto diff = data_matrix.get(i, j) - centroid_matrix.get(cluster_id, j);
            distance_sq += diff * diff;
        }
        scatter_ptr[cluster_id] += sqrt(distance_sq);
    }
    for (std::int64_t i = 0; i < cluster_count; ++i) {
        scatter_ptr[i] = scatter_ptr[i] / counter_ptr[i];
    }
    Float dbi = 0.0;
    for (std::int64_t i = 0; i < cluster_count; ++i) {
        Float r = 0;
        for (std::int64_t j = 0; j < cluster_count; ++j) {
            if (j == i)
                continue;
            Float separation_ij = 0.0;
            for (std::int64_t k = 0; k < feature_count; ++k) {
                auto diff = centroid_matrix.get(i, k) - centroid_matrix.get(j, k);
                separation_ij += diff * diff;
            }
            separation_ij = sqrt(separation_ij);
            if (separation_ij > 0.0) {
                auto cur_r = (scatter_ptr[i] + scatter_ptr[j]) / separation_ij;
                r = std::max(r, cur_r);
            }
        }
        dbi += r;
    }
    dbi /= cluster_count;
    return dbi;
}

template <typename Float = double>
table centers_of_mass(const table& data, const table& assignments, std::int64_t cluster_count) {
    INFO("check if data shape is expected to be consistent");
    REQUIRE(data.get_row_count() == assignments.get_row_count());
    REQUIRE(assignments.get_column_count() == 1);

    const auto feature_count = data.get_column_count();
    const auto row_count = data.get_row_count();

    auto centers = array<Float>::zeros(cluster_count * feature_count);
    auto counters = array<std::int32_t>::zeros(cluster_count);
    auto center_ptr = centers.get_mutable_data();
    auto counter_ptr = counters.get_mutable_data();

    const auto data_matrix =
        linalg::matrix<Float>::wrap(row_accessor<const Float>(data).pull({ 0, -1 }),
                                    { row_count, feature_count });

    const auto cluster_ids = row_accessor<const std::int32_t>(assignments).pull({ 0, -1 });

    for (std::int64_t i = 0; i < row_count; ++i) {
        auto cluster_id = cluster_ids[i];
        if (cluster_id < 0) {
            continue;
        }
        counter_ptr[cluster_id]++;
        for (std::int64_t j = 0; j < feature_count; ++j) {
            center_ptr[cluster_id * feature_count + j] += data_matrix.get(i, j);
        }
    }
    for (std::int64_t i = 0; i < cluster_count; ++i) {
        const auto count = counter_ptr[i];
        for (std::int64_t j = 0; j < feature_count; ++j) {
            center_ptr[i * feature_count + j] /= count;
        }
    }

    return dal::homogen_table::wrap(centers, cluster_count, feature_count);
}

} // namespace oneapi::dal::test::engine
