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
    INFO("check if data shape is expected to be consistent")
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

//this function works as sklearn.metrics.davies_bouldin_score
template <typename Float = double>
Float davies_bouldin_index(const table& data, const table& assignments) {
    INFO("check if data shape is expected to be consistent")
    REQUIRE(data.get_row_count() == assignments.get_row_count());
    REQUIRE(assignments.get_column_count() == 1);

    const auto data_column_count = data.get_column_count();
    const auto data_matrix =
        linalg::matrix<Float>::wrap(row_accessor<const Float>(data).pull({ 0, -1 }),
                                    { data.get_row_count(), data_column_count });
    const auto assignments_nd = linalg::matrix<std::int32_t>::wrap(
        row_accessor<const std::int32_t>(assignments).pull({ 0, -1 }),
        { assignments.get_row_count(), assignments.get_column_count() });

    auto assignments_ptr = assignments_nd.get_data();
    std::int64_t assignments_size = assignments.get_row_count() * assignments.get_column_count();
    std::set<double> cluster_indices(assignments_ptr, assignments_ptr + assignments_size);

    std::int64_t num_clusters = cluster_indices.size();

    std::vector<double> values(cluster_indices.begin(), cluster_indices.end());
    cluster_indices.clear();
    std::vector<double> dispersions(num_clusters, 0.0);
    std::vector<double> centroids(num_clusters * data_column_count, 0.0);

    for (std::int64_t cluster_idx = 0; cluster_idx < num_clusters; cluster_idx++) {
        std::vector<double> x_k;
        for (std::int64_t assignment_idx = 0; assignment_idx < assignments.get_row_count();
             assignment_idx++) {
            if (values[cluster_idx] == assignments_ptr[assignment_idx]) {
                for (std::int64_t col = 0; col < data_column_count; col++) {
                    x_k.push_back(data_matrix.get(assignment_idx * data_column_count + col));
                }
            }
        }
        std::int64_t row_count = x_k.size() / data_column_count;

        for (std::int64_t col = 0; col < data_column_count; col++) {
            double sum_per_col = 0.0;
            for (std::int64_t row = 0; row < row_count; row++) {
                sum_per_col += x_k[row * data_column_count + col];
            }
            centroids[cluster_idx * data_column_count + col] = sum_per_col / row_count;
        }
        double dist_sum = 0;
        for (std::int64_t row = 0; row < row_count; row++) {
            double sum = 0.0;
            for (std::int64_t col = 0; col < data_column_count; col++) {
                sum += ((x_k[row * data_column_count + col] -
                         centroids[col + data_column_count * cluster_idx]) *
                        (x_k[row * data_column_count + col] -
                         centroids[col + data_column_count * cluster_idx]));
            }
            dist_sum += sqrt(sum);
        }
        dispersions[cluster_idx] = dist_sum / row_count;
    }

    double dbi = 0.0;
    for (std::int64_t cluster_i = 0; cluster_i < num_clusters; cluster_i++) {
        double max_ratio = 0.0;
        for (std::int64_t cluster_j = 0; cluster_j < num_clusters; cluster_j++) {
            if (cluster_i != cluster_j) {
                double m_ij = 0.0;
                for (std::int64_t col = 0; col < data_column_count; col++) {
                    m_ij += ((centroids[cluster_i * data_column_count + col] -
                              centroids[cluster_j * data_column_count + col]) *
                             (centroids[cluster_i * data_column_count + col] -
                              centroids[cluster_j * data_column_count + col]));
                }
                const double s_ij = dispersions[cluster_i] + dispersions[cluster_j];
                if (std::sqrt(m_ij) > 0) {
                    const double ratio = s_ij / std::sqrt(m_ij);
                    max_ratio = std::max(ratio, max_ratio);
                }
            }
        }
        dbi += max_ratio;
    }

    dbi /= num_clusters;

    return dbi;
}

template <typename Float = double>
table centers_of_mass(const table& data, const table& assignments, std::int64_t cluster_count) {
    INFO("check if data shape is expected to be consistent")
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
