/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "gtest/gtest.h"
#include "oneapi/dal/algo/kmeans_init/compute.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

using namespace oneapi::dal;

TEST(kmeans_init_cpu, compute_result) {
    constexpr std::int64_t row_count = 8;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t cluster_count = 2;

    const float data[] = { 1.0,  1.0,  2.0,  2.0,  1.0,  2.0,  2.0,  1.0,
                           -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0 };
    const auto data_table = homogen_table::wrap(data, row_count, column_count);

    const float centroids[] = { 1.0, 1.0, 2.0, 2.0 };

    const auto kmeans_desc =
        kmeans_init::descriptor<float, kmeans_init::method::dense>().set_cluster_count(
            cluster_count);

    const auto result_compute = compute(kmeans_desc, data_table);

    const auto compute_centroids =
        row_accessor<const float>(result_compute.get_centroids()).pull().get_data();
    for (std::int64_t i = 0; i < cluster_count * column_count; ++i) {
        ASSERT_FLOAT_EQ(centroids[i], compute_centroids[i]);
    }
}
