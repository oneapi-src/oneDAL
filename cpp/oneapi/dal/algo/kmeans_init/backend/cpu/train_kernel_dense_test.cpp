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

#include <CL/sycl.hpp>

#include "gtest/gtest.h"
#define ONEAPI_DAL_DATA_PARALLEL
#include "oneapi/dal/algo/kmeans_init.hpp"
#include "oneapi/dal/data/accessor.hpp"
#include "oneapi/dal/data/table.hpp"

using namespace oneapi::dal;

TEST(train_kernel_lloyd_dense, test1) {
    constexpr std::int64_t row_count     = 8;
    constexpr std::int64_t column_count  = 2;
    constexpr std::int64_t cluster_count = 2;

    const float data[]    = { 1.0,  1.0,  2.0,  2.0,  1.0,  2.0,  2.0,  1.0,
                           -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0 };
    const auto data_table = homogen_table{ row_count, column_count, data };

    const float centroids[] = { 1.0, 1.0, 2.0, 2.0 };

    const auto kmeans_desc = kmeans_init::descriptor<>().set_cluster_count(cluster_count);

    const auto result_train = train(kmeans_desc, data_table);

    const auto train_centroids =
        row_accessor<const float>(result_train.get_centroids()).pull().get_data();
    for (std::int64_t i = 0; i < cluster_count * column_count; ++i) {
        ASSERT_FLOAT_EQ(centroids[i], train_centroids[i]);
    }
}
