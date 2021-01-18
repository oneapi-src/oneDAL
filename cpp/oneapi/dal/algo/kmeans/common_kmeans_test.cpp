/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
#include "oneapi/dal/algo/kmeans/infer.hpp"
#include "oneapi/dal/algo/kmeans/train.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

using namespace oneapi;
namespace kmn = oneapi::dal::kmeans;

using kmn_desc = kmn::descriptor<float, kmn::method::lloyd_dense, kmn::task::clustering>;

TEST(kmeans_bad_arg_tests, set_cluster_count) {
    ASSERT_THROW((kmn_desc{}.set_cluster_count(0)), dal::domain_error);
    ASSERT_THROW((kmn_desc{}.set_cluster_count(-1)), dal::domain_error);
}

TEST(kmeans_bad_arg_tests, set_max_iteration_count) {
    ASSERT_THROW((kmn_desc{}.set_max_iteration_count(-1)), dal::domain_error);
}

TEST(kmeans_bad_arg_tests, set_accuracy_threshold) {
    ASSERT_THROW((kmn_desc{}.set_accuracy_threshold(-0.1)), dal::domain_error);
}

TEST(kmeans_bad_arg_tests, throws_if_train_data_table_is_empty) {
    constexpr std::int64_t cluster_count = 2;

    dal::homogen_table data_table;

    const auto kmeans_desc = kmn_desc{}
                                 .set_cluster_count(cluster_count)
                                 .set_max_iteration_count(4)
                                 .set_accuracy_threshold(0.001);

    ASSERT_THROW(train(kmeans_desc, data_table), dal::domain_error);
}

TEST(kmeans_bad_arg_tests, throws_if_initial_centroids_rows_dont_match_clusters_count_host_api) {
    constexpr std::int64_t row_count = 8;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t cluster_count = 2;

    const float data[] = { 1.0,  1.0,  2.0,  2.0,  1.0,  2.0,  2.0,  1.0,
                           -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0 };

    const float initial_centroids[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    const auto data_table = dal::homogen_table::wrap(data, row_count, column_count);
    const auto initial_centroids_table = dal::homogen_table::wrap(initial_centroids, 3, 2);

    const auto kmeans_desc = kmn_desc()
                                 .set_cluster_count(cluster_count)
                                 .set_max_iteration_count(4)
                                 .set_accuracy_threshold(0.001);

    ASSERT_THROW(train(kmeans_desc, data_table, initial_centroids_table), dal::invalid_argument);
}

TEST(kmeans_bad_arg_tests, throws_if_initial_centroids_columns_dont_match_data_columns_host_api) {
    constexpr std::int64_t row_count = 8;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t cluster_count = 2;

    const float data[] = { 1.0,  1.0,  2.0,  2.0,  1.0,  2.0,  2.0,  1.0,
                           -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0 };

    const float initial_centroids[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    const auto data_table = dal::homogen_table::wrap(data, row_count, column_count);
    const auto initial_centroids_table = dal::homogen_table::wrap(initial_centroids, 2, 3);

    const auto kmeans_desc = kmn_desc()
                                 .set_cluster_count(cluster_count)
                                 .set_max_iteration_count(4)
                                 .set_accuracy_threshold(0.001);

    ASSERT_THROW(train(kmeans_desc, data_table, initial_centroids_table), dal::invalid_argument);
}

TEST(kmeans_bad_arg_tests, throws_if_infer_data_table_is_empty_host_api) {
    constexpr std::int64_t row_count = 8;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t cluster_count = 2;

    const float data[] = { 1.0,  1.0,  2.0,  2.0,  1.0,  2.0,  2.0,  1.0,
                           -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0 };

    const auto data_table = dal::homogen_table::wrap(data, row_count, column_count);

    const auto kmeans_desc = kmn_desc()
                                 .set_cluster_count(cluster_count)
                                 .set_max_iteration_count(4)
                                 .set_accuracy_threshold(0.001);

    const auto result_train = train(kmeans_desc, data_table);
    ASSERT_THROW(infer(kmeans_desc, result_train.get_model(), dal::homogen_table()),
                 dal::domain_error);
}
