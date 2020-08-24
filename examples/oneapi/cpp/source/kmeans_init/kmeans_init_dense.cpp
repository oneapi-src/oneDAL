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

#include <iomanip>
#include <iostream>

#include "example_util/utils.hpp"
#include "oneapi/dal/algo/kmeans_init.hpp"
#include "oneapi/dal/io/csv.hpp"

using namespace oneapi;

template <typename Method>
void run_compute(dal::table x_train,
                 dal::table x_test,
                 std::int64_t cluster_count,
                 std::int64_t column_count,
                 const char *method_name) {
    const auto kmeans_init_desc =
        dal::kmeans_init::descriptor<float, Method>().set_cluster_count(cluster_count);

    const auto result = dal::compute(kmeans_init_desc, x_train);

    std::cout << "Initial cetroids for " << method_name << ":" << std::endl
              << result.get_centroids() << std::endl;
    std::cout << "Ground truth for " << method_name << ":" << std::endl << x_test << std::endl;
}

int main(int argc, char const *argv[]) {
    constexpr std::int64_t column_count  = 2;
    constexpr std::int64_t cluster_count = 2;

    const std::string train_data_file_name = get_data_path("kmeans_init_dense.csv");
    const std::string test_dense_data_file_name =
        get_data_path("kmeans_init_dense_ground_truth.csv");
    const std::string test_random_dense_data_file_name =
        get_data_path("kmeans_init_random_dense_ground_truth.csv");
    const std::string test_plus_plus_dense_data_file_name =
        get_data_path("kmeans_init_plus_plus_dense_ground_truth.csv");
    const std::string test_parallel_plus_data_file_name =
        get_data_path("kmeans_init_parallel_plus_dense_ground_truth.csv");

    const auto x_train = dal::read<dal::table>(dal::csv::data_source{ train_data_file_name });
    const auto x_test_dense =
        dal::read<dal::table>(dal::csv::data_source{ test_dense_data_file_name });
    const auto x_test_random_dense =
        dal::read<dal::table>(dal::csv::data_source{ test_random_dense_data_file_name });
    const auto x_test_plus_plus_dense =
        dal::read<dal::table>(dal::csv::data_source{ test_plus_plus_dense_data_file_name });
    const auto x_test_parallel_plus_dense =
        dal::read<dal::table>(dal::csv::data_source{ test_parallel_plus_data_file_name });

    run_compute<dal::kmeans_init::method::dense>(x_train,
                                                 x_test_dense,
                                                 cluster_count,
                                                 column_count,
                                                 "dense");
    run_compute<dal::kmeans_init::method::random_dense>(x_train,
                                                        x_test_random_dense,
                                                        cluster_count,
                                                        column_count,
                                                        "random_dense");
    run_compute<dal::kmeans_init::method::plus_plus_dense>(x_train,
                                                           x_test_plus_plus_dense,
                                                           cluster_count,
                                                           column_count,
                                                           "plus_plus_dense");
    run_compute<dal::kmeans_init::method::parallel_plus_dense>(x_train,
                                                               x_test_parallel_plus_dense,
                                                               cluster_count,
                                                               column_count,
                                                               "parallel_plus_dense");

    return 0;
}
