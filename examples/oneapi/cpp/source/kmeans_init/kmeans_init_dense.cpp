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
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/algo/kmeans_init.hpp"
#include "oneapi/dal/io/csv.hpp"

namespace onedal = oneapi::dal;

template <typename Method>
void run(const onedal::table& x_train, const std::string& method_name) {
    constexpr std::int64_t cluster_count = 20;
    constexpr std::int64_t max_iteration_count = 1000;
    constexpr double accuracy_threshold = 0.01;

    const auto kmeans_init_desc =
        onedal::kmeans_init::descriptor<float, Method>().set_cluster_count(cluster_count);

    const auto result_init = onedal::compute(kmeans_init_desc, x_train);

    const auto kmeans_desc = onedal::kmeans::descriptor<>()
                                 .set_cluster_count(cluster_count)
                                 .set_max_iteration_count(max_iteration_count)
                                 .set_accuracy_threshold(accuracy_threshold);

    const auto result_train = onedal::train(kmeans_desc, x_train, result_init.get_centroids());

    std::cout << "Method: " << method_name << std::endl;
    std::cout << "Max iteration count: " << max_iteration_count
              << ", Accuracy threshold: " << accuracy_threshold << std::endl;
    std::cout << "Iteration count: " << result_train.get_iteration_count()
              << ", Objective function value: " << result_train.get_objective_function_value()
              << '\n'
              << std::endl;
}

int main(int argc, char const *argv[]) {
    const std::string train_data_file_name = get_data_path("kmeans_init_dense.csv");

    const auto x_train = onedal::read<onedal::table>(onedal::csv::data_source{ train_data_file_name });

    run<onedal::kmeans_init::method::dense>(x_train, "dense");
    run<onedal::kmeans_init::method::random_dense>(x_train, "random_dense");
    run<onedal::kmeans_init::method::plus_plus_dense>(x_train, "plus_plus_dense");
    run<onedal::kmeans_init::method::parallel_plus_dense>(x_train, "parallel_plus_dense");

    return 0;
}
