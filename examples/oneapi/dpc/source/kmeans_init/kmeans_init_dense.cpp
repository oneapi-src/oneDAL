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

#include <sycl/sycl.hpp>
#include <iomanip>
#include <iostream>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "example_util/utils.hpp"
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/algo/kmeans_init.hpp"
#include "oneapi/dal/io/csv.hpp"

namespace dal = oneapi::dal;

template <typename Method>
void run(sycl::queue& q, const dal::table& x_train, const std::string& method_name) {
    constexpr std::int64_t cluster_count = 20;
    constexpr std::int64_t max_iteration_count = 1000;
    constexpr double accuracy_threshold = 0.01;

    const auto kmeans_init_desc =
        dal::kmeans_init::descriptor<float, Method>().set_cluster_count(cluster_count);

    const auto result_init = dal::compute(q, kmeans_init_desc, x_train);

    const auto kmeans_desc = dal::kmeans::descriptor<>()
                                 .set_cluster_count(cluster_count)
                                 .set_max_iteration_count(max_iteration_count)
                                 .set_accuracy_threshold(accuracy_threshold);

    const auto result_train = dal::train(q, kmeans_desc, x_train, result_init.get_centroids());

    std::cout << "Method: " << method_name << std::endl;
    std::cout << "Max iteration count: " << max_iteration_count
              << ", Accuracy threshold: " << accuracy_threshold << std::endl;
    std::cout << "Iteration count: " << result_train.get_iteration_count()
              << ", Objective function value: " << result_train.get_objective_function_value()
              << '\n'
              << std::endl;
}

int main(int argc, char const* argv[]) {
    const auto train_data_file_name = get_data_path("kmeans_init_dense.csv");

    for (auto d : list_devices()) {
        std::cout << "Running on " << d.get_platform().get_info<sycl::info::platform::name>()
                  << ", " << d.get_info<sycl::info::device::name>() << '\n'
                  << std::endl;
        auto q = sycl::queue{ d };

        const auto x_train =
            dal::read<dal::table>(q, dal::csv::data_source{ train_data_file_name });

        run<dal::kmeans_init::method::dense>(q, x_train, "dense");
        run<dal::kmeans_init::method::random_dense>(q, x_train, "random_dense");
        run<dal::kmeans_init::method::plus_plus_dense>(q, x_train, "plus_plus_dense");
    }
    return 0;
}
