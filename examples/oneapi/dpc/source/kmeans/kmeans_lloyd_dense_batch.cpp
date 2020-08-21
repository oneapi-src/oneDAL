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
#include <iomanip>
#include <iostream>

#define ONEAPI_DAL_DATA_PARALLEL
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

using namespace oneapi;

void run(sycl::queue &queue) {
    const std::string train_data_file_name        = get_data_path("kmeans_dense_train_data.csv");
    const std::string initial_centroids_file_name = get_data_path("kmeans_dense_train_centroids.csv");
    const std::string test_data_file_name         = get_data_path("kmeans_dense_test_data.csv");
    const std::string test_label_file_name        = get_data_path("kmeans_dense_test_label.csv");

    const auto x_train           = dal::read<dal::table>(queue, dal::csv::data_source{train_data_file_name});
    const auto initial_centroids = dal::read<dal::table>(queue, dal::csv::data_source{initial_centroids_file_name});

    const auto x_test = dal::read<dal::table>(queue, dal::csv::data_source{test_data_file_name});
    const auto y_test = dal::read<dal::table>(queue, dal::csv::data_source{test_label_file_name});

    const auto kmeans_desc = dal::kmeans::descriptor<>()
                                 .set_cluster_count(20)
                                 .set_max_iteration_count(5)
                                 .set_accuracy_threshold(0.001);

    const auto result_train =
        dal::train(queue, kmeans_desc, x_train, initial_centroids);

    std::cout << "Iteration count: " << result_train.get_iteration_count() << std::endl;
    std::cout << "Objective function value: " << result_train.get_objective_function_value()
              << std::endl;
    std::cout << "Lables:" << std::endl << result_train.get_labels() << std::endl;
    std::cout << "Centroids:" << std::endl << result_train.get_model().get_centroids() << std::endl;

    const auto result_test = dal::infer(queue, kmeans_desc, result_train.get_model(), x_test);

    std::cout << "Infer result:" << std::endl << result_test.get_labels() << std::endl;

    std::cout << "Ground truth:" << std::endl << y_test << std::endl;
}

int main(int argc, char const *argv[]) {
    for (auto device : list_devices()) {
        std::cout << "Running on " << device.get_info<sycl::info::device::name>() << std::endl;
        auto queue = sycl::queue{ device };
        run(queue);
    }
    return 0;
}
