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

#include "example_util/utils.hpp"
#include "oneapi/dal/algo/kmeans.hpp"

using namespace oneapi;

int main(int argc, char const *argv[]) {
    constexpr std::int64_t row_count     = 8;
    constexpr std::int64_t column_count  = 2;
    constexpr std::int64_t cluster_count = 2;

    const float data[] = { 1.0,  1.0,  2.0,  2.0,  1.0,  2.0,  2.0,  1.0,
                           -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0 };

    const float initial_centroids[] = { 0.0, 0.0, 0.0, 0.0 };

    const auto data_table = dal::homogen_table{ row_count, column_count, data };
    const auto initial_centroids_table =
        dal::homogen_table{ cluster_count, column_count, initial_centroids };

    const auto kmeans_desc = dal::kmeans::descriptor<>()
                                 .set_cluster_count(cluster_count)
                                 .set_max_iteration_count(100)
                                 .set_accuracy_threshold(0.001);

    const auto result_train = dal::train(kmeans_desc, data_table, initial_centroids_table);

    std::cout << "Iteration count: " << result_train.get_iteration_count() << std::endl;
    std::cout << "Objective function value: " << result_train.get_objective_function_value()
              << std::endl;
    std::cout << "Lables:" << std::endl << result_train.get_labels() << std::endl;
    std::cout << "Centroids:" << std::endl << result_train.get_model().get_centroids() << std::endl;

    constexpr std::int64_t row_count_test = 2;
    const float data_test[]               = { 10.0, 10.0, -10.0, -10.0 };

    const auto data_test_table = dal::homogen_table{ row_count_test, column_count, data_test };

    const auto result_test = dal::infer(kmeans_desc, result_train.get_model(), data_test_table);

    std::cout << "Infer result:" << std::endl << result_test.get_labels() << std::endl;

    return 0;
}
