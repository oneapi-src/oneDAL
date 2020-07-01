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

#include "oneapi/dal/algo/kmeans.hpp"
#include "example_util/utils.hpp"

using namespace oneapi;

int main(int argc, char const* argv[]) {
    constexpr std::int64_t row_count    = 8;
    constexpr std::int64_t column_count = 7;

    const float data[] = { 1.f,  2.f, 3.f,  4.f,  5.f,  6.f,  -5.f,
                           1.f, -1.f, 0.f,  3.f, 1.f, 2.f,  3.f,
                           4.f,  5.f,  6.f,  1.f,  0.f,  0.f, 0.f,
                           1.f,  2.f, 5.f, 2.f,  9.f, 3.f,  2.f,
                           -4.f, 3.f,  0.f,  4.f, 2.f,  7.f,  5.f,
                           4.f, 2.f,  0.f, -4.f, 0.f,  3.f,  -8.f,
                           2.f,  5.f, 5.f,  -6.f, 3.f, 0.f, -9.f,
                           3.f, 1.f,  -3.f, 3.f,  5.f,  1.f,  7.f };

    const float initial_centroids[] = { 1.f,  2.f, 3.f,  4.f,  5.f,  6.f,  -5.f,
                                        1.f,  2.f, 5.f, 2.f,  9.f, 3.f,  2.f };

    const auto data_table = dal::homogen_table{ row_count, column_count, data };
    const auto initial_centroids_table = dal::homogen_table{ 1, column_count, initial_centroids };

    const auto kmeans_desc = dal::kmeans::descriptor<>()
                                 .set_cluster_count(2)
                                 .set_max_iteration_count(5)
                                 .set_accuracy_threshold(1.0);

    const auto result_train = dal::train(kmeans_desc, data_table, initial_centroids_table);

    std::cout << "Iteration count: " << result_train.get_iteration_count() << std::endl;
    std::cout << "Objective function value: " << result_train.get_objective_function_value() << std::endl;
    std::cout << "Lables:" << std::endl << result_train.get_labels() << std::endl;
    std::cout << "Centroids:" << std::endl << result_train.get_model().get_centroids() << std::endl;

    const float data_test[] = { 1.f,  2.f, 5.f,  4.f,  0.f,  3.f,  1.f,
                                2.f,  3.f, 1.f, 2.f,  9.f, 2.f,  2.f };

    const auto data_test_table = dal::homogen_table{ 2, column_count, data_test };

    const auto result_test = dal::infer(kmeans_desc, result_train.get_model(), data_test_table);

    std::cout << "Infer result:" << std::endl
        << result_test.get_labels() << std::endl;

    return 0;
}
