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
#include "oneapi/dal/data/accessor.hpp"

#include "example_util/utils.hpp"

using namespace oneapi;

void run(sycl::queue &queue) {
    constexpr std::int64_t row_count_train = 8;
    constexpr std::int64_t row_count_test  = 2;
    constexpr std::int64_t column_count    = 2;
    constexpr std::int64_t cluster_count   = 2;

    const float x_train_host[] = { 1.0,  1.0,  2.0,  2.0,  1.0,  2.0,  2.0,  1.0,
                                   -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0 };
    const float x_test_host[]  = { 10.0, 10.0, -10.0, -10.0 };

    const float y_test_host[] = { 0.0, 1.0 };

    const float initial_centroids_host[] = { 1.0, 0.0, -1.0, 0.0 };

    auto x_train = sycl::malloc_shared<float>(row_count_train * column_count, queue);
    queue.memcpy(x_train, x_train_host, sizeof(float) * row_count_train * column_count).wait();
    const auto x_train_table = dal::homogen_table{ queue, x_train, row_count_train, column_count, dal::empty_delete<const float>() };

    auto initial_centroids = sycl::malloc_shared<float>(cluster_count * column_count, queue);
    queue
        .memcpy(initial_centroids,
                initial_centroids_host,
                sizeof(float) * cluster_count * column_count)
        .wait();
    const auto initial_centroids_table =
        dal::homogen_table{ queue, initial_centroids, cluster_count, column_count, dal::empty_delete<const float>() };

    auto x_test = sycl::malloc_shared<float>(row_count_test * column_count, queue);
    queue.memcpy(x_test, x_test_host, sizeof(float) * row_count_test * column_count).wait();
    const auto x_test_table = dal::homogen_table{ queue, x_test, row_count_test, column_count, dal::empty_delete<const float>() };

    const auto y_test_table = dal::homogen_table{ queue, y_test_host, row_count_test, 1, dal::empty_delete<const float>() };

    const auto kmeans_desc = dal::kmeans::descriptor<>()
                                 .set_cluster_count(cluster_count)
                                 .set_max_iteration_count(100)
                                 .set_accuracy_threshold(0.001);

    const auto result_train =
        dal::train(queue, kmeans_desc, x_train_table, initial_centroids_table);

    std::cout << "Iteration count: " << result_train.get_iteration_count() << std::endl;
    std::cout << "Objective function value: " << result_train.get_objective_function_value()
              << std::endl;
    std::cout << "Lables:" << std::endl << result_train.get_labels() << std::endl;
    std::cout << "Centroids:" << std::endl << result_train.get_model().get_centroids() << std::endl;

    const auto result_test = dal::infer(queue, kmeans_desc, result_train.get_model(), x_test_table);

    std::cout << "Infer result:" << std::endl << result_test.get_labels() << std::endl;

    std::cout << "Ground truth:" << std::endl << y_test_table << std::endl;

    sycl::free(x_train, queue);
    sycl::free(x_test, queue);
}

int main(int argc, char const *argv[]) {
    for (auto device : list_devices()) {
        std::cout << "Running on " << device.get_info<sycl::info::device::name>() << std::endl;
        auto queue = sycl::queue{ device };
        run(queue);
    }
    return 0;
}
