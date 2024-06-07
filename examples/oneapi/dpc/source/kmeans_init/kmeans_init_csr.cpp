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
#include "oneapi/dal/table/csr.hpp"

namespace dal = oneapi::dal;

const dal::csr_table make_device_csr_table_from_host_data(sycl::queue &q,
                                                          std::int64_t row_count,
                                                          std::int64_t column_count,
                                                          const float * const x_values_host,
                                                          const std::int64_t* const x_column_indices_host,
                                                          const std::int64_t* x_row_offsets_host,
                                                          const dal::array<float>& x_values_array,
                                                          const dal::array<std::int64_t>& x_column_indices_array,
                                                          const dal::array<std::int64_t>& x_row_offsets_array) {
    const std::int64_t element_count = x_values_array.get_count();
    const std::int64_t x_row_offsets_count = x_row_offsets_array.get_count();

    auto* const x_values = x_values_array.get_mutable_data();
    auto* const x_column_indices = x_column_indices_array.get_mutable_data();
    auto* const x_row_offsets = x_row_offsets_array.get_mutable_data();

    auto x_values_event = q.submit([&](sycl::handler& cgh) {
        cgh.memcpy(x_values, x_values_host, element_count * sizeof(float));
    });

    auto x_column_indices_event = q.submit([&](sycl::handler& cgh) {
        cgh.memcpy(x_column_indices, x_column_indices_host, element_count * sizeof(std::int64_t));
    });

    auto x_row_offsets_event = q.submit([&](sycl::handler& cgh) {
        cgh.memcpy(x_row_offsets, x_row_offsets_host, x_row_offsets_count * sizeof(std::int64_t));
    });

    return dal::csr_table::wrap(q,
                                x_values,
                                x_column_indices,
                                x_row_offsets,
                                row_count,
                                column_count,
                                dal::sparse_indexing::one_based,
                                { x_values_event, x_column_indices_event, x_row_offsets_event });
}

template <typename Method>
void run(sycl::queue& q, const dal::table& x_train, const std::string& method_name) {
    constexpr std::int64_t cluster_count = 3;
    constexpr std::int64_t max_iteration_count = 1000;
    constexpr double accuracy_threshold = 0.0001;

    const auto kmeans_init_desc =
        dal::kmeans_init::descriptor<float, Method>().set_cluster_count(cluster_count);

    const auto result_init = dal::compute(q, kmeans_init_desc, x_train);

    const auto kmeans_desc = dal::kmeans::descriptor<>()
                                 .set_cluster_count(cluster_count)
                                 .set_max_iteration_count(max_iteration_count)
                                 .set_accuracy_threshold(accuracy_threshold);

    const auto result_train = dal::train(q, kmeans_desc, x_train, result_init.get_centroids());

    std::cout << "Method: " << method_name << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "centroids: \n" << result_init.get_centroids() << std::endl;

    std::cout << "Max iteration count: " << max_iteration_count
              << ", Accuracy threshold: " << accuracy_threshold << std::endl;
    std::cout << "Iteration count: " << result_train.get_iteration_count()
              << ", Objective function value: " << result_train.get_objective_function_value()
              << '\n'
              << std::endl;
}

int main(int argc, char const* argv[]) {
    // const auto train_data_file_name = get_data_path("kmeans_init_dense.csv");

    for (auto d : list_devices()) {
        std::cout << "Running on " << d.get_platform().get_info<sycl::info::platform::name>()
                  << ", " << d.get_info<sycl::info::device::name>() << '\n'
                  << std::endl;
        auto q = sycl::queue{ d };

        constexpr std::int64_t row_count{ 13 };
        constexpr std::int64_t column_count{ 6 };
        constexpr std::int64_t element_count{ 26 };

        const float x_values_host[] = { 1, 1, 1.1, 1, 1, 1.1, 0.9, 1, -1, -1, -1.1, -1, -1, -1.1, -0.9, -1, -1, -0.9, 1, -1, 1.1, -1, 1, -1.1, 0.9, -1 };
        const std::int64_t x_column_indices_host[] = { 1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 5, 6, 5, 6, 5, 6, 5, 6 };
        const std::int64_t x_row_offsets_host[] = { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27 };

        auto x_values_array = dal::array<float>::empty(q, element_count, sycl::usm::alloc::device);
        auto x_column_indices_array = dal::array<std::int64_t>::empty(q, element_count, sycl::usm::alloc::device);
        auto x_row_offsets_array = dal::array<std::int64_t>::empty(q, row_count + 1, sycl::usm::alloc::device);

        const auto x_train = make_device_csr_table_from_host_data(q,
                                                                row_count,
                                                                column_count,
                                                                x_values_host,
                                                                x_column_indices_host,
                                                                x_row_offsets_host,
                                                                x_values_array,
                                                                x_column_indices_array,
                                                                x_row_offsets_array);

        run<dal::kmeans_init::method::plus_plus_csr>(q, x_train, "plus_plus_csr");
        run<dal::kmeans_init::method::random_csr>(q, x_train, "random_csr");
        // run<dal::kmeans_init::method::plus_plus_csr>(q, x_train, "parallel_plus_csr");
    }
    return 0;
}
