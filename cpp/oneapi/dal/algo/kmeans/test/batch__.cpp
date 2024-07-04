/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/algo/kmeans/test/fixture.hpp"
#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/table/csr_accessor.hpp"

constexpr std::int64_t n_iter = 2;

namespace oneapi::dal::kmeans::test {

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

TEST("can train dense K-means") {
    // number of observations is equal to number of centroids (obvious clustering)
    /// SKIP_IF(this->not_float64_friendly());

    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();
    constexpr std::int64_t k{ 3 };

    constexpr std::int64_t row_count{ 13 };
    constexpr std::int64_t column_count{ 6 };

    const float x_values_host[] = { 1.0f, 1.0f,  0.0f,  0.0f, 0.0f,  0.0f,
                                    1.1f, 1.0f,  0.0f,  0.0f, 0.0f,  0.0f,
                                    1.0f, 1.1f,  0.0f,  0.0f, 0.0f,  0.0f,
                                    0.9f, 1.0f,  0.0f,  0.0f, 0.0f,  0.0f,
                                    0.0f, 0.0f, -1.0f, -1.0f, 0.0f,  0.0f,
                                    0.0f, 0.0f, -1.1f, -1.0f, 0.0f,  0.0f,
                                    0.0f, 0.0f, -1.0f, -1.1f, 0.0f,  0.0f,
                                    0.0f, 0.0f, -0.9f, -1.0f, 0.0f,  0.0f,
                                    0.0f, 0.0f, -1.0f, -0.9f, 0.0f,  0.0f,
                                    0.0f, 0.0f,  0.0f,  0.0f, 1.0f, -1.0f,
                                    0.0f, 0.0f,  0.0f,  0.0f, 1.1f, -1.0f,
                                    0.0f, 0.0f,  0.0f,  0.0f, 1.0f, -1.1f,
                                    0.0f, 0.0f,  0.0f,  0.0f, 0.9f, -1.0f };

    auto x_values_array = dal::array<float>::empty(q, row_count * column_count, sycl::usm::alloc::device);

    q.submit([&](sycl::handler& cgh) {
        cgh.memcpy(x_values_array.get_mutable_data(), x_values_host, row_count * column_count * sizeof(float));
    }).wait_and_throw();

    const auto x_train = dal::homogen_table::wrap(x_values_array, row_count, column_count);

    float init_centroids_ptr[] = { 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f};

    const auto init_centroids = homogen_table::wrap(init_centroids_ptr, k, column_count);

    const auto kmeans_desc = dal::kmeans::descriptor<float>()
                                 .set_cluster_count(k)
                                 .set_max_iteration_count(n_iter)
                                 .set_accuracy_threshold(0.001);

    const auto result_train = dal::train(q, kmeans_desc, x_train, init_centroids);

    std::cout << " ============ SPARSE ============ "  << std::endl;
    std::cout << "Iteration count: " << result_train.get_iteration_count() << std::endl;
    std::cout << "Iteration count: " << result_train.get_iteration_count() << std::endl;
    std::cout << "Objective function value: " << result_train.get_objective_function_value()
              << std::endl;

    const auto& responses = result_train.get_responses();
    auto responses_arr = array<float>::empty(row_count);
    auto responses_ptr = row_accessor<const float>(responses).pull(responses_arr, { 0, -1 });
    std::cout << "Responses:" << std::endl;
    for (std::int64_t i = 0; i < row_count; ++i) {
        std::cout << responses_ptr[i] << ", ";
    }
    std::cout << std::endl;
    // std::cout << "Responses:\n" << result_train.get_responses() << std::endl;
    // std::cout << "Centroids:\n" << result_train.get_model().get_centroids() << std::endl;

}

TEST("can train sparse K-means") {
    // number of observations is equal to number of centroids (obvious clustering)
    /// SKIP_IF(this->not_float64_friendly());

    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();
    constexpr std::int64_t k{ 3 };

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

    float init_centroids_ptr[] = { 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f};


    const auto init_centroids = homogen_table::wrap(init_centroids_ptr, k, column_count);

    const auto kmeans_desc = dal::kmeans::descriptor<float, dal::kmeans::method::lloyd_csr>()
                                 .set_cluster_count(k)
                                 .set_max_iteration_count(n_iter)
                                 .set_accuracy_threshold(0.001);

    const auto result_train = dal::train(q, kmeans_desc, x_train, init_centroids);

    std::cout << " ============ DENSE ============ "  << std::endl;
    std::cout << "Iteration count: " << result_train.get_iteration_count() << std::endl;
    std::cout << "Objective function value: " << result_train.get_objective_function_value()
              << std::endl;

    const auto& responses = result_train.get_responses();
    auto responses_arr = array<float>::empty(row_count);
    auto responses_ptr = row_accessor<const float>(responses).pull(responses_arr, { 0, -1 });
    std::cout << "Responses:" << std::endl;
    for (std::int64_t i = 0; i < row_count; ++i) {
        std::cout << responses_ptr[i] << ", ";
    }
    std::cout << std::endl;
}

} // namespace oneapi::dal::kmeans::test
