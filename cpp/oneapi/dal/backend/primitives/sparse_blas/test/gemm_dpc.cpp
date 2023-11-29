/*******************************************************************************
* Copyright 2023 Intel Corporation
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


#include "oneapi/dal/backend/primitives/sparse_blas.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::test {

namespace te = dal::test::engine;

TEST("can create sparse matrix handle") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();
    constexpr std::int64_t row_count{ 4 };
    constexpr std::int64_t column_count{ 4 };
    constexpr std::int64_t element_count{ 7 };

    const float data_host[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    const std::int64_t column_indices_host[] = { 1, 2, 4, 3, 2, 4, 2 };
    const std::int64_t row_offsets_host[] = { 1, 4, 5, 7, 8 };

    auto* const data = sycl::malloc_device<float>(element_count, q);
    auto* const column_indices = sycl::malloc_device<std::int64_t>(element_count, q);
    auto* const row_offsets = sycl::malloc_device<std::int64_t>(row_count + 1, q);

    auto data_event = q.submit([&](sycl::handler& cgh) {
        cgh.memcpy(data, data_host, element_count * sizeof(float));
    });

    auto column_indices_event = q.submit([&](sycl::handler& cgh) {
        cgh.memcpy(column_indices, column_indices_host, element_count * sizeof(std::int64_t));
    });

    auto row_offsets_event = q.submit([&](sycl::handler& cgh) {
        cgh.memcpy(row_offsets, row_offsets_host, (row_count + 1) * sizeof(std::int64_t));
    });

    oneapi::mkl::sparse::matrix_handle_t handle = nullptr;
    oneapi::mkl::sparse::init_matrix_handle(&handle);

    oneapi::mkl::sparse::set_csr_data(q, handle, row_count, column_count, oneapi::mkl::index_base::one,
        row_offsets,
        column_indices,
        data,
        {data_event, column_indices_event, row_offsets_event} ).wait();

    sycl::free(data, q);
    sycl::free(column_indices, q);
    sycl::free(row_offsets, q);
}


TEST("can create oneDAL sparse matrix handle") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();
    constexpr std::int64_t row_count{ 4 };
    constexpr std::int64_t column_count{ 4 };
    constexpr std::int64_t element_count{ 7 };

    const float data_host[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    const std::int64_t column_indices_host[] = { 1, 2, 4, 3, 2, 4, 2 };
    const std::int64_t row_offsets_host[] = { 1, 4, 5, 7, 8 };

    auto* const data = sycl::malloc_device<float>(element_count, q);
    auto* const column_indices = sycl::malloc_device<std::int64_t>(element_count, q);
    auto* const row_offsets = sycl::malloc_device<std::int64_t>(row_count + 1, q);

    auto data_event = q.submit([&](sycl::handler& cgh) {
        cgh.memcpy(data, data_host, element_count * sizeof(float));
    });

    auto column_indices_event = q.submit([&](sycl::handler& cgh) {
        cgh.memcpy(column_indices, column_indices_host, element_count * sizeof(std::int64_t));
    });

    auto row_offsets_event = q.submit([&](sycl::handler& cgh) {
        cgh.memcpy(row_offsets, row_offsets_host, (row_count + 1) * sizeof(std::int64_t));
    });


    dal::backend::primitives::sparse_matrix_handle handle;

    dal::backend::primitives::set_csr_data(q, handle, row_count, column_count, sparse_indexing::one_based,
        data,
        column_indices,
        row_offsets,
        {data_event, column_indices_event, row_offsets_event} ).wait();

    sycl::free(data, q);
    sycl::free(column_indices, q);
    sycl::free(row_offsets, q);
}

}
