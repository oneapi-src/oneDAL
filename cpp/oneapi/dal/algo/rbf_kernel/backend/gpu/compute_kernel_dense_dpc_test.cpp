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

#include "gtest/gtest.h"
#define ONEAPI_DAL_DATA_PARALLEL
#include "oneapi/dal/algo/rbf_kernel.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include <cmath>

using namespace oneapi::dal;
using std::int32_t;

TEST(rbf_kernel_dense_gpu_test, can_compute_unit_matrix) {
    constexpr std::int64_t row_count = 2;
    constexpr std::int64_t column_count = 2;
    const float x_host[] = {
        1.f,
        1.f,
        1.f,
        1.f,
    };
    const float y_host[] = {
        1.f,
        1.f,
        1.f,
        1.f,
    };

    auto selector = sycl::gpu_selector();
    auto queue = sycl::queue(selector);

    auto x = sycl::malloc_shared<float>(row_count * column_count, queue);
    queue.memcpy(x, x_host, sizeof(float) * row_count * column_count).wait();

    auto y = sycl::malloc_shared<float>(row_count * column_count, queue);
    queue.memcpy(y, y_host, sizeof(float) * row_count * column_count).wait();

    const auto x_table = homogen_table::wrap(queue, x, row_count, column_count);
    const auto y_table = homogen_table::wrap(queue, y, row_count, column_count);

    const auto kernel_desc = rbf_kernel::descriptor{};
    const auto values_table = compute(queue, kernel_desc, x_table, y_table).get_values();
    ASSERT_EQ(values_table.get_row_count(), row_count);
    ASSERT_EQ(values_table.get_column_count(), row_count);

    const auto values = row_accessor<const float>(values_table).pull(queue);
    for (size_t i = 0; i < values.get_count(); i++) {
        ASSERT_FLOAT_EQ(values[i], 1.f);
    }

    sycl::free(x, queue);
    sycl::free(y, queue);
}

TEST(rbf_kernel_dense_gpu_test, can_compute_same_unit_matrix) {
    constexpr std::int64_t row_count = 2;
    constexpr std::int64_t column_count = 2;
    const float x_host[] = {
        1.f,
        1.f,
        1.f,
        1.f,
    };

    auto selector = sycl::gpu_selector();
    auto queue = sycl::queue(selector);

    auto x = sycl::malloc_shared<float>(row_count * column_count, queue);
    queue.memcpy(x, x_host, sizeof(float) * row_count * column_count).wait();
    const auto x_table = homogen_table::wrap(queue, x, row_count, column_count);

    const auto kernel_desc = rbf_kernel::descriptor{};
    const auto values_table = compute(queue, kernel_desc, x_table, x_table).get_values();
    ASSERT_EQ(values_table.get_row_count(), row_count);
    ASSERT_EQ(values_table.get_column_count(), row_count);

    const auto values = row_accessor<const float>(values_table).pull(queue);
    for (size_t i = 0; i < values.get_count(); i++) {
        ASSERT_FLOAT_EQ(values[i], 1.f);
    }

    sycl::free(x, queue);
}

TEST(rbf_kernel_dense_gpu_test, can_compute_one_element) {
    constexpr std::int64_t row_count = 1;
    constexpr std::int64_t column_count = 1;
    const float x_host[] = {
        1.f,
    };
    const float y_host[] = {
        0.1f,
    };

    auto selector = sycl::gpu_selector();
    auto queue = sycl::queue(selector);

    auto x = sycl::malloc_shared<float>(row_count * column_count, queue);
    queue.memcpy(x, x_host, sizeof(float) * row_count * column_count).wait();

    auto y = sycl::malloc_shared<float>(row_count * column_count, queue);
    queue.memcpy(y, y_host, sizeof(float) * row_count * column_count).wait();

    const auto x_table = homogen_table::wrap(queue, x, row_count, column_count);
    const auto y_table = homogen_table::wrap(queue, y, row_count, column_count);

    const auto kernel_desc = rbf_kernel::descriptor{};
    const auto values_table = compute(queue, kernel_desc, x_table, y_table).get_values();
    ASSERT_EQ(values_table.get_row_count(), row_count);
    ASSERT_EQ(values_table.get_column_count(), row_count);

    const auto values = row_accessor<const float>(values_table).pull(queue);
    const double ref_value = std::exp(-0.5 * (x_host[0] - y_host[0]) * (x_host[0] - y_host[0]));
    ASSERT_FLOAT_EQ(values[0], ref_value);

    sycl::free(x, queue);
    sycl::free(y, queue);
}

TEST(rbf_kernel_dense_gpu_test, can_compute_diff_matrix) {
    constexpr std::int64_t row_count_x = 2;
    constexpr std::int64_t row_count_y = 3;
    constexpr std::int64_t column_count = 1;
    const float x_host[] = {
        1.f,
        2.f,
    };
    const float y_host[] = {
        1.f,
        1.f,
        3.f,
    };

    auto selector = sycl::gpu_selector();
    auto queue = sycl::queue(selector);

    auto x = sycl::malloc_shared<float>(row_count_x * column_count, queue);
    queue.memcpy(x, x_host, sizeof(float) * row_count_x * column_count).wait();

    auto y = sycl::malloc_shared<float>(row_count_y * column_count, queue);
    queue.memcpy(y, y_host, sizeof(float) * row_count_y * column_count).wait();

    const auto x_table = homogen_table::wrap(queue, x, row_count_x, column_count);
    const auto y_table = homogen_table::wrap(queue, y, row_count_y, column_count);

    const auto kernel_desc = rbf_kernel::descriptor{};
    const auto values_table = compute(queue, kernel_desc, x_table, y_table).get_values();
    ASSERT_EQ(values_table.get_row_count(), row_count_x);
    ASSERT_EQ(values_table.get_column_count(), row_count_y);

    const auto values = row_accessor<const float>(values_table).pull(queue);
    for (std::int64_t i = 0; i < row_count_x; i++) {
        for (std::int64_t j = 0; j < row_count_y; j++) {
            const double ref_value =
                std::exp(-0.5 * (x_host[i] - y_host[j]) * (x_host[i] - y_host[j]));

            ASSERT_FLOAT_EQ(values[i * row_count_y + j], ref_value);
        }
    }

    sycl::free(x, queue);
    sycl::free(y, queue);
}

TEST(rbf_kernel_dense_gpu_test, can_compute_diff_matrix_not_default_params) {
    constexpr std::int64_t row_count_x = 2;
    constexpr std::int64_t row_count_y = 3;
    constexpr std::int64_t column_count = 1;
    const float x_host[] = {
        1.f,
        2.f,
    };
    const float y_host[] = {
        1.f,
        1.f,
        3.f,
    };

    auto selector = sycl::gpu_selector();
    auto queue = sycl::queue(selector);

    auto x = sycl::malloc_shared<float>(row_count_x * column_count, queue);
    queue.memcpy(x, x_host, sizeof(float) * row_count_x * column_count).wait();

    auto y = sycl::malloc_shared<float>(row_count_y * column_count, queue);
    queue.memcpy(y, y_host, sizeof(float) * row_count_y * column_count).wait();

    const auto x_table = homogen_table::wrap(queue, x, row_count_x, column_count);
    const auto y_table = homogen_table::wrap(queue, y, row_count_y, column_count);

    constexpr double sigma = 0.9;
    const auto kernel_desc = rbf_kernel::descriptor{}.set_sigma(sigma);
    const auto values_table = compute(queue, kernel_desc, x_table, y_table).get_values();
    ASSERT_EQ(values_table.get_row_count(), row_count_x);
    ASSERT_EQ(values_table.get_column_count(), row_count_y);

    const auto values = row_accessor<const float>(values_table).pull(queue);
    for (std::int64_t i = 0; i < row_count_x; i++) {
        for (std::int64_t j = 0; j < row_count_y; j++) {
            const double inv_sigma = 1.0 / (sigma * sigma);
            const double ref_value =
                std::exp(-0.5 * inv_sigma * (x_host[i] - y_host[j]) * (x_host[i] - y_host[j]));

            ASSERT_FLOAT_EQ(values[i * row_count_y + j], ref_value);
        }
    }

    sycl::free(x, queue);
    sycl::free(y, queue);
}
