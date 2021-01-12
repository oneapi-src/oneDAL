/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "gtest/gtest.h"
#include "oneapi/dal/algo/linear_kernel/compute.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

using namespace oneapi;
namespace linear_kernel = oneapi::dal::linear_kernel;

TEST(linear_kernel_bad_arg_tests, set_x) {
    constexpr std::int64_t row_count = 2;
    constexpr std::int64_t column_count = 2;

    const float y_train[] = { 0.f, 1.f, 1.f, 1.f };
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count, column_count);
    dal::homogen_table x_train_table;

    const auto linear_kernel_desc = linear_kernel::descriptor{};

    auto selector = sycl::gpu_selector();
    auto queue = sycl::queue(selector);

    ASSERT_THROW(dal::compute(queue, linear_kernel_desc, x_train_table, y_train_table),
                 dal::domain_error);
}

TEST(linear_kernel_bad_arg_tests, set_y) {
    constexpr std::int64_t row_count = 2;
    constexpr std::int64_t column_count = 2;

    const float x_train[] = { 0.f, 1.f, 1.f, 1.f };
    const auto x_train_table = dal::homogen_table::wrap(x_train, row_count, column_count);
    dal::homogen_table y_train_table;

    const auto linear_kernel_desc = linear_kernel::descriptor{};

    auto selector = sycl::gpu_selector();
    auto queue = sycl::queue(selector);

    ASSERT_THROW(dal::compute(queue, linear_kernel_desc, x_train_table, y_train_table),
                 dal::domain_error);
}

TEST(linear_kernel_bad_arg_tests, incorect_columns) {
    constexpr std::int64_t row_count = 2;
    constexpr std::int64_t column_count_x = 1;
    constexpr std::int64_t column_count_y = 2;

    const float x_train[] = { 0.f, 1.f };
    const float y_train[] = { 0.f, 1.f, 1.f, 1.f };
    const auto x_train_table = dal::homogen_table::wrap(x_train, row_count, column_count_x);
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count, column_count_y);

    const auto linear_kernel_desc = linear_kernel::descriptor{};

    auto selector = sycl::gpu_selector();
    auto queue = sycl::queue(selector);

    ASSERT_THROW(dal::compute(queue, linear_kernel_desc, x_train_table, y_train_table),
                 dal::invalid_argument);
}
