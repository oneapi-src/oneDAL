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
#include "oneapi/dal/algo/rbf_kernel/compute.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

using namespace oneapi;
namespace rbf_kernel = oneapi::dal::rbf_kernel;

TEST(rbf_kernel_bad_arg_tests, set_sigma) {
    ASSERT_THROW((rbf_kernel::descriptor{}.set_sigma(0.0)), dal::domain_error);
    ASSERT_THROW((rbf_kernel::descriptor{}.set_sigma(-1.0)), dal::domain_error);
}

TEST(rbf_kernel_bad_arg_tests, set_x) {
    constexpr std::int64_t row_count = 2;
    constexpr std::int64_t column_count = 2;

    const float y_train[] = { 0.f, 1.f, 1.f, 1.f };
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count, column_count);
    dal::homogen_table x_train_table;

    const auto rbf_kernel_desc = rbf_kernel::descriptor{};

    ASSERT_THROW(dal::compute(rbf_kernel_desc, x_train_table, y_train_table), dal::domain_error);
}

TEST(rbf_kernel_bad_arg_tests, set_y) {
    constexpr std::int64_t row_count = 2;
    constexpr std::int64_t column_count = 2;

    const float x_train[] = { 0.f, 1.f, 1.f, 1.f };
    const auto x_train_table = dal::homogen_table::wrap(x_train, row_count, column_count);
    dal::homogen_table y_train_table;

    const auto rbf_kernel_desc = rbf_kernel::descriptor{};

    ASSERT_THROW(dal::compute(rbf_kernel_desc, x_train_table, y_train_table), dal::domain_error);
}

TEST(rbf_kernel_bad_arg_tests, incorect_columns) {
    constexpr std::int64_t row_count = 2;
    constexpr std::int64_t column_count_x = 1;
    constexpr std::int64_t column_count_y = 2;

    const float x_train[] = { 0.f, 1.f };
    const float y_train[] = { 0.f, 1.f, 1.f, 1.f };
    const auto x_train_table = dal::homogen_table::wrap(x_train, row_count, column_count_x);
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count, column_count_y);

    const auto rbf_kernel_desc = rbf_kernel::descriptor{};

    ASSERT_THROW(dal::compute(rbf_kernel_desc, x_train_table, y_train_table),
                 dal::invalid_argument);
}
