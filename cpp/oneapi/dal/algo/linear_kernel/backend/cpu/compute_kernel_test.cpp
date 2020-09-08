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

#include "gtest/gtest.h"
#include "oneapi/dal/algo/linear_kernel/compute.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

using namespace oneapi::dal;
using std::int32_t;

TEST(linear_kernel_dense_test, can_compute_unit_matrix) {
    constexpr std::int64_t row_count = 2;
    constexpr std::int64_t column_count = 2;
    const float x_data[] = {
        1.f,
        1.f,
        1.f,
        1.f,
    };
    const float y_data[] = {
        1.f,
        1.f,
        1.f,
        1.f,
    };

    const auto x_table = homogen_table::wrap(x_data, row_count, column_count);
    const auto y_table = homogen_table::wrap(y_data, row_count, column_count);

    const auto kernel_desc = linear_kernel::descriptor{};
    const auto values_table = compute(kernel_desc, x_table, y_table).get_values();

    ASSERT_EQ(values_table.get_row_count(), row_count);
    ASSERT_EQ(values_table.get_column_count(), row_count);

    const auto values = row_accessor<const float>(values_table).pull();

    for (std::int64_t i = 0; i < values.get_count(); i++) {
        ASSERT_FLOAT_EQ(values[i], static_cast<float>(column_count));
    }
}

TEST(linear_kernel_dense_test, can_compute_same_unit_matrix) {
    constexpr std::int64_t row_count = 2;
    constexpr std::int64_t column_count = 2;
    const float x_data[] = {
        1.f,
        1.f,
        1.f,
        1.f,
    };

    const auto x_table = homogen_table::wrap(x_data, row_count, column_count);

    const auto kernel_desc = linear_kernel::descriptor{};
    const auto values_table = compute(kernel_desc, x_table, x_table).get_values();
    ASSERT_EQ(values_table.get_row_count(), row_count);
    ASSERT_EQ(values_table.get_column_count(), row_count);

    const auto values = row_accessor<const float>(values_table).pull();
    for (std::int64_t i = 0; i < values.get_count(); i++) {
        ASSERT_FLOAT_EQ(values[i], static_cast<float>(column_count));
    }
}

TEST(linear_kernel_dense_test, can_compute_one_element) {
    constexpr std::int64_t row_count = 1;
    constexpr std::int64_t column_count = 1;
    const float x_data[] = {
        0.5f,
    };
    const float y_data[] = {
        10.f,
    };

    const auto x_table = homogen_table::wrap(x_data, row_count, column_count);
    const auto y_table = homogen_table::wrap(y_data, row_count, column_count);

    const auto kernel_desc = linear_kernel::descriptor{};
    const auto values_table = compute(kernel_desc, x_table, y_table).get_values();
    ASSERT_EQ(values_table.get_row_count(), row_count);
    ASSERT_EQ(values_table.get_column_count(), row_count);

    const auto values = row_accessor<const float>(values_table).pull();
    ASSERT_FLOAT_EQ(values[0], 5.f);
}

TEST(linear_kernel_dense_test, can_compute_simple_matrix) {
    constexpr std::int64_t row_count = 2;
    constexpr std::int64_t column_count = 2;
    const float x_data[] = {
        1.f,
        2.f,
        1.f,
        1.f,
    };
    const float y_data[] = {
        1.f,
        1.f,
        3.f,
        1.f,
    };

    const auto x_table = homogen_table::wrap(x_data, row_count, column_count);
    const auto y_table = homogen_table::wrap(y_data, row_count, column_count);

    const auto kernel_desc = linear_kernel::descriptor{};
    const auto values_table = compute(kernel_desc, x_table, y_table).get_values();
    ASSERT_EQ(values_table.get_row_count(), row_count);
    ASSERT_EQ(values_table.get_column_count(), row_count);

    const auto values = row_accessor<const float>(values_table).pull();
    ASSERT_FLOAT_EQ(values[0], 3.f);
    ASSERT_FLOAT_EQ(values[1], 5.f);
    ASSERT_FLOAT_EQ(values[2], 2.f);
    ASSERT_FLOAT_EQ(values[3], 4.f);
}

TEST(linear_kernel_dense_test, can_compute_same_simple_matrix) {
    constexpr std::int64_t row_count = 2;
    constexpr std::int64_t column_count = 2;
    const float x_data[] = {
        1.f,
        2.f,
        1.f,
        3.f,
    };

    const auto x_table = homogen_table::wrap(x_data, row_count, column_count);

    const auto kernel_desc = linear_kernel::descriptor{};
    const auto values_table = compute(kernel_desc, x_table, x_table).get_values();
    ASSERT_EQ(values_table.get_row_count(), row_count);
    ASSERT_EQ(values_table.get_column_count(), row_count);

    const auto values = row_accessor<const float>(values_table).pull();
    ASSERT_FLOAT_EQ(values[0], 5.f);
    ASSERT_FLOAT_EQ(values[1], 7.f);
    ASSERT_FLOAT_EQ(values[2], 7.f);
    ASSERT_FLOAT_EQ(values[3], 10.f);
}

TEST(linear_kernel_dense_test, can_compute_diff_matrix) {
    constexpr std::int64_t row_count_x = 2;
    constexpr std::int64_t row_count_y = 3;
    constexpr std::int64_t column_count = 1;
    const float x_data[] = {
        1.f,
        2.f,
    };
    const float y_data[] = {
        1.f,
        1.f,
        3.f,
    };

    const auto x_table = homogen_table::wrap(x_data, row_count_x, column_count);
    const auto y_table = homogen_table::wrap(y_data, row_count_y, column_count);

    const auto kernel_desc = linear_kernel::descriptor{};
    const auto values_table = compute(kernel_desc, x_table, y_table).get_values();
    ASSERT_EQ(values_table.get_row_count(), row_count_x);
    ASSERT_EQ(values_table.get_column_count(), row_count_y);

    const auto values = row_accessor<const float>(values_table).pull();
    ASSERT_FLOAT_EQ(values[0], 1.f);
    ASSERT_FLOAT_EQ(values[1], 1.f);
    ASSERT_FLOAT_EQ(values[2], 3.f);
    ASSERT_FLOAT_EQ(values[3], 2.f);
}

TEST(linear_kernel_dense_test, can_compute_diff_matrix_not_default_params) {
    constexpr std::int64_t row_count_x = 2;
    constexpr std::int64_t row_count_y = 3;
    constexpr std::int64_t column_count = 1;
    const float x_data[] = {
        1.f,
        2.f,
    };
    const float y_data[] = {
        1.f,
        1.f,
        3.f,
    };

    const auto x_table = homogen_table::wrap(x_data, row_count_x, column_count);
    const auto y_table = homogen_table::wrap(y_data, row_count_y, column_count);

    const auto kernel_desc = linear_kernel::descriptor{}.set_scale(2.0).set_shift(1.0);
    const auto values_table = compute(kernel_desc, x_table, y_table).get_values();
    ASSERT_EQ(values_table.get_row_count(), row_count_x);
    ASSERT_EQ(values_table.get_column_count(), row_count_y);

    const auto values = row_accessor<const float>(values_table).pull();
    ASSERT_FLOAT_EQ(values[0], 3.f);
    ASSERT_FLOAT_EQ(values[1], 3.f);
    ASSERT_FLOAT_EQ(values[2], 7.f);
    ASSERT_FLOAT_EQ(values[3], 5.f);
}
