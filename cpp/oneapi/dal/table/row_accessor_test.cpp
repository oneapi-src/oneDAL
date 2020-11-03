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

#include "oneapi/dal/table/row_accessor.hpp"
#include "gtest/gtest.h"
#include "oneapi/dal/table/homogen.hpp"

using namespace oneapi::dal;

TEST(homogen_table_test, can_read_table_data_via_row_accessor) {
    using oneapi::dal::detail::empty_delete;

    double data[] = { 1.0, 2.0, 3.0, -1.0, -2.0, -3.0 };

    homogen_table t{ data, 2, 3, empty_delete<const double>() };
    const auto rows_block = row_accessor<const double>(t).pull({ 0, -1 });

    ASSERT_EQ(t.get_row_count() * t.get_column_count(), rows_block.get_count());
    ASSERT_EQ(data, rows_block.get_data());

    for (std::int64_t i = 0; i < rows_block.get_count(); i++) {
        ASSERT_EQ(rows_block[i], data[i]);
    }
}

TEST(homogen_table_test, can_read_table_data_via_row_accessor_with_conversion) {
    using oneapi::dal::detail::empty_delete;

    float data[] = { 1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f };

    homogen_table t{ data, 2, 3, empty_delete<const float>() };
    auto rows_block = row_accessor<const double>(t).pull({ 0, -1 });

    ASSERT_EQ(t.get_row_count() * t.get_column_count(), rows_block.get_count());
    ASSERT_NE((void*)data, (void*)rows_block.get_data());

    for (std::int64_t i = 0; i < rows_block.get_count(); i++) {
        ASSERT_DOUBLE_EQ(rows_block[i], static_cast<double>(data[i]));
    }
}

TEST(homogen_table_test, can_read_table_data_via_row_accessor_and_array_outside) {
    using oneapi::dal::detail::empty_delete;

    float data[] = { 1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f };

    homogen_table t{ data, 2, 3, empty_delete<const float>() };
    auto arr = array<float>::empty(10);

    auto rows_ptr = row_accessor<const float>(t).pull(arr, { 0, -1 });

    ASSERT_EQ(t.get_row_count() * t.get_column_count(), arr.get_count());

    ASSERT_EQ(data, rows_ptr);
    ASSERT_EQ(data, arr.get_data());

    auto data_ptr = arr.get_data();
    for (std::int64_t i = 0; i < arr.get_count(); i++) {
        ASSERT_EQ(rows_ptr[i], data[i]);
        ASSERT_EQ(data_ptr[i], data[i]);
    }
}

TEST(homogen_table_test, can_read_rows_from_column_major_table) {
    float data[] = { 1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f };

    auto t = homogen_table::wrap(data, 3, 2, data_layout::column_major);

    auto rows_data = row_accessor<const float>(t).pull({ 1, -1 });

    ASSERT_EQ(rows_data.get_count(), 2 * t.get_column_count());

    ASSERT_FLOAT_EQ(rows_data[0], 2.0f);
    ASSERT_FLOAT_EQ(rows_data[1], -2.0f);
    ASSERT_FLOAT_EQ(rows_data[2], 3.0f);
    ASSERT_FLOAT_EQ(rows_data[3], -3.0f);
}

TEST(homogen_table_test, can_read_rows_from_column_major_table_with_conversion) {
    float data[] = { 1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f };

    auto t = homogen_table::wrap(data, 3, 2, data_layout::column_major);

    auto rows_data = row_accessor<const std::int32_t>(t).pull({ 1, 2 });

    ASSERT_EQ(rows_data.get_count(), 1 * t.get_column_count());

    ASSERT_EQ(rows_data[0], 2);
    ASSERT_EQ(rows_data[1], -2);
}
