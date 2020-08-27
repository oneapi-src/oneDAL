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

#include "oneapi/dal/table/column_accessor.hpp"
#include "gtest/gtest.h"
#include "oneapi/dal/table/homogen.hpp"

using namespace oneapi::dal;
using std::int32_t;

TEST(column_accessor_test, can_get_first_column_from_homogen_table) {
    float data[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f };

    homogen_table t{ data, 4, 2, empty_delete<const float>() };
    column_accessor<const float> acc{ t };
    auto col = acc.pull(0);

    ASSERT_EQ(col.get_count(), t.get_row_count());
    ASSERT_TRUE(col.has_mutable_data());

    for (std::int64_t i = 0; i < col.get_count(); i++) {
        ASSERT_FLOAT_EQ(col[i], t.get_data<float>()[i * t.get_column_count()]);
    }
}

TEST(column_accessor_test, can_get_second_column_from_homogen_table_with_conversion) {
    float data[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f };

    homogen_table t{ data, 4, 2, empty_delete<const float>() };
    column_accessor<const double> acc{ t };
    auto col = acc.pull(1);

    ASSERT_EQ(col.get_count(), t.get_row_count());
    ASSERT_TRUE(col.has_mutable_data());

    for (std::int64_t i = 0; i < col.get_count(); i++) {
        ASSERT_DOUBLE_EQ(col[i], double(t.get_data<float>()[i * t.get_column_count() + 1]));
    }
}

TEST(column_accessor_test, can_get_first_column_from_homogen_table_with_subset_of_rows) {
    float data[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f };

    homogen_table t{ data, 4, 2, empty_delete<const float>() };
    column_accessor<const float> acc{ t };
    auto col = acc.pull(0, { 1, 3 });

    ASSERT_EQ(col.get_count(), 2);
    ASSERT_TRUE(col.has_mutable_data());

    for (std::int64_t i = 0; i < col.get_count(); i++) {
        ASSERT_FLOAT_EQ(col[i], t.get_data<float>()[2 + i * t.get_column_count()]);
    }
}

TEST(column_accessor_test, can_get_columns_from_homogen_table_builder) {
    detail::homogen_table_builder b;
    b.reset(array<float>::zeros(3 * 2), 3, 2);
    {
        column_accessor<double> acc{ b };
        for (std::int64_t col_idx = 0; col_idx < 2; col_idx++) {
            auto col = acc.pull(col_idx);

            ASSERT_EQ(col.get_count(), 3);
            col.need_mutable_data();
            double* col_data = col.get_mutable_data();
            for (std::int64_t i = 0; i < col.get_count(); i++) {
                ASSERT_DOUBLE_EQ(col[i], 0.0);
                col_data[i] = col_idx + 1;
            }

            acc.push(col, col_idx);
        }
    }

    auto t = b.build();
    {
        column_accessor<const float> acc{ t };
        for (std::int64_t col_idx = 0; col_idx < 2; col_idx++) {
            const auto col = acc.pull(col_idx);

            ASSERT_EQ(col.get_count(), 3);
            for (std::int64_t i = 0; i < col.get_count(); i++) {
                ASSERT_FLOAT_EQ(col[i], col_idx + 1);
            }
        }
    }
}
