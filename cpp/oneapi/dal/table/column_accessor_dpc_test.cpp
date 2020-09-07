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

#define ONEAPI_DAL_DATA_PARALLEL
#include "oneapi/dal/table/column_accessor.hpp"
#include "oneapi/dal/table/homogen.hpp"

using namespace oneapi::dal;
using std::int32_t;

TEST(column_accessor_test, can_get_first_column_from_homogen_table) {
    sycl::queue q;
    constexpr std::int64_t data_size = 8;
    auto data = sycl::malloc_shared<float>(data_size, q);

    auto event = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(data_size), [=](sycl::id<1> idx) {
            data[idx[0]] = idx[0] + 1;
        });
    });

    homogen_table t{ q, data, 4, 2, make_default_delete<const float>(q), { event } };
    column_accessor<const float> acc{ t };
    auto col = acc.pull(q, 0);

    ASSERT_EQ(col.get_count(), t.get_row_count());
    ASSERT_TRUE(col.has_mutable_data());

    for (std::int64_t i = 0; i < col.get_count(); i++) {
        ASSERT_FLOAT_EQ(col[i], i * 2 + 1);
    }
}

TEST(column_accessor_test, can_get_second_column_from_homogen_table_with_conversion) {
    sycl::queue q;
    constexpr std::int64_t data_size = 8;
    auto data = sycl::malloc_shared<float>(data_size, q);

    auto event = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(data_size), [=](sycl::id<1> idx) {
            data[idx[0]] = idx[0] + 1;
        });
    });

    homogen_table t{ q, data, 4, 2, make_default_delete<const float>(q), { event } };
    column_accessor<const double> acc{ t };
    auto col = acc.pull(q, 1);

    ASSERT_EQ(col.get_count(), t.get_row_count());
    ASSERT_TRUE(col.has_mutable_data());

    for (std::int64_t i = 0; i < col.get_count(); i++) {
        ASSERT_DOUBLE_EQ(col[i], i * 2 + 2);
    }
}

TEST(column_accessor_test, can_get_first_column_from_homogen_table_with_subset_of_rows) {
    sycl::queue q;
    constexpr std::int64_t data_size = 8;
    auto data = sycl::malloc_shared<float>(data_size, q);

    auto event = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(data_size), [=](sycl::id<1> idx) {
            data[idx[0]] = idx[0] + 1;
        });
    });

    homogen_table t{ q, data, 4, 2, make_default_delete<const float>(q), { event } };
    column_accessor<const float> acc{ t };
    auto col = acc.pull(q, 0, { 1, 3 });

    ASSERT_EQ(col.get_count(), 2);
    ASSERT_TRUE(col.has_mutable_data());

    for (std::int64_t i = 0; i < col.get_count(); i++) {
        ASSERT_FLOAT_EQ(col[i], t.get_data<float>()[2 + i * t.get_column_count()]);
    }
}

TEST(column_accessor_test, can_get_columns_from_homogen_table_builder) {
    sycl::queue q;

    detail::homogen_table_builder b;
    b.reset(array<float>::zeros(q, 3 * 2), 3, 2);
    {
        column_accessor<double> acc{ b };
        for (std::int64_t col_idx = 0; col_idx < 2; col_idx++) {
            auto col = acc.pull(q, col_idx);

            ASSERT_EQ(col.get_count(), 3);
            col.need_mutable_data(q);
            for (std::int64_t i = 0; i < col.get_count(); i++) {
                ASSERT_DOUBLE_EQ(col[i], 0.0);
                col[i] = col_idx + 1;
            }

            acc.push(q, col, col_idx);
        }
    }

    auto t = b.build();
    {
        column_accessor<const float> acc{ t };
        for (std::int64_t col_idx = 0; col_idx < 2; col_idx++) {
            const auto col = acc.pull(q, col_idx);

            ASSERT_EQ(col.get_count(), 3);
            for (std::int64_t i = 0; i < col.get_count(); i++) {
                ASSERT_FLOAT_EQ(col[i], col_idx + 1);
            }
        }
    }
}
