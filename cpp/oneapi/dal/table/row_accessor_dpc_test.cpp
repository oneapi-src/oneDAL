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

#define ONEDAL_DATA_PARALLEL
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/homogen.hpp"

using namespace oneapi::dal;
using std::int32_t;

TEST(row_accessor_dpc_test, can_get_rows_from_column_major_homogen_table) {
    sycl::queue q;
    constexpr std::int64_t row_count = 4;
    constexpr std::int64_t column_count = 3;
    constexpr std::int64_t data_size = row_count * column_count;

    auto data = sycl::malloc_shared<float>(data_size, q);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(data_size), [=](sycl::id<1> idx) {
            data[idx[0]] = idx[0];
        });
    }).wait();

    auto t =
        homogen_table::wrap(q, data, row_count, column_count, {}, data_layout::column_major);
    row_accessor<const float> acc{ t };
    auto block = acc.pull(q, { 1, 3 });

    ASSERT_EQ(block.get_count(), 2 * column_count);

    ASSERT_FLOAT_EQ(block[0], 1);
    ASSERT_FLOAT_EQ(block[1], 5);
    ASSERT_FLOAT_EQ(block[2], 9);

    ASSERT_FLOAT_EQ(block[3], 2);
    ASSERT_FLOAT_EQ(block[4], 6);
    ASSERT_FLOAT_EQ(block[5], 10);

    sycl::free(data, q);
}
