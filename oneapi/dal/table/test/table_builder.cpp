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

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

using namespace oneapi::dal;

// TODO: need to implement move constructor from table in table_builder class

// TEST(table_builder_test, can_modify_table) {
//     float data[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };

//     float data2[] = { 1.f, 2.f, 3.f, 4.f, -5.f, -6.f };

//     table t = homogen_table{ 3, 2, data };

//     {
//         table_builder b{ std::move(t) };
//         row_accessor<float> acc{ b };

//         auto arr = acc.pull({ 2, -1 });
//         ASSERT_EQ(arr.get_count(), 2);
//         arr.need_mutable_data();
//         arr[0] = data2[2 * 2];
//         arr[1] = data2[2 * 2 + 1];
//         acc.push(arr, { 2, -1 });

//         t = b.build();
//     }

//     row_accessor<const float> acc{ t };
//     auto arr = acc.pull();

//     for (std::int64_t i = 0; i < arr.get_count(); i++) {
//         ASSERT_FLOAT_EQ(arr.get_data()[i], data2[i]);
//     }
// }

TEST("can construct table") {
    float data[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };

    array<float> arr{ data, 3 * 2, [](auto) {} };
    homogen_table t = detail::homogen_table_builder{}.reset(arr, 3, 2).build();

    REQUIRE(t.has_data());
    REQUIRE(t.get_row_count() == 3);
    REQUIRE(t.get_column_count() == 2);
    REQUIRE(t.get_data_layout() == data_layout::row_major);

    auto meta = t.get_metadata();
    for (std::int64_t i = 0; i < t.get_column_count(); i++) {
        REQUIRE(meta.get_data_type(i) == data_type::float32);
        REQUIRE(meta.get_feature_type(i) == feature_type::ratio);
    }

    REQUIRE(t.get_data<float>() == data);
}
