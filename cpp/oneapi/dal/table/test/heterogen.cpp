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

#include "oneapi/dal/chunked_array.hpp"

#include "oneapi/dal/table/heterogen.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::test {

TEST("can construct empty table") {
    heterogen_table t;

    REQUIRE(t.has_data() == false);
    REQUIRE(t.get_row_count() == 0);
    REQUIRE(t.get_column_count() == 0);
    REQUIRE(t.get_kind() == heterogen_table::kind());
}

TEST("Can create table from chunked arrays") {
    constexpr float src1[] = { 4.f, 5.f };
    constexpr float src2[] = { 1.f, 2.f, 3.f };

    auto arr1 = array<float>::wrap(src1, 2l);
    auto arr2 = array<float>::wrap(src2, 3l);

    chunked_array<float> chunked1(2);
    chunked1.set_chunk(0l, arr1);
    chunked1.set_chunk(1l, arr2);
    ONEDAL_ASSERT(chunked1.get_count() == 5l);

    chunked_array<float> chunked2(2);
    chunked2.set_chunk(0l, arr2);
    chunked2.set_chunk(1l, arr1);
    ONEDAL_ASSERT(chunked2.get_count() == 5l);

{
    auto table = heterogen_table::wrap( //
        chunked1, chunked2, chunked1, chunked2);
//
    REQUIRE(table.has_data() == true);
    REQUIRE(table.get_row_count() == 5l);
    REQUIRE(table.get_column_count() == 4l);
    REQUIRE(table.get_kind() == heterogen_table::kind());
}
}

TEST("Can create table from different chunked arrays") {
    constexpr float src1[] = { 4.f, 5.f };
    constexpr float src2[] = { 1.f, 2.f, 3.f };

    auto arr1 = array<float>::wrap(src1, 2l);
    auto arr2 = array<float>::wrap(src2, 3l);

    chunked_array<float> chunked1(2);
    chunked1.set_chunk(0l, arr1);
    chunked1.set_chunk(1l, arr2);
    ONEDAL_ASSERT(chunked1.get_count() == 5l);

    constexpr std::int8_t src3[] = { 4, 5 };
    constexpr std::int8_t src4[] = { 1, 2, 3 };

    auto arr3 = array<std::int8_t>::wrap(src3, 2l);
    auto arr4 = array<std::int8_t>::wrap(src4, 3l);

    chunked_array<std::int8_t> chunked2(2);
    chunked2.set_chunk(0l, arr4);
    chunked2.set_chunk(1l, arr3);
    ONEDAL_ASSERT(chunked2.get_count() == 5l);

    auto table = heterogen_table::wrap( //
            chunked1, chunked1, chunked2);
//
    REQUIRE(table.has_data() == true);
    REQUIRE(table.get_row_count() == 5l);
    REQUIRE(table.get_column_count() == 3l);
    REQUIRE(table.get_kind() == heterogen_table::kind());
}

} // namespace oneapi::dal::test
