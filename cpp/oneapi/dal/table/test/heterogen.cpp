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

TEST("can construct rowmajor table 3x2") {
    using oneapi::dal::detail::empty_delete;

    float data[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };

    homogen_table t{ data, 3, 2, empty_delete<const float>() };

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
    // TODO: now have no ctor to specify feature_type of the data
}

} // namespace oneapi::dal::test
