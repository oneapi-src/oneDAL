/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#pragma once

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::test::engine {

inline void check_if_metadata_equal(const table_metadata& actual, const table_metadata& reference) {
    REQUIRE(actual.get_feature_count() == reference.get_feature_count());
    for (std::int64_t i = 0; i < reference.get_feature_count(); i++) {
        REQUIRE(actual.get_feature_type(i) == reference.get_feature_type(i));
        REQUIRE(actual.get_data_type(i) == reference.get_data_type(i));
    }
}

template <typename Data>
inline void check_if_table_content_equal(const table& actual, const table& reference) {
    const auto actual_ary = row_accessor<const Data>{ actual }.pull();
    const auto reference_ary = row_accessor<const Data>{ reference }.pull();

    for (std::int64_t i = 0; i < reference_ary.get_count(); i++) {
        if (actual_ary[i] != reference_ary[i]) {
            CAPTURE(i, actual_ary[i], reference_ary[i]);
            FAIL("Found elements mismatch in tables");
            break;
        }
    }
}

template <typename Data>
inline void check_if_tables_equal(const table& actual, const table& reference) {
    REQUIRE(actual.get_row_count() == reference.get_row_count());
    REQUIRE(actual.get_column_count() == reference.get_column_count());
    REQUIRE(actual.get_data_layout() == reference.get_data_layout());
    REQUIRE(actual.get_kind() == reference.get_kind());

    check_if_metadata_equal(actual.get_metadata(), reference.get_metadata());
    check_if_table_content_equal<Data>(actual, reference);
}

} // namespace oneapi::dal::test::engine
