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

#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/test/engine/common.hpp"


namespace oneapi::dal::test {

TEST("can construct empty table") {
    csr_table t;

    REQUIRE(t.has_data() == false);
    REQUIRE(t.get_kind() == csr_table::kind());
    REQUIRE(t.get_row_count() == 0);
    REQUIRE(t.get_column_count() == 0);
}

TEST("can construct CSR table from raw data pointers") {
    using oneapi::dal::detail::empty_delete;

    float data[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    std::int64_t column_indices[] = { 1, 2, 4, 3, 2, 4, 2 };
    std::int64_t row_offsets[] = { 1, 4, 5, 7, 8 };

    const std::int64_t row_count{ 4 };
    const std::int64_t column_count{ 4 };
    const std::int64_t element_count{ 7 };

    csr_table t{ data, column_indices, row_offsets, row_count, column_count,
        empty_delete<const float>(), empty_delete<const std::int64_t>(), empty_delete<const std::int64_t>() };

    REQUIRE(t.has_data());
    REQUIRE(t.get_row_count() == row_count);
    REQUIRE(t.get_column_count() == column_count);
    REQUIRE(t.get_non_zero_count() == element_count);

    REQUIRE(t.get_indexing() == sparse_indexing::one_based);

    auto meta = t.get_metadata();
    for (std::int64_t i = 0; i < t.get_column_count(); i++) {
        REQUIRE(meta.get_data_type(i) == data_type::float32);
        REQUIRE(meta.get_feature_type(i) == feature_type::ratio);
    }

    REQUIRE(t.get_data<float>() == data);
    REQUIRE(t.get_column_indices() == column_indices);
    REQUIRE(t.get_row_offsets() == row_offsets);
}

} // namespace oneapi::dal::test
