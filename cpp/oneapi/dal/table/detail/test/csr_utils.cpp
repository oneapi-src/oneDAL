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

#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::detail {

TEST("can get original data from CSR table constructed from builder") {
    constexpr std::int64_t row_count{ 4 };
    constexpr std::int64_t column_count{ 4 };
    constexpr std::int64_t element_count{ 7 };
    constexpr sparse_indexing indexing = sparse_indexing::one_based;

    const std::int64_t column_indices_buffer[] = { 1, 2, 4, 3, 2, 4, 2 };
    const std::int64_t row_offsets_buffer[] = { 1, 4, 5, 7, 8 };

    auto data = array<float>::full(element_count, 1.0f);
    auto column_indices = array<std::int64_t>::wrap(column_indices_buffer, element_count);
    auto row_offsets = array<std::int64_t>::wrap(row_offsets_buffer, row_count + 1);

    auto t = csr_table_builder{}
                 .reset(data, column_indices, row_offsets, row_count, column_count, indexing)
                 .build();

    REQUIRE(t.get_data() == reinterpret_cast<const byte_t*>(data.get_data()));
    REQUIRE(t.get_column_indices() == column_indices_buffer);
    REQUIRE(t.get_row_offsets() == row_offsets_buffer);
    REQUIRE(t.get_non_zero_count() == element_count);
    REQUIRE(t.get_indexing() == indexing);
}

} // namespace oneapi::dal::detail
