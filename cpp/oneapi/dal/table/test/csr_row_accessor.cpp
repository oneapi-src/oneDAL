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

#include "oneapi/dal/table/csr_row_accessor.hpp"
#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal {

namespace te = dal::test::engine;

TEST("can read CSR table via CSR accessor") {
    using oneapi::dal::detail::empty_delete;

    /// Input matrix in dense format:
    ///     1.0     2.0       0     3.0
    ///       0       0     4.0       0
    ///       0     1.0       0    11.0
    ///       0     8.0       0       0

    double data[] = { 1.0, 2.0, 3.0, 4.0, 1.0, 11.0, 8.0 };
    std::int64_t column_indices[] = { 1, 2, 4, 3, 2, 4, 2 };
    std::int64_t row_offsets[] = { 1, 4, 5, 7, 8 };

    const std::int64_t row_count{ 4 };
    const std::int64_t column_count{ 4 };
    const std::int64_t element_count{ 7 };

    csr_table t{ array<double>::wrap(data, element_count),
                 array<std::int64_t>::wrap(column_indices, element_count),
                 array<std::int64_t>::wrap(row_offsets, row_count + 1),
                 column_count };
    const auto [data_array, cidx_array, roffs_array] =
        csr_row_accessor<const double>(t).pull({ 0, -1 });

    REQUIRE(data == data_array.get_data());
    REQUIRE(column_indices == cidx_array.get_data());
    REQUIRE(row_offsets == roffs_array.get_data());

    for (std::int64_t i = 0; i < data_array.get_count(); i++) {
        REQUIRE(data_array[i] == data[i]);
        REQUIRE(cidx_array[i] == column_indices[i]);
    }
    for (std::int64_t i = 0; i < roffs_array.get_count(); i++) {
        REQUIRE(roffs_array[i] == row_offsets[i]);
    }
}

TEST("can read CSR table via CSR accessor and create smaller block") {
    using oneapi::dal::detail::empty_delete;

    /// Input matrix in dense format:
    ///     1.0     2.0       0     3.0
    ///       0       0     4.0       0
    ///       0     1.0       0    11.0
    ///       0     8.0       0       0

    double data[] = { 1.0, 2.0, 3.0, 4.0, 1.0, 11.0, 8.0 };
    std::int64_t column_indices[] = { 1, 2, 4, 3, 2, 4, 2 };
    std::int64_t row_offsets[] = { 1, 4, 5, 7, 8 };

    const std::int64_t row_count{ 4 };
    const std::int64_t column_count{ 4 };
    const std::int64_t element_count{ 7 };

    csr_table t{ array<double>::wrap(data, element_count),
                 array<std::int64_t>::wrap(column_indices, element_count),
                 array<std::int64_t>::wrap(row_offsets, row_count + 1),
                 column_count };
    const std::int64_t start_idx = 1;
    const std::int64_t end_idx = 3;
    const auto [data_array, cidx_array, roffs_array] =
        csr_row_accessor<const double>(t).pull({ start_idx, end_idx });

    std::int64_t element_count_before_start = row_offsets[start_idx] - row_offsets[0];
    REQUIRE(data + element_count_before_start == data_array.get_data());
    REQUIRE(column_indices + element_count_before_start == cidx_array.get_data());
    REQUIRE(row_offsets + start_idx != roffs_array.get_data());

    for (std::int64_t i = 0; i < data_array.get_count(); i++) {
        REQUIRE(data_array[i] == data[element_count_before_start + i]);
        REQUIRE(cidx_array[i] == column_indices[element_count_before_start + i]);
    }
    for (std::int64_t i = 0; i < roffs_array.get_count(); i++) {
        REQUIRE(roffs_array[i] == row_offsets[start_idx + i] - element_count_before_start);
    }
}

TEST("can read CSR table via CSR accessor with conversion") {
    using oneapi::dal::detail::empty_delete;

    /// Input matrix in dense format:
    ///     1.0     2.0       0     3.0
    ///       0       0     4.0       0
    ///       0     1.0       0    11.0
    ///       0     8.0       0       0

    float data[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    std::int64_t column_indices[] = { 1, 2, 4, 3, 2, 4, 2 };
    std::int64_t row_offsets[] = { 1, 4, 5, 7, 8 };

    const std::int64_t row_count{ 4 };
    const std::int64_t column_count{ 4 };
    const std::int64_t element_count{ 7 };

    csr_table t{ array<float>::wrap(data, element_count),
                 array<std::int64_t>::wrap(column_indices, element_count),
                 array<std::int64_t>::wrap(row_offsets, row_count + 1),
                 column_count };
    const auto [data_array, cidx_array, roffs_array] =
        csr_row_accessor<const double>(t).pull({ 0, -1 });

    REQUIRE((void*)data != (void*)data_array.get_data());
    REQUIRE(column_indices == cidx_array.get_data());
    REQUIRE(row_offsets == roffs_array.get_data());

    for (std::int64_t i = 0; i < data_array.get_count(); i++) {
        REQUIRE(data_array[i] == Approx(static_cast<double>(data[i])));
        REQUIRE(cidx_array[i] == column_indices[i]);
    }
    for (std::int64_t i = 0; i < roffs_array.get_count(); i++) {
        REQUIRE(roffs_array[i] == row_offsets[i]);
    }
}

} // namespace oneapi::dal
