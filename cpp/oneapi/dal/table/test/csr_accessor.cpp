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

#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/table/csr_accessor.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal {

TEST("can read csr table via csr accessor") {
    using oneapi::dal::detail::empty_delete;

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
    const auto [data_array, cidx_array, ridx_array] = csr_accessor<const double>(t).pull({ 0, -1 });

    REQUIRE(data == data_array.get_data());
    REQUIRE(column_indices == cidx_array.get_data());
    REQUIRE(row_offsets == ridx_array.get_data());

    for (std::int64_t i = 0; i < data_array.get_count(); i++) {
        REQUIRE(data_array[i] == data[i]);
        REQUIRE(cidx_array[i] == column_indices[i]);
    }
    for (std::int64_t i = 0; i < ridx_array.get_count(); i++) {
        REQUIRE(ridx_array[i] == row_offsets[i]);
    }
}

TEST("can read a portion of csr table via csr accessor and create a smaller data block") {
    using oneapi::dal::detail::empty_delete;

    double data[] = { 1.0, 2.0, 3.0, 4.0, 1.0, 11.0, 8.0 };
    std::int64_t column_indices[] = { 1, 2, 4, 3, 2, 4, 2 };
    std::int64_t row_offsets[] = { 1, 4, 5, 7, 8 };

    constexpr std::int64_t row_count{ 4L };
    constexpr std::int64_t column_count{ 4L };
    constexpr std::int64_t element_count{ 7L };

    csr_table t{ array<double>::wrap(data, element_count),
                 array<std::int64_t>::wrap(column_indices, element_count),
                 array<std::int64_t>::wrap(row_offsets, row_count + 1),
                 column_count };
    constexpr std::int64_t start_row{ 1L };
    constexpr std::int64_t end_row{ 3L };
    const auto [data_array, cidx_array, ridx_array] =
        csr_accessor<const double>(t).pull({ start_row, end_row });

    const std::int64_t data_shift = row_offsets[start_row] - row_offsets[0];

    REQUIRE(data_array.get_count() == row_offsets[end_row] - row_offsets[start_row]);
    REQUIRE(data_array.get_count() == cidx_array.get_count());
    REQUIRE(ridx_array.get_count() == end_row - start_row + 1);

    REQUIRE(data + data_shift == data_array.get_data());
    REQUIRE(column_indices + data_shift == cidx_array.get_data());
    REQUIRE(row_offsets + start_row != ridx_array.get_data());

    for (std::int64_t i = 0; i < data_array.get_count(); i++) {
        REQUIRE(data_array[i] == data[data_shift + i]);
        REQUIRE(cidx_array[i] == column_indices[data_shift + i]);
    }
    for (std::int64_t i = 0; i < ridx_array.get_count(); i++) {
        REQUIRE(ridx_array[i] == row_offsets[start_row + i] - data_shift);
    }
}

TEST("can read csr table via csr accessor with conversion") {
    using oneapi::dal::detail::empty_delete;

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
    const auto [data_array, cidx_array, ridx_array] = csr_accessor<const double>(t).pull({ 0, -1 });

    REQUIRE((void*)data != (void*)data_array.get_data());
    REQUIRE(column_indices == cidx_array.get_data());
    REQUIRE(row_offsets == ridx_array.get_data());

    for (std::int64_t i = 0; i < data_array.get_count(); i++) {
        REQUIRE(data_array[i] == Approx(static_cast<double>(data[i])));
        REQUIRE(cidx_array[i] == column_indices[i]);
    }
    for (std::int64_t i = 0; i < ridx_array.get_count(); i++) {
        REQUIRE(ridx_array[i] == row_offsets[i]);
    }
}

TEST("can read zero-based data from a zero-based csr table via csr accessor") {
    using oneapi::dal::detail::empty_delete;

    double data[] = { 1.0, 2.0, 3.0, 4.0, 1.0, 11.0, 8.0 };
    std::int64_t column_indices[] = { 0, 1, 3, 2, 1, 3, 1 };
    std::int64_t row_offsets[] = { 0, 3, 4, 6, 7 };

    const std::int64_t row_count{ 4 };
    const std::int64_t column_count{ 4 };
    const std::int64_t element_count{ 7 };

    csr_table t{ array<double>::wrap(data, element_count),
                 array<std::int64_t>::wrap(column_indices, element_count),
                 array<std::int64_t>::wrap(row_offsets, row_count + 1),
                 column_count,
                 sparse_indexing::zero_based };
    const auto [data_array, cidx_array, ridx_array] =
        csr_accessor<const double>(t).pull({ 0, -1 }, sparse_indexing::zero_based);

    REQUIRE(data == data_array.get_data());
    REQUIRE(column_indices == cidx_array.get_data());
    REQUIRE(row_offsets == ridx_array.get_data());

    for (std::int64_t i = 0; i < data_array.get_count(); i++) {
        REQUIRE(data_array[i] == data[i]);
        REQUIRE(cidx_array[i] == column_indices[i]);
    }
    for (std::int64_t i = 0; i < ridx_array.get_count(); i++) {
        REQUIRE(ridx_array[i] == row_offsets[i]);
    }
}

TEST("can read one-based data from a zero-based csr table via csr accessor") {
    using oneapi::dal::detail::empty_delete;

    double data[] = { 1.0, 2.0, 3.0, 4.0, 1.0, 11.0, 8.0 };
    std::int64_t column_indices[] = { 0, 1, 3, 2, 1, 3, 1 };
    std::int64_t row_offsets[] = { 0, 3, 4, 6, 7 };

    const std::int64_t row_count{ 4 };
    const std::int64_t column_count{ 4 };
    const std::int64_t element_count{ 7 };

    csr_table t{ array<double>::wrap(data, element_count),
                 array<std::int64_t>::wrap(column_indices, element_count),
                 array<std::int64_t>::wrap(row_offsets, row_count + 1),
                 column_count,
                 sparse_indexing::zero_based };

    const auto [data_array, cidx_array, ridx_array] = csr_accessor<const double>(t).pull({ 0, -1 });

    REQUIRE(data == data_array.get_data());

    for (std::int64_t i = 0; i < data_array.get_count(); i++) {
        REQUIRE(data_array[i] == data[i]);
        REQUIRE(cidx_array[i] == column_indices[i] + 1);
    }

    for (std::int64_t i = 0; i < ridx_array.get_count(); i++) {
        REQUIRE(ridx_array[i] == row_offsets[i] + 1);
    }
}

} // namespace oneapi::dal
