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

#include <random>
#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/table/csr_accessor.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/serialization.hpp"

namespace oneapi::dal::test {

namespace te = dal::test::engine;
namespace de = dal::detail;

template <typename Data>
void check_csr_tables_equal(const csr_table& a, const csr_table& b) {
    REQUIRE(a.get_column_count() == b.get_column_count());
    REQUIRE(a.get_row_count() == b.get_row_count());

    if (a.get_column_count() == 0 || a.get_row_count() == 0) {
        return;
    }

    auto [a_data_ptr, a_col_ptr, a_row_ptr] =
        csr_accessor<const Data>(a).pull({ 0, -1 }, a.get_indexing());
    auto [b_data_ptr, b_col_ptr, b_row_ptr] =
        csr_accessor<const Data>(b).pull({ 0, -1 }, b.get_indexing());

    for (std::uint32_t i = 0; i < a_data_ptr.get_count(); ++i) {
        REQUIRE(a_data_ptr[i] == b_data_ptr[i]);
        REQUIRE(a_col_ptr[i] == b_col_ptr[i]);
    }
    for (std::uint32_t i = 0; i < a_row_ptr.get_count(); ++i) {
        REQUIRE(a_row_ptr[i] == b_row_ptr[i]);
    }
}

TEST("Serialize/deserialize empty CSR table") {
    csr_table init_table;
    auto deser_table = te::serialize_deserialize(init_table);

    check_csr_tables_equal<const float>(init_table, deser_table);
}

TEST("Serialize/deserialize small CSR table one-based indexing") {
    using oneapi::dal::detail::empty_delete;

    float data[] = { 1.05f, 1.75f, 4.6f, 0.05f, 3.14f, 4.0f, 1.0f, 11.89f, 2.7f };
    std::int64_t column_indices[] = { 1, 3, 8, 2, 4, 4, 2, 4, 8 };
    std::int64_t row_indices[] = { 1, 4, 6, 8, 9, 9, 9, 9, 10 };

    const std::int64_t row_count{ 8 };
    const std::int64_t column_count{ 8 };
    const std::int64_t element_count{ 9 };

    csr_table init_table{ array<float>::wrap(data, element_count),
                          array<std::int64_t>::wrap(column_indices, element_count),
                          array<std::int64_t>::wrap(row_indices, row_count + 1),
                          column_count };
    auto deser_table = te::serialize_deserialize(init_table);

    check_csr_tables_equal<const float>(init_table, deser_table);
}

TEST("Serialize/deserialize small CSR table zero-based indexing") {
    using oneapi::dal::detail::empty_delete;

    float data[] = { 1.05f, 1.75f, 4.6f, 0.05f, 3.14f, 4.0f, 1.0f, 11.89f, 2.7f };
    std::int64_t column_indices[] = { 0, 2, 7, 1, 3, 3, 1, 3, 7 };
    std::int64_t row_indices[] = { 0, 3, 5, 7, 8, 8, 8, 8, 9 };

    const std::int64_t row_count{ 8 };
    const std::int64_t column_count{ 8 };
    const std::int64_t element_count{ 9 };

    csr_table init_table{ array<float>::wrap(data, element_count),
                          array<std::int64_t>::wrap(column_indices, element_count),
                          array<std::int64_t>::wrap(row_indices, row_count + 1),
                          column_count,
                          sparse_indexing::zero_based };
    auto deser_table = te::serialize_deserialize(init_table);

    check_csr_tables_equal<const float>(init_table, deser_table);
}

TEST("Serialize/deserialize big random CSR table one-based indexing") {
    std::uint32_t seed = 42;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform_data(-3.0f, 3.0f);
    std::uniform_int_distribution<std::uint32_t> uniform_columns(1, 32);

    std::uint32_t row_count = 128;
    std::uint32_t column_count = 32;
    std::uint32_t nonzero_count = 128; // one non-zero value in one row
    dal::array<float> data = dal::array<float>::empty(nonzero_count);
    auto data_ptr = data.get_mutable_data();
    dal::array<std::int64_t> col_indices = dal::array<std::int64_t>::empty(nonzero_count);
    auto col_indices_ptr = col_indices.get_mutable_data();
    dal::array<std::int64_t> row_offsets = dal::array<std::int64_t>::empty(row_count + 1);
    auto row_offsets_ptr = row_offsets.get_mutable_data();

    for (std::uint32_t i = 0; i < nonzero_count; ++i) {
        data_ptr[i] = uniform_data(rng);
        col_indices_ptr[i] = uniform_columns(rng);
    }

    for (std::uint32_t i = 0; i <= row_count; ++i) {
        row_offsets_ptr[i] = i + 1;
    }

    auto init_table =
        csr_table::wrap(data, col_indices, row_offsets, column_count, sparse_indexing::one_based);

    auto deser_table = te::serialize_deserialize(init_table);

    check_csr_tables_equal<float>(init_table, deser_table);
}

TEST("Serialize/deserialize big random CSR table zero-based indexing") {
    std::uint32_t seed = 42;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform_data(-3.0f, 3.0f);
    std::uniform_int_distribution<std::uint32_t> uniform_columns(0, 31);

    std::uint32_t row_count = 128;
    std::uint32_t column_count = 32;
    std::uint32_t nonzero_count = 128; // one non-zero value in one row
    dal::array<float> data = dal::array<float>::empty(nonzero_count);
    auto data_ptr = data.get_mutable_data();
    dal::array<std::int64_t> col_indices = dal::array<std::int64_t>::empty(nonzero_count);
    auto col_indices_ptr = col_indices.get_mutable_data();
    dal::array<std::int64_t> row_offsets = dal::array<std::int64_t>::empty(row_count + 1);
    auto row_offsets_ptr = row_offsets.get_mutable_data();

    for (std::uint32_t i = 0; i < nonzero_count; ++i) {
        data_ptr[i] = uniform_data(rng);
        col_indices_ptr[i] = uniform_columns(rng);
    }

    for (std::uint32_t i = 0; i <= row_count; ++i) {
        row_offsets_ptr[i] = i;
    }

    auto init_table =
        csr_table::wrap(data, col_indices, row_offsets, column_count, sparse_indexing::zero_based);

    auto deser_table = te::serialize_deserialize(init_table);

    check_csr_tables_equal<float>(init_table, deser_table);
}

} // namespace oneapi::dal::test
