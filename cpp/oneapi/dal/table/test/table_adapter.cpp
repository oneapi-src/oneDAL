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

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

using namespace oneapi::dal;

TEST("Homogen adapter is used") {
    const float data[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };

    const auto t = homogen_table::wrap(data, 3, 2, data_layout::row_major);
    auto dt = backend::interop::convert_to_daal_table<float>(t);

    REQUIRE(dynamic_cast<backend::interop::host_homogen_table_adapter<float>*>(dt.get()) !=
            nullptr);
}

TEST("SOA adapter is used") {
    const float data[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };

    const auto t = homogen_table::wrap(data, 3, 2, data_layout::column_major);
    auto dt = backend::interop::convert_to_daal_table<float>(t);

    REQUIRE(dynamic_cast<backend::interop::host_soa_table_adapter*>(dt.get()) != nullptr);
}

TEST("Can create SOA table adapter") {
    constexpr float src1[] = { 1.f, 2.f };
    constexpr float src2[] = { 3.f, 4.f, 5.f };

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

    auto table = heterogen_table::wrap( //
        chunked1,
        chunked2);

    auto dt = backend::interop::convert_to_daal_table(table);
    REQUIRE(dynamic_cast<backend::interop::host_heterogen_table_adapter*>(dt.get()) != nullptr);
}

TEST("CSR adapter is used, one-based indexing") {
    const float data[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    const std::int64_t column_indices[] = { 1, 2, 4, 3, 2, 4, 2 };
    const std::int64_t row_offsets[] = { 1, 4, 5, 7, 8 };

    constexpr std::int64_t row_count{ 4 };
    constexpr std::int64_t column_count{ 4 };

    csr_table t = csr_table::wrap(data, column_indices, row_offsets, row_count, column_count);

    auto dt = backend::interop::convert_to_daal_table<float>(t);

    REQUIRE(dynamic_cast<backend::interop::host_csr_table_adapter<float>*>(dt.get()) != nullptr);
}

TEST("CSR adapter is used, zero-based indexing") {
    const float data[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    const std::int64_t column_indices[] = { 0, 1, 3, 2, 1, 3, 1 };
    const std::int64_t row_offsets[] = { 0, 3, 4, 6, 7 };

    constexpr std::int64_t row_count{ 4 };
    constexpr std::int64_t column_count{ 4 };

    csr_table t = csr_table::wrap(data,
                                  column_indices,
                                  row_offsets,
                                  row_count,
                                  column_count,
                                  sparse_indexing::zero_based);

    auto dt = backend::interop::convert_to_daal_table<float>(t);

    REQUIRE(dynamic_cast<backend::interop::host_csr_table_adapter<float>*>(dt.get()) != nullptr);
}
