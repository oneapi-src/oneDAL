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
#include "oneapi/dal/table/detail/csr.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal {

namespace te = dal::test::engine;

TEST("can read CSR table via CSR accessor") {
    using oneapi::dal::detail::empty_delete;

    double data[] = { 1.0, 2.0, 3.0, 4.0, 1.0, 11.0, 8.0 };
    std::int64_t column_indices[] = { 1, 2, 4, 3, 2, 4, 2 };
    std::int64_t row_indices[] = { 1, 4, 5, 7, 8 };

    const std::int64_t row_count{ 4 };
    const std::int64_t column_count{ 4 };
    const std::int64_t element_count{ 7 };

    detail::csr_table t{ array<double>::wrap(data, element_count),
                         array<std::int64_t>::wrap(column_indices, element_count),
                         array<std::int64_t>::wrap(row_indices, row_count + 1),
                         column_count };
    const auto [data_array, cidx_array, ridx_array] =
        csr_row_accessor<const double>(t).pull({ 0, -1 });

    REQUIRE(data == data_array.get_data());
    REQUIRE(column_indices == cidx_array.get_data());
    REQUIRE(row_indices == ridx_array.get_data());

    for (std::int64_t i = 0; i < data_array.get_count(); i++) {
        REQUIRE(data_array[i] == data[i]);
        REQUIRE(cidx_array[i] == column_indices[i]);
    }
    for (std::int64_t i = 0; i < ridx_array.get_count(); i++) {
        REQUIRE(ridx_array[i] == row_indices[i]);
    }
}

} // namespace oneapi::dal