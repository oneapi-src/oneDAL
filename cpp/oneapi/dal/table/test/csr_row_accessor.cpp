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

#include "oneapi/dal/table/detail/csr_accessor.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal {

namespace te = dal::test::engine;

TEST("can read csr table via csr row accessor") {
    using oneapi::dal::detail::empty_delete;

    double data[] = { 1.0, 2.0, 3.0, 4.0, 1.0, 11.0, 8.0 };
    std::int64_t column_indices[] = { 1, 2, 4, 3, 2, 4, 2 };
    std::int64_t row_indices[] = { 1, 4, 5, 7, 8 };

    double data[] = { 1.0, 2.0, 3.0, -1.0, -2.0, -3.0 };

    homogen_table t{ data, 2, 3, empty_delete<const double>() };
    const auto rows_block = row_accessor<const double>(t).pull({ 0, -1 });

    REQUIRE(t.get_row_count() * t.get_column_count() == rows_block.get_count());
    REQUIRE(data == rows_block.get_data());

    for (std::int64_t i = 0; i < rows_block.get_count(); i++) {
        REQUIRE(rows_block[i] == data[i]);
    }
}

} // namespace oneapi::dal