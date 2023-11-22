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

#include <memory>
#include <iostream>

#include "example_util/utils.hpp"

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/homogen.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace dal = oneapi::dal;

template <typename Type = float>
dal::table get_table(std::int64_t row_count, std::int64_t column_count) {
    const std::int64_t elem_count = row_count * column_count;
    auto* const raw_data = new Type[elem_count];

    // Create an array using raw pointer and delete[ ]
    auto data = dal::array<Type>(raw_data,
                                 elem_count, //
                                 [](Type* const ptr) -> void {
                                     delete[] ptr;
                                 });

    // Fill array with structured data
    for (std::int64_t row = 0l; row < row_count; ++row) {
        for (std::int64_t col = 0l; col < column_count; ++col) {
            const std::int64_t idx = row * column_count + col;
            raw_data[idx] = static_cast<Type>(row * col);
        }
    }

    return dal::homogen_table::wrap(data, row_count, column_count);
}

int main(int argc, char** argv) {
    constexpr std::int64_t row_count = 4;
    constexpr std::int64_t column_count = 3;

    // Generate table on host
    const dal::table test_table = get_table(row_count, column_count);

    // Some sanity checks for the table shape
    std::cout << "Number of rows in table: " << test_table.get_row_count() << '\n';
    std::cout << "Number of columns in table: " << test_table.get_column_count() << '\n';

    // Check the type of abstract table
    const bool is_homogen = test_table.get_kind() == dal::homogen_table::kind();
    std::cout << "Is homogeneous table: " << is_homogen << '\n';

    // Extracting row slice of data on host
    dal::row_accessor<const double> accessor{ test_table };
    dal::array<double> slice = accessor.pull({ 1l, 3l });

    std::cout << "Slice of elements: " << slice << std::endl;

    return 0;
}
