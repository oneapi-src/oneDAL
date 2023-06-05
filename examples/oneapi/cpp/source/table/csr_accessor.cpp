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

#include <iostream>

#include "oneapi/dal/table/csr_accessor.hpp"
#include "oneapi/dal/table/csr.hpp"

namespace dal = oneapi::dal;

int main(int argc, char const *argv[]) {
    constexpr std::int64_t row_count = 4;
    constexpr std::int64_t column_count = 4;
    constexpr std::int64_t element_count = 7;
    const float data[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    const std::int64_t column_indices[] = { 1, 2, 4, 3, 2, 4, 2 };
    const std::int64_t row_offsets[] = { 1, 4, 5, 7, 8 };

    // create sparse table in CSR format from arrays of data, column indices and row offsets
    auto table = dal::csr_table::wrap(data, column_indices, row_offsets, row_count, column_count);
    dal::csr_accessor<const float> acc{ table };

    // pull 2 rows, starting from row number 1, from the sparse table,
    // resulting row offsets are re-calculated to have the first offset equal to 1
    // in case of one-based indexing
    const auto [subtable_data, subtable_column_indices, subtable_row_offsets] = acc.pull({ 1, 3 });

    std::cout << "Print the original sparse data table as 3 arrays in CSR storage format:"
              << std::endl;
    std::cout << "Values of the table:" << std::endl;
    for (std::int64_t i = 0; i < element_count; i++) {
        std::cout << data[i] << ", ";
    }
    std::cout << std::endl << "Column indices of the table:" << std::endl;
    for (std::int64_t i = 0; i < element_count; i++) {
        std::cout << column_indices[i] << ", ";
    }
    std::cout << std::endl << "Row offsets of the table:" << std::endl;
    for (std::int64_t i = 0; i < row_count + 1; i++) {
        std::cout << row_offsets[i] << ", ";
    }
    std::cout << std::endl;

    std::cout << std::endl << "Print 2 rows from CSR table as dense float arrays" << std::endl;
    std::cout << "Values in 2 rows as dense float array:" << std::endl;
    for (std::int64_t i = 0; i < subtable_data.get_count(); i++) {
        std::cout << subtable_data[i] << ", ";
    }
    std::cout << std::endl << "Column indices in 2 rows from CSR table:" << std::endl;
    for (std::int64_t i = 0; i < subtable_column_indices.get_count(); i++) {
        std::cout << subtable_column_indices[i] << ", ";
    }
    std::cout << std::endl << "Row offsets in 2 rows from CSR table:" << std::endl;
    for (std::int64_t i = 0; i < subtable_row_offsets.get_count(); i++) {
        std::cout << subtable_row_offsets[i] << ", ";
    }
    std::cout << std::endl;

    return 0;
}
