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
#include "oneapi/dal/table/csr.hpp"
#include <cmath>

namespace dal = oneapi::dal;
using oneapi::dal::detail::empty_delete;

/// Converts a homogen table to one-based CSR table.
///
/// @param[in] data     an input homogen table
inline const dal::csr_table convert_to_csr(const dal::table& data) {
    std::int64_t non_zero_count = 0;
    dal::row_accessor<const float> accessor{ data };
    const auto data_ptr = accessor.pull({ 0, -1 });
    const std::int64_t column_count = data.get_column_count();
    const std::int64_t row_count = data.get_row_count();

    for (std::int64_t i = 0; i < data_ptr.get_count(); i++) {
        if (std::fabs(data_ptr[i]) > std::numeric_limits<float>::epsilon()) {
            non_zero_count++;
        }
    }
    float* compressed_data = new float[non_zero_count];
    std::int64_t* row_offsets = new std::int64_t[row_count + 1];
    std::int64_t* col_indices = new std::int64_t[non_zero_count];

    std::uint64_t compressed_idx = 1;
    for (std::int64_t i = 0; i <= row_count; i++) {
        row_offsets[i] = 1;
    }

    for (std::int64_t i = 0; i < data_ptr.get_count(); i++) {
        if (std::fabs(data_ptr[i]) > std::numeric_limits<float>::epsilon()) {
            compressed_data[compressed_idx - 1] = data_ptr[i];
            std::int64_t row_idx = i / column_count + 1;
            std::int64_t col_idx = i % column_count + 1;
            row_offsets[row_idx] = compressed_idx;
            col_indices[compressed_idx - 1] = col_idx;
            compressed_idx++;
        }
    }
    // Fill empty rows with previous value
    for (std::int64_t i = 1; i <= row_count; i++) {
        row_offsets[i] = std::max(row_offsets[i - 1], row_offsets[i]);
    }

    dal::csr_table t{ compressed_data,
                      col_indices,
                      row_offsets,
                      row_count,
                      column_count,
                      empty_delete<const float>(),
                      empty_delete<const std::int64_t>(),
                      empty_delete<const std::int64_t>(),
                      oneapi::dal::sparse_indexing::one_based };
    return t;
}
