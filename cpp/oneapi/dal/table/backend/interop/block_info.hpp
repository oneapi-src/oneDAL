/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#pragma once

#include <limits>
#include <daal/include/data_management/data/numeric_table.h>

#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::backend::interop {

class block_info {
    template <typename T>
    using block_desc_t = daal::data_management::BlockDescriptor<T>;

public:
    template <typename BlockData>
    block_info(const block_desc_t<BlockData>& block,
               std::size_t row_begin_index,
               std::size_t value_count,
               std::size_t column_index)
            : block_info(block, row_begin_index, value_count) {
        this->column_index = detail::integral_cast<std::int64_t>(column_index);
        this->single_column_requested = true;
    }

    template <typename BlockData>
    block_info(const block_desc_t<BlockData>& block,
               std::size_t row_begin_index,
               std::size_t row_count) {
        const auto bd_row_count = detail::integral_cast<std::int64_t>(block.getNumberOfRows());
        const auto bd_column_count =
            detail::integral_cast<std::int64_t>(block.getNumberOfColumns());

        this->row_begin_index = detail::integral_cast<std::int64_t>(row_begin_index);
        this->row_count = detail::integral_cast<std::int64_t>(row_count);

        this->bd_element_count = bd_row_count * bd_column_count;
        ONEDAL_ASSERT((bd_element_count > bd_row_count && bd_element_count > bd_column_count) ||
                      (bd_element_count == 0 && (bd_row_count == 0 || bd_column_count == 0)));

        this->row_end_index = this->row_begin_index + this->row_count;
        ONEDAL_ASSERT(row_end_index > this->row_begin_index && row_end_index >= this->row_count);

        this->column_index = -1;
        this->single_column_requested = false;
    }

    std::int64_t bd_element_count;
    std::int64_t column_index;
    std::int64_t row_begin_index;
    std::int64_t row_end_index;
    std::int64_t row_count;
    bool single_column_requested;
};

} // namespace oneapi::dal::backend::interop
