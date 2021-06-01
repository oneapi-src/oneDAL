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

#pragma once

#include "oneapi/dal/table/backend/common_kernels.hpp"

namespace oneapi::dal::backend {

struct homogen_info {
    homogen_info(std::int64_t row_count,
                 std::int64_t column_count,
                 data_type dtype,
                 data_layout layout)
            : row_count_(row_count),
              column_count_(column_count),
              dtype_(dtype),
              layout_(layout) {
        ONEDAL_ASSERT(row_count > 0);
        ONEDAL_ASSERT(column_count > 0);
        ONEDAL_ASSERT_MUL_OVERFLOW(std::int64_t, row_count_, column_count_);
    }

    std::int64_t get_row_count() const {
        return row_count_;
    }

    std::int64_t get_column_count() const {
        return column_count_;
    }

    data_type get_data_type() const {
        return dtype_;
    }

    data_layout get_layout() const {
        return layout_;
    }

    std::int64_t get_data_type_size() const {
        return detail::get_data_type_size(dtype_);
    }

    std::int64_t get_element_count() const {
        return row_count_ * column_count_;
    }

private:
    std::int64_t row_count_;
    std::int64_t column_count_;
    data_type dtype_;
    data_layout layout_;
};

template <typename Policy, typename BlockData>
void homogen_pull_rows(const Policy& policy,
                       const homogen_info& origin_info,
                       const array<byte_t>& origin_data,
                       array<BlockData>& block_data,
                       const range& rows_range,
                       alloc_kind requested_alloc_kind,
                       bool preserve_mutability = false);

template <typename Policy, typename BlockData>
void homogen_pull_column(const Policy& policy,
                         const homogen_info& origin_info,
                         const array<byte_t>& origin_data,
                         array<BlockData>& block_data,
                         std::int64_t column_index,
                         const range& rows_range,
                         alloc_kind requested_alloc_kind,
                         bool preserve_mutability = false);

template <typename Policy, typename BlockData>
void homogen_push_rows(const Policy& policy,
                       const homogen_info& origin_info,
                       array<byte_t>& origin_data,
                       const array<BlockData>& block_data,
                       const range& rows_range);

template <typename Policy, typename BlockData>
void homogen_push_column(const Policy& policy,
                         const homogen_info& origin_info,
                         array<byte_t>& origin_data,
                         const array<BlockData>& block_data,
                         std::int64_t column_index,
                         const range& rows_range);

} // namespace oneapi::dal::backend
