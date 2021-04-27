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
#include "oneapi/dal/table/detail/sparse_access_iface.hpp"

namespace oneapi::dal::backend {

struct csr_info {
    csr_info(data_type dtype,
             data_layout layout,
             std::int64_t row_count,
             std::int64_t column_count,
             std::int64_t element_count,
             detail::csr_indexing indexing)
            : dtype_(dtype),
              layout_(layout),
              row_count_(row_count),
              column_count_(column_count),
              element_count_(element_count),
              indexing_(indexing) {
        ONEDAL_ASSERT(row_count_ > 0);
        ONEDAL_ASSERT(column_count_ > 0);
        ONEDAL_ASSERT(element_count_ > 0);
        ONEDAL_ASSERT(indexing_ == detail::csr_indexing::one_based);
        ONEDAL_ASSERT(layout_ == data_layout::row_major);
    }

    data_type dtype_;
    data_layout layout_;
    std::int64_t row_count_;
    std::int64_t column_count_;
    std::int64_t element_count_;
    detail::csr_indexing indexing_;
};

struct block_info {
    block_info(std::int64_t row_offset, std::int64_t row_count, detail::csr_indexing indexing)
            : row_offset_(row_offset),
              row_count_(row_count),
              indexing_(indexing) {
        ONEDAL_ASSERT(row_offset >= 0);
        ONEDAL_ASSERT(row_count > 0);
        ONEDAL_ASSERT(indexing == detail::csr_indexing::one_based);
    }

    std::int64_t row_offset_;
    std::int64_t row_count_;
    detail::csr_indexing indexing_;
};

template <typename Policy, typename BlockData>
void csr_pull_sparse_block(const Policy& policy,
                           const csr_info& origin_info,
                           const block_info& block_info,
                           const array<byte_t>& origin_data,
                           const array<std::int64_t>& origin_column_indices,
                           const array<std::int64_t>& origin_row_indices,
                           detail::sparse_block<BlockData>& block,
                           alloc_kind requested_alloc_kind,
                           bool preserve_mutability = false);

} // namespace oneapi::dal::backend
