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

#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::backend {

enum class alloc_kind {
    host, /// Non-USM pointer allocated on host
    usm_host, /// USM pointer allocated by sycl::alloc_host
    usm_device, /// USM pointer allocated by sycl::alloc_device
    usm_shared /// USM pointer allocated by sycl::alloc_shared
};

#ifdef ONEDAL_DATA_PARALLEL
inline alloc_kind alloc_kind_from_sycl(sycl::usm::alloc alloc) {
    using error_msg = dal::detail::error_messages;
    switch (alloc) {
        case sycl::usm::alloc::host: return alloc_kind::usm_host;
        case sycl::usm::alloc::device: return alloc_kind::usm_device;
        case sycl::usm::alloc::shared: return alloc_kind::usm_shared;
        default: throw invalid_argument{ error_msg::unsupported_usm_alloc() };
    }
}
#endif

#ifdef ONEDAL_DATA_PARALLEL
inline sycl::usm::alloc alloc_kind_to_sycl(alloc_kind kind) {
    using error_msg = dal::detail::error_messages;
    switch (kind) {
        case alloc_kind::usm_host: return sycl::usm::alloc::host;
        case alloc_kind::usm_device: return sycl::usm::alloc::device;
        case alloc_kind::usm_shared: return sycl::usm::alloc::shared;
        default: throw invalid_argument{ error_msg::unsupported_usm_alloc() };
    }
}
#endif

inline bool alloc_kind_requires_copy(alloc_kind src_alloc_kind, alloc_kind dst_alloc_kind) {
#ifdef ONEDAL_DATA_PARALLEL
    switch (dst_alloc_kind) {
        case alloc_kind::host: //
            return (src_alloc_kind == alloc_kind::usm_device);
        case alloc_kind::usm_host: //
            return (src_alloc_kind == alloc_kind::host) || //
                   (src_alloc_kind == alloc_kind::usm_device);
        case alloc_kind::usm_device: //
            return (src_alloc_kind == alloc_kind::host) || //
                   (src_alloc_kind == alloc_kind::usm_host);
        case alloc_kind::usm_shared: //
            return (src_alloc_kind != alloc_kind::usm_shared);
        default: //
            ONEDAL_ASSERT(!"Unsupported alloc_kind");
    }
#else
    ONEDAL_ASSERT(src_alloc_kind == alloc_kind::host);
    ONEDAL_ASSERT(dst_alloc_kind == alloc_kind::host);
    return false;
#endif
}

template <typename T>
inline alloc_kind get_alloc_kind(const array<T>& array) {
#ifdef ONEDAL_DATA_PARALLEL
    const auto opt_queue = array.get_queue();
    if (opt_queue.has_value()) {
        const auto queue = opt_queue.value();
        const auto array_alloc = sycl::get_pointer_type(array.get_data(), queue.get_context());
        return alloc_kind_from_sycl(array_alloc);
    }
#endif
    return alloc_kind::host;
}

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
