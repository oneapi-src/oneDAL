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
#include "oneapi/dal/table/detail/csr_access_iface.hpp"
#include "oneapi/dal/detail/policy.hpp"

namespace oneapi::dal::backend {

/// The status of out-of-bound error
enum class out_of_bound_type { less_than_min = -1, within_bounds = 0, greater_than_max = 1 };

struct csr_info {
    csr_info(data_type dtype,
             data_layout layout,
             std::int64_t row_count,
             std::int64_t column_count,
             std::int64_t element_count,
             sparse_indexing indexing)
            : dtype_(dtype),
              layout_(layout),
              row_count_(row_count),
              column_count_(column_count),
              element_count_(element_count),
              indexing_(indexing) {
        ONEDAL_ASSERT(row_count_ > 0);
        ONEDAL_ASSERT(column_count_ > 0);
        ONEDAL_ASSERT(element_count_ > 0);
        ONEDAL_ASSERT(indexing_ == sparse_indexing::one_based);
        ONEDAL_ASSERT(layout_ == data_layout::row_major);
    }

    data_type dtype_;
    data_layout layout_;
    std::int64_t row_count_;
    std::int64_t column_count_;
    std::int64_t element_count_;
    sparse_indexing indexing_;
};

struct block_info {
    block_info(std::int64_t row_offset, std::int64_t row_count, sparse_indexing indexing)
            : row_offset_(row_offset),
              row_count_(row_count),
              indexing_(indexing) {
        ONEDAL_ASSERT(row_offset >= 0);
        ONEDAL_ASSERT(row_count > 0);
        ONEDAL_ASSERT(indexing == sparse_indexing::one_based);
    }

    std::int64_t row_offset_;
    std::int64_t row_count_;
    sparse_indexing indexing_;
};

template <typename Policy, typename BlockData>
void csr_pull_block(const Policy& policy,
                    const csr_info& origin_info,
                    const block_info& block_info,
                    const array<byte_t>& origin_data,
                    const array<std::int64_t>& origin_column_indices,
                    const array<std::int64_t>& origin_row_indices,
                    array<BlockData>& data,
                    array<std::int64_t>& column_indices,
                    array<std::int64_t>& row_indices,
                    alloc_kind requested_alloc_kind,
                    bool preserve_mutability = false);

/// The number of non-zero elements in the table calculated from the row offsets array stored on host.
///
/// @param[in] policy       Default host execution policy
/// @param[in] row_count    The number of rows in the table
/// @param[in] row_offsets  The pointer to row offsets block in CSR layout stored on host
///
/// @return The number of non-zero elements
std::int64_t csr_get_non_zero_count(const detail::default_host_policy& policy,
                                    const std::int64_t row_count,
                                    const std::int64_t* row_offsets);

#ifdef ONEDAL_DATA_PARALLEL

/// The number of non-zero elements in the table calculated from the row offsets array stored in USM
///
/// @param[in] policy       Data parallel execution policy
/// @param[in] row_count    The number of rows in the table
/// @param[in] row_offsets  The pointer to row offsets block in CSR layout stored in USM
///
/// @return The number of non-zero elements
std::int64_t csr_get_non_zero_count(const detail::data_parallel_policy& policy,
                                    const std::int64_t row_count,
                                    const std::int64_t* row_offsets);

#endif

/// The number of non-zero elements in the table calculated from the row offsets array in CSR format.
///
/// This function dispatches the execution:
///     If data parallel policy is enabled and there is a sycl queue associated with row offsets array,
///         then DPC++ implementation of the function is called.
///     Otherwise, C++ implementation is called.
///
/// @param[in] row_offsets  Row offsets array in CSR layout
///
/// @return The number of non-zero elements in CSR table
std::int64_t csr_get_non_zero_count(const array<std::int64_t>& row_offsets);

/// Checks that the elements in the input array are not descending
///
/// This function dispatches the execution:
///     If data parallel policy is enabled and there is a sycl queue associated with the array `arr`,
///         then DPC++ implementation of the function is called.
///     Otherwise, std::is_sorted is called.
///
/// @tparam T   The type of elements in the input array
///
/// @param arr  Input array
///
/// @return true, if the elements in the array are not descending;
///         false, otherwise
template <typename T>
bool is_sorted(const array<T>& arr);

/// Given the array A[0], ..., A[n-1] and two values: `min_value` and `max_value`,
/// checks that min_value <= A[i] <= max_value for each i = 0, ..., n-1.
///
/// This function dispatches the execution:
///     If data parallel policy is enabled and there is a sycl queue associated with the array `arr`,
///         then DPC++ implementation of the function is called.
///     Otherwise, C++ implementation is called.
///
/// @tparam T   The type of elements in the input array
///
/// @param[in] arr       Input array
/// @param[in] min_value The lower boundary for the values in the input array
/// @param[in] max_value The upper boundary for the values in the input array
///
/// @return less_than_min,    if there exists i for which A[i] < min_value;
///         within_bounds,    if min_value <= A[i] <= max_value for each i = 0, ..., n-1;
///         greater_than_max, if there exists i for which A[i] > max_value.
template <typename T>
out_of_bound_type check_bounds(const array<T>& arr, const T& min_value, const T& max_value);

} // namespace oneapi::dal::backend
