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

/// Infomation about the table in CSR format
struct csr_info {
    /// Creates the information instance from external information about the data type,
    /// elements count and the indexing scheme.
    ///
    /// @param[in] dtype         The data types of the values stored in the table.
    /// @param[in] row_count     Number of rows in the table.
    /// @param[in] column_count  Number of columns in the table.
    /// @param[in] element_count Number of non-zero elements in the table.
    /// @param[in] indexing      The indexing scheme used to access data in the CSR layout.
    ///                          Should be :literal:`sparse_indexing::zero_based` or
    ///                          :literal:`sparse_indexing::one_based`.
    csr_info(data_type dtype,
             std::int64_t row_count,
             std::int64_t column_count,
             std::int64_t element_count,
             sparse_indexing indexing)
            : dtype_(dtype),
              row_count_(row_count),
              column_count_(column_count),
              element_count_(element_count),
              indexing_(indexing) {
        ONEDAL_ASSERT(row_count_ > 0);
        ONEDAL_ASSERT(column_count_ > 0);
        ONEDAL_ASSERT(element_count_ > 0);
    }

    data_type dtype_;
    std::int64_t row_count_;
    std::int64_t column_count_;
    std::int64_t element_count_;
    sparse_indexing indexing_;
};

/// Infomation about the data block to be pulled from the data table
struct block_info {
    /// Creates the information instance from external information about the block's starting row,
    /// number of rows in the block and the indexing scheme.
    /// @param[in] row_offset    Zero-based index of the starting row in the block of rows to be pulled
    ///                          from the data table.
    /// @param[in] row_count     Number of rows in the block to be pulled from the data table.
    /// @param[in] indexing      The indexing scheme used to access data in the CSR layout.
    ///                          Should be :literal:`sparse_indexing::zero_based` or
    ///                          :literal:`sparse_indexing::one_based`.
    block_info(std::int64_t row_offset, std::int64_t row_count, sparse_indexing indexing)
            : row_offset_(row_offset),
              row_count_(row_count),
              indexing_(indexing) {
        ONEDAL_ASSERT(row_offset >= 0);
        ONEDAL_ASSERT(row_count > 0);
    }

    std::int64_t row_offset_;
    std::int64_t row_count_;
    sparse_indexing indexing_;
};

/// Provides access to the rows of the table in CSR format.
/// The method outputs arrays that directly point to the memory within the table
/// if it is possible. In that case, the arrays refer to the memory as to immutable data.
/// Otherwise, new memory blocks are allocated, the data from the table rows is converted
/// and copied into those blocks. In this case, arrays refer to the blocks as to mutable data.
///
/// @tparam Policy      Execution policy. The :literal:`Policy` should be :literal:`default_host_policy`
///                     or :literal:`data_parallel_policy`.
/// @tparam BlockData   The type of elements in the output data block.
///                     While `csr_table` supports wide range of data types for data representation,
///                     only 3 data types are supported currently for data pulling: :expr:`float`,
///                     :expr:`double` and :expr:`std::int32_t`.
///
/// @param[in] queue                 The SYCL* queue object.
/// @param[in] origin_info           Metainformation about the input data table in CSR format.
/// @param[in] block_info            Metainformation about the output data block.
/// @param[in] origin_data           The array that stores values block of the input data table
///                                  in the CSR layout.
/// @param[in] origin_column_indices The array that stores column indices block of the input data table
///                                  in the CSR layout.
/// @param[in] origin_row_offsets    The array that stores row offsets block of the input data table
///                                  in the CSR layout.
/// @param[out] data                 The output array that stores values block of the pulled sub-table
///                                  in the CSR layout.
/// @param[out] column_indices       The output array that stores column indices block of the pulled
///                                  sub-table in the CSR layout.
/// @param[out] row_offsets          The output array that stores row_offsets block of the pulled
///                                  sub-table in the CSR layout.
/// @param[in] requested_alloc_kind  The requested kind of USM or non-USM allocation in the returned block.
/// @param[in] preserve_mutability   True, if the mutability of the data should be preserverved.
///                                  False, otherwise.
template <typename Policy, typename BlockData>
void csr_pull_block(const Policy& policy,
                    const csr_info& origin_info,
                    const block_info& block_info,
                    const array<byte_t>& origin_data,
                    const array<std::int64_t>& origin_column_indices,
                    const array<std::int64_t>& origin_row_offsets,
                    array<BlockData>& data,
                    array<std::int64_t>& column_indices,
                    array<std::int64_t>& row_indices,
                    alloc_kind requested_alloc_kind,
                    bool preserve_mutability = false);

#ifdef ONEDAL_DATA_PARALLEL
/// The number of non-zero elements in the table calculated from the row offsets array stored in USM
///
/// @param[in] queue        The SYCL* queue object
/// @param[in] row_count    The number of rows in the table
/// @param[in] row_offsets  The pointer to row offsets block in CSR layout stored in USM
/// @param[in] dependencies Events indicating availability of the `row_offsets` for reading.
///
/// @return The number of non-zero elements
std::int64_t csr_get_non_zero_count(sycl::queue& queue,
                                    const std::int64_t row_count,
                                    const std::int64_t* row_offsets,
                                    const std::vector<sycl::event>& dependencies);
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

#ifdef ONEDAL_DATA_PARALLEL

/// Checks that the elements in the input array are not descending
///
/// @tparam T   The type of elements in the input array
///
/// @param[in] arr          Input array
/// @param[in] dependencies Events indicating availability of the `arr` for reading.
///
/// @return true, if the elements in the array are not descending;
///         false, otherwise
template <typename T>
bool is_sorted(const array<T>& arr, const std::vector<sycl::event>& dependencies);

#endif

/// Given the array A[0], ..., A[n-1] and two values: `min_value` and `max_value`,
/// checks that min_value <= A[i] <= max_value for each i = 0, ..., n-1.
///
/// @tparam T   The type of elements in the input array
///
/// @param[in] arr       Input array
/// @param[in] min_value The lower boundary for the values in the input array
/// @param[in] max_value The upper boundary for the values in the input array
///
/// @return less_than_min,    if there exists i, 0 <= i <= n-1: A[i] < min_value;
///         within_bounds,    if min_value <= A[i] <= max_value for each i = 0, ..., n-1;
///         greater_than_max, if there exists i, 0 <= i <= n-1: A[i] > max_value.
template <typename T>
out_of_bound_type check_bounds(const array<T>& arr, T min_value, T max_value);

#ifdef ONEDAL_DATA_PARALLEL

/// Given the array A[0], ..., A[n-1] and two values: `min_value` and `max_value`,
/// checks that min_value <= A[i] <= max_value for each i = 0, ..., n-1.
///
/// @tparam T   The type of elements in the input array
///
/// @param[in] arr          Input array
/// @param[in] min_value    The lower boundary for the values in the input array
/// @param[in] max_value    The upper boundary for the values in the input array
/// @param[in] dependencies Events indicating availability of the `arr` for reading.
///
/// @return less_than_min,    if there exists i, 0 <= i <= n-1: A[i] < min_value;
///         within_bounds,    if min_value <= A[i] <= max_value for each i = 0, ..., n-1;
///         greater_than_max, if there exists i, 0 <= i <= n-1: A[i] > max_value.
template <typename T>
out_of_bound_type check_bounds(const array<T>& arr,
                               T min_value,
                               T max_value,
                               const std::vector<sycl::event>& dependencies);

#endif

} // namespace oneapi::dal::backend
