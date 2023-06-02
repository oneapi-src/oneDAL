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

#include "oneapi/dal/table/backend/csr_kernels.hpp"
#include "oneapi/dal/table/backend/convert.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/memory.hpp"

#include <algorithm>

namespace oneapi::dal::backend {

using error_msg = dal::detail::error_messages;

ONEDAL_FORCEINLINE void check_origin_data(const array<byte_t>& origin_data,
                                          std::int64_t element_count,
                                          std::int64_t origin_dtype_size,
                                          std::int64_t block_dtype_size) {
    detail::check_mul_overflow(element_count, std::max(origin_dtype_size, block_dtype_size));
    ONEDAL_ASSERT(origin_data.get_count() >= element_count * origin_dtype_size);
}

/// Provides access to the block of rows in `data` array of the table in CSR format.
/// The method returns an array that directly points to the memory within the table
/// if it is possible. In that case, the array refers to the memory as to immutable data.
/// Otherwise, the new memory block is allocated, the data from the table rows is converted
/// and copied into this block. In this case, the array refers to the block as to mutable data.
///
/// @tparam Policy       Execution policy type. The :literal:`Policy` type should be `default_host_policy`
///                      or `data_parallel_policy`.
/// @tparam BlockData    The type of elements in the output array. In case the data values in CSR table
///                      are stored in the different data type, the conversion to `BlockData` type needs
///                      to be performed.
///                      The :literal:`BlockData` type should be at least :expr:`float`, :expr:`double`
///                      or :expr:`std::int32_t`.
///
/// @param[in] policy               Execution policy.
/// @param[in] origin_info          Information about this block of rows in the table.
///                                 Contains: layout, number of rows, information about data type,
///                                 indexing, etc.
/// @param[in] origin_data          Memory block that stores `data` array in the CSR table.
/// @param[in] same_data_type       True if the data type of the data in CSR table is the same as the data
///                                 type requested by the accessor's pull method. False otherwise.
/// @param[in] origin_offset        Index of the starting element of the `data` array in CSR table
///                                 to be pulled.
/// @param[in] block_size           Number of elemenst of the `data` array in CSR table to be pulled.
/// @param[out] data                Array that stores the block of rows of `data` array pulled
///                                 from the CSR table.
/// @param[in] kind                 The requested kind of USM in the returned block.
/// @param[in] preserve_mutability  True if the mutability should be preserved in the output array.
///                                 False otherwise.
template <typename Policy, typename BlockData>
void pull_data_impl(const Policy& policy,
                    const csr_info& origin_info,
                    const array<byte_t>& origin_data,
                    const bool same_data_type,
                    const std::int64_t origin_offset,
                    const std::int64_t block_size,
                    array<BlockData>& data,
                    alloc_kind kind,
                    bool preserve_mutability) {
    constexpr std::int64_t block_dtype_size = sizeof(BlockData);
    constexpr data_type block_dtype = detail::make_data_type<BlockData>();

    const auto origin_dtype_size = detail::get_data_type_size(origin_info.dtype_);

    if (same_data_type && !alloc_kind_requires_copy(get_alloc_kind(origin_data), kind)) {
        refer_origin_data(origin_data,
                          origin_offset * block_dtype_size,
                          block_size,
                          data,
                          preserve_mutability);
    }
    else {
        if (data.get_count() < block_size || !data.has_mutable_data() ||
            alloc_kind_requires_copy(get_alloc_kind(data), kind)) {
            reset_array(policy, data, block_size, kind);
        }

        const byte_t* const src_data = origin_data.get_data() + origin_offset * origin_dtype_size;
        BlockData* const dst_data = data.get_mutable_data();

        backend::convert_vector(policy,
                                src_data + origin_offset * block_dtype_size,
                                dst_data,
                                origin_info.dtype_,
                                block_dtype,
                                block_size);
    }
}

/// Provides access to the block of rows in `column_indices` array of the table in CSR format.
/// The method returns an array that directly points to the memory within the table
/// if it is possible. In that case, the array refers to the memory as to immutable data.
/// Otherwise, the new memory block is allocated, the data from the table rows is converted
/// and copied into this block. In this case, the array refers to the block as to mutable data.
///
/// @tparam Policy       Execution policy type. The :literal:`Policy` type should be `default_host_policy`
///                      or `data_parallel_policy`.
///
/// @param[in] policy                   Execution policy.
/// @param[in] origin_column_indices    `column_indices` array in the CSR table.
/// @param[in] origin_offset            Index of the starting element of the `column_indices` array
///                                     in CSR table to be pulled.
/// @param[in] block_size               Number of elemenst of the `column_indices` array in CSR table
///                                     to be pulled.
/// @param[in] indices_offset           The offset between the indices in the CSR table and the indices
///                                     requested by the pull method:
///                                         0, if the indexing is the same in the table
///                                            and in the pull method;
///                                         1, if the table has zero-based indexing
///                                            and pull method resuests one=based indexing;
///                                        -1, if the table has one-based indexing
///                                            and pull method resuests zero=based indexing.
/// @param[out] column_indices      Array that stores the block of rows of `column_indices` array pulled
///                                 from the CSR table.
/// @param[in] kind                 The requested kind of USM in the returned block.
/// @param[in] preserve_mutability  True if the mutability should be preserved in the output array.
///                                 False otherwise.
template <typename Policy>
void pull_column_indices_impl(const Policy& policy,
                              const array<std::int64_t>& origin_column_indices,
                              const std::int64_t origin_offset,
                              const std::int64_t block_size,
                              const std::int64_t indices_offset,
                              array<std::int64_t>& column_indices,
                              alloc_kind kind,
                              bool preserve_mutability) {
    if (!alloc_kind_requires_copy(get_alloc_kind(origin_column_indices), kind) &&
        indices_offset == 0) {
        refer_origin_data(origin_column_indices,
                          origin_offset,
                          block_size,
                          column_indices,
                          preserve_mutability);
    }
    else {
        reset_array(policy, column_indices, block_size, kind);

        const auto dtype_size = sizeof(std::int64_t);
        const std::int64_t* const src_data =
            origin_column_indices.get_data() + origin_offset * dtype_size;
        std::int64_t* const dst_data = column_indices.get_mutable_data();

        backend::convert_vector(policy,
                                src_data + origin_offset * dtype_size,
                                dst_data,
                                data_type::int64,
                                data_type::int64,
                                block_size);

        if (indices_offset != 0) {
            shift_array_values(policy, dst_data, block_size, indices_offset);
        }
    }
}

/// Provides access to the block of values in `row_offsets` array of the table in CSR format.
/// The method returns an array that directly points to the memory within the table
/// if it is possible. In that case, the array refers to the memory as to immutable data.
/// Otherwise, the new memory block is allocated, the data from the table rows is converted
/// and copied into this block. In this case, the array refers to the block as to mutable data.
///
/// @tparam Policy       Execution policy type. The :literal:`Policy` type should be `default_host_policy`
///                      or `data_parallel_policy`.
///
/// @param[in] policy                   Execution policy.
/// @param[in] origin_row_offsets       `row_offsets` array in the CSR table.
/// @param[in] block_info               Information about the block of rows requested by the pull method.
///                                     Contains: layout, number of rows, information about data type,
///                                     indexing, etc.
/// @param[in] indices_offset           The offset between the indices in the CSR table and the indices
///                                     requested by the pull method:
///                                         0, if the indexing is the same in the table
///                                            and in the pull method;
///                                         1, if the table has zero-based indexing
///                                            and pull method resuests one=based indexing;
///                                        -1, if the table has one-based indexing
///                                            and pull method resuests zero=based indexing.
/// @param[out] row_offsets         Array that stores the block of values of `row_offsets` array pulled
///                                 from the CSR table.
/// @param[in] kind                 The requested kind of USM in the returned block.
/// @param[in] preserve_mutability  True if the mutability should be preserved in the output array.
///                                 False otherwise.
template <typename Policy>
void pull_row_offsets_impl(const Policy& policy,
                           const array<std::int64_t>& origin_row_offsets,
                           const block_info& block_info,
                           const std::int64_t indices_offset,
                           array<std::int64_t>& row_offsets,
                           alloc_kind kind,
                           bool preserve_mutability) {
    if (row_offsets.get_count() < block_info.row_count_ + 1 || !row_offsets.has_mutable_data() ||
        alloc_kind_requires_copy(get_alloc_kind(row_offsets), kind)) {
        reset_array(policy, row_offsets, block_info.row_count_ + 1, kind);
    }
    if (block_info.row_offset_ == 0 && indices_offset == 0) {
        refer_origin_data(origin_row_offsets,
                          0,
                          block_info.row_count_,
                          row_offsets,
                          preserve_mutability);
    }
    else {
        const std::int64_t* const src_row_offsets = origin_row_offsets.get_data();
        std::int64_t* const dst_row_offsets = row_offsets.get_mutable_data();
        const std::int64_t dst_row_offsets_count = block_info.row_count_ + 1;

        for (std::int64_t i = 0; i < dst_row_offsets_count; i++) {
            dst_row_offsets[i] = src_row_offsets[block_info.row_offset_ + i] -
                                 src_row_offsets[block_info.row_offset_] + 1;
        }

        if (indices_offset != 0) {
            shift_array_values(policy, dst_row_offsets, dst_row_offsets_count, indices_offset);
        }
    }
}

template <typename Policy, typename BlockData>
void pull_csr_block_impl(const Policy& policy,
                         const csr_info& origin_info,
                         const block_info& block_info,
                         const array<byte_t>& origin_data,
                         const array<std::int64_t>& origin_column_indices,
                         const array<std::int64_t>& origin_row_offsets,
                         array<BlockData>& data,
                         array<std::int64_t>& column_indices,
                         array<std::int64_t>& row_offsets,
                         alloc_kind kind,
                         bool preserve_mutability) {
    constexpr std::int64_t block_dtype_size = sizeof(BlockData);
    constexpr data_type block_dtype = detail::make_data_type<BlockData>();

    const auto origin_dtype_size = detail::get_data_type_size(origin_info.dtype_);

    // overflows checked here
    check_origin_data(origin_data, origin_info.element_count_, origin_dtype_size, block_dtype_size);

    const std::int64_t origin_offset =
        origin_row_offsets[block_info.row_offset_] - origin_row_offsets[0];
    ONEDAL_ASSERT(origin_offset >= 0);

    const std::int64_t block_size =
        origin_row_offsets[block_info.row_offset_ + block_info.row_count_] -
        origin_row_offsets[block_info.row_offset_];
    ONEDAL_ASSERT(block_size >= 0);

    const bool same_data_type(block_dtype == origin_info.dtype_);

    const int indices_offset = (origin_info.indexing_ == block_info.indexing_)
                                   ? 0
                                   : ((origin_info.indexing_ == sparse_indexing::zero_based &&
                                       block_info.indexing_ == sparse_indexing::one_based)
                                          ? 1
                                          : -1);
    pull_data_impl<Policy, BlockData>(policy,
                                      origin_info,
                                      origin_data,
                                      same_data_type,
                                      origin_offset,
                                      block_size,
                                      data,
                                      kind,
                                      preserve_mutability);

    pull_column_indices_impl<Policy>(policy,
                                     origin_column_indices,
                                     origin_offset,
                                     block_size,
                                     indices_offset,
                                     column_indices,
                                     kind,
                                     preserve_mutability);

    pull_row_offsets_impl<Policy>(policy,
                                  origin_row_offsets,
                                  block_info,
                                  indices_offset,
                                  row_offsets,
                                  kind,
                                  preserve_mutability);
}

template <typename Policy, typename BlockData>
void csr_pull_block(const Policy& policy,
                    const csr_info& origin_info,
                    const block_info& block_info,
                    const array<byte_t>& origin_data,
                    const array<std::int64_t>& origin_column_indices,
                    const array<std::int64_t>& origin_row_offsets,
                    array<BlockData>& data,
                    array<std::int64_t>& column_indices,
                    array<std::int64_t>& row_offsets,
                    alloc_kind requested_alloc_kind,
                    bool preserve_mutability) {
    switch (origin_info.layout_) {
        case data_layout::row_major:
            pull_csr_block_impl(policy,
                                origin_info,
                                block_info,
                                origin_data,
                                origin_column_indices,
                                origin_row_offsets,
                                data,
                                column_indices,
                                row_offsets,
                                requested_alloc_kind,
                                preserve_mutability);
            break;
        default: throw dal::domain_error(error_msg::unsupported_data_layout());
    }
}

std::int64_t csr_get_non_zero_count(const detail::default_host_policy& policy,
                                    const std::int64_t row_count,
                                    const std::int64_t* row_offsets) {
    if (row_count == 0)
        return 0;

    return (row_offsets[row_count] - row_offsets[0]);
}

#ifdef ONEDAL_DATA_PARALLEL

std::int64_t csr_get_non_zero_count(const detail::data_parallel_policy& policy,
                                    const std::int64_t row_count,
                                    const std::int64_t* row_offsets) {
    if (row_count == 0)
        return 0;

    auto q = policy.get_queue();
    std::int64_t first_row_offset{ 0L }, last_row_offset{ 0L };
    auto first_row_event = copy_usm2host(q, &first_row_offset, row_offsets, 1, {});
    auto last_row_event = copy_usm2host(q, &last_row_offset, row_offsets + row_count, 1, {});
    sycl::event::wait_and_throw({ first_row_event, last_row_event });

    return (last_row_offset - first_row_offset);
}

#endif

std::int64_t csr_get_non_zero_count(const array<std::int64_t>& row_offsets) {
    const std::int64_t row_count = row_offsets.get_count() - 1;
#ifdef ONEDAL_DATA_PARALLEL
    const auto optional_queue = row_offsets.get_queue();
    if (optional_queue) {
        return csr_get_non_zero_count(detail::data_parallel_policy{ optional_queue.value() },
                                      row_count,
                                      row_offsets.get_data());
    }
#endif
    return csr_get_non_zero_count(detail::default_host_policy{}, row_count, row_offsets.get_data());
}

#ifdef ONEDAL_DATA_PARALLEL

/// Checks that the elements in the array allocated on device are not descending
///
/// @tparam T   The type of elements in the input array
///
/// @param[in,out] queue    The SYCL* queue object
/// @param[in]     count    The number of elements in the array
/// @param[in]     data     The pointer to the input array
///
/// @return true, if the elements in the array are not descending;
///         false, otherwise
template <typename T>
bool is_sorted(sycl::queue& queue, const std::int64_t count, const T* data) {
    const std::int64_t range_size = count - 1;
    sycl::buffer<T, 1> data_buf(data, sycl::range<1>(range_size));

    // number of pairs of the subsequent elements in the data array that are sorted in desccending order,
    // i.e. for which data[i] > data[i + 1] is true.
    std::int64_t count_descending_pairs{ 0 };
    sycl::buffer<std::int64_t, 1> count_buf(&count_descending_pairs, sycl::range<1>(1));

    // count the number of pairs of the subsequent elements in the data array that are sorted
    // in desccending order using sycl::reduction
    queue
        .submit([&](sycl::handler& cgh) {
            sycl::accessor data_acc(data_buf, cgh, sycl::read_only);
            auto count_descending_reduction =
                sycl::reduction(count_buf, cgh, sycl::ext::oneapi::plus<std::int64_t>());

            cgh.parallel_for(sycl::nd_range<1>{ range_size, 1 },
                             count_descending_reduction,
                             [=](sycl::nd_item<1> idx, auto& count_descending) {
                                 const auto i = idx.get_global_id(0);
                                 if (data_acc[i] > data_acc[i + 1])
                                     count_descending.combine(1);
                             });
        })
        .wait_and_throw();

    return (count_descending_pairs == 0);
}

#endif

template <typename T>
bool is_sorted(const array<T>& arr) {
    const T* data = arr.get_data();
    const std::int64_t count = arr.get_count();
#ifdef ONEDAL_DATA_PARALLEL
    const auto optional_queue = arr.get_queue();
    if (optional_queue) {
        sycl::queue q = optional_queue.value();
        return is_sorted(q, count, data);
    }
#endif
    return std::is_sorted(data, data + count);
}

/// Given the array A[0], ..., A[n-1] which is alolcated on host and two values:
/// `min_value` and `max_value`, checks that min_value <= A[i] <= max_value
/// for each i = 0, ..., n-1.
///
/// @tparam T   The type of elements in the input array
///
/// @param[in]     count     The number of elements in the array, count == n
/// @param[in]     data      The pointer to the input array
/// @param[in]     min_value The lower boundary for the values in the input array
/// @param[in]     max_value The upper boundary for the values in the input array
///
/// @return less_than_min,    if there exists i, 0 <= i <= n-1: A[i] < min_value;
///         within_bounds,    if min_value <= A[i] <= max_value for each i = 0, ..., n-1;
///         greater_than_max, if there exists i, 0 <= i <= n-1: A[i] > max_value.
template <typename T>
out_of_bound_type check_bounds(const std::int64_t count,
                               const T* data,
                               const T& min_value,
                               const T& max_value) {
    out_of_bound_type result{ out_of_bound_type::within_bounds };
    for (std::int64_t i = 0; i < count; ++i) {
        if (data[i] < min_value) {
            result = out_of_bound_type::less_than_min;
            break;
        }
        if (data[i] > max_value) {
            result = out_of_bound_type::greater_than_max;
            break;
        }
    }
    return result;
}

#ifdef ONEDAL_DATA_PARALLEL

/// Given the array A[0], ..., A[n-1] which is allocated on device and two values:
/// `min_value` and `max_value`, checks that min_value <= A[i] <= max_value
/// for each i = 0, ..., n-1.
///
/// @tparam T   The type of elements in the input array
///
/// @param[in,out] queue     The SYCL* queue object
/// @param[in]     count     The number of elements in the array, count == n
/// @param[in]     data      The pointer to the input array
/// @param[in]     min_value The lower boundary for the values in the input array
/// @param[in]     max_value The upper boundary for the values in the input array
///
/// @return less_than_min,    if there exists i, 0 <= i <= n-1: A[i] < min_value;
///         within_bounds,    if min_value <= A[i] <= max_value for each i = 0, ..., n-1;
///         greater_than_max, if there exists i, 0 <= i <= n-1: A[i] > max_value.
template <typename T>
out_of_bound_type check_bounds(sycl::queue& queue,
                               const std::int64_t count,
                               const T* data,
                               const T& min_value,
                               const T& max_value) {
    sycl::buffer<T, 1> data_buf(data, sycl::range<1>(count));

    // number of elements in the data array that are less than [lt] the min_value,
    // number of elements in the data array that are greater than [gt] the max_value.
    std::int64_t count_lt_min{ 0 }, count_gt_max{ 0 };
    sycl::buffer<std::int64_t, 1> count_lt_buf(&count_lt_min, sycl::range<1>(1));
    sycl::buffer<std::int64_t, 1> count_gt_buf(&count_gt_max, sycl::range<1>(1));

    // count the number of elements which are less than min_vaule using sycl::reduction
    auto event_count_lt_min = queue.submit([&](sycl::handler& cgh) {
        sycl::accessor data_acc(data_buf, cgh, sycl::read_only);
        auto count_lt_reduction = sycl::reduction(count_lt_buf, cgh, sycl::ext::oneapi::plus<std::int64_t>());

        cgh.parallel_for(sycl::nd_range<1>{ count, 1 },
                         count_lt_reduction,
                         [=](sycl::nd_item<1> idx, auto& count_lt) {
                             const auto i = idx.get_global_id(0);
                             if (data_acc[i] < min_value) {
                                 count_lt.combine(1);
                             }
                         });
    });

    // count the number of elements which are greater than max_vaule using sycl::reduction
    auto event_count_gt_max = queue.submit([&](sycl::handler& cgh) {
        sycl::accessor data_acc(data_buf, cgh, sycl::read_only);
        auto count_gt_reduction = sycl::reduction(count_gt_buf, cgh, sycl::ext::oneapi::plus<std::int64_t>());

        cgh.parallel_for(sycl::nd_range<1>{ count, 1 },
                         count_gt_reduction,
                         [=](sycl::nd_item<1> idx, auto& count_gt) {
                             const auto i = idx.get_global_id(0);
                             if (data_acc[i] > max_value) {
                                 count_gt.combine(1);
                             }
                         });
    });

    sycl::event::wait_and_throw({ event_count_lt_min, event_count_gt_max });

    out_of_bound_type result{ out_of_bound_type::within_bounds };
    if (count_lt_min > 0)
        result = out_of_bound_type::less_than_min;
    else if (count_gt_max > 0)
        result = out_of_bound_type::greater_than_max;
    return result;
}

#endif

template <typename T>
out_of_bound_type check_bounds(const array<T>& arr, const T& min_value, const T& max_value) {
    const T* data = arr.get_data();
    const std::int64_t count = arr.get_count();
#ifdef ONEDAL_DATA_PARALLEL
    const auto optional_queue = arr.get_queue();
    if (optional_queue) {
        sycl::queue q = optional_queue.value();
        return check_bounds(q, count, data, min_value, max_value);
    }
#endif
    return check_bounds(count, data, min_value, max_value);
}

#define INSTANTIATE(Policy, BlockData)                                             \
    template void csr_pull_block(const Policy& policy,                             \
                                 const csr_info& origin_info,                      \
                                 const block_info& block_info,                     \
                                 const array<byte_t>& origin_data,                 \
                                 const array<std::int64_t>& origin_column_indices, \
                                 const array<std::int64_t>& origin_row_offsets,    \
                                 array<BlockData>& data,                           \
                                 array<std::int64_t>& column_indices,              \
                                 array<std::int64_t>& row_offsets,                 \
                                 alloc_kind requested_alloc_kind,                  \
                                 bool preserve_mutability);

#define INSTANTIATE_HOST_POLICY(Data) INSTANTIATE(detail::default_host_policy, Data)

INSTANTIATE_HOST_POLICY(float)
INSTANTIATE_HOST_POLICY(double)
INSTANTIATE_HOST_POLICY(std::int32_t)

#ifdef ONEDAL_DATA_PARALLEL

template bool is_sorted(sycl::queue& queue, const std::int64_t count, const std::int64_t* data);
template out_of_bound_type check_bounds(sycl::queue& queue,
                                        const std::int64_t count,
                                        const std::int64_t* data,
                                        const std::int64_t& min_value,
                                        const std::int64_t& max_value);

#endif

template bool is_sorted(const array<std::int64_t>& arr);
template out_of_bound_type check_bounds(const std::int64_t count,
                                        const std::int64_t* data,
                                        const std::int64_t& min_value,
                                        const std::int64_t& max_value);
template out_of_bound_type check_bounds(const array<std::int64_t>& arr,
                                        const std::int64_t& min_value,
                                        const std::int64_t& max_value);

} // namespace oneapi::dal::backend
