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

namespace oneapi::dal::backend {

using error_msg = dal::detail::error_messages;

ONEDAL_FORCEINLINE void check_origin_data(const array<byte_t>& origin_data,
                                          std::int64_t element_count,
                                          std::int64_t origin_dtype_size,
                                          std::int64_t block_dtype_size) {
    detail::check_mul_overflow(element_count, std::max(origin_dtype_size, block_dtype_size));
    ONEDAL_ASSERT(origin_data.get_count() >= element_count * origin_dtype_size);
}

template <typename Policy, typename BlockData>
void pull_data_impl(const Policy& policy,
                    const csr_info& origin_info,
                    const array<byte_t>& origin_data,
                    const bool same_data_type,
                    const std::int64_t origin_offset,
                    const std::int64_t block_size,
                    detail::csr_block<BlockData>& block,
                    alloc_kind kind,
                    bool preserve_mutability) {
    constexpr std::int64_t block_dtype_size = sizeof(BlockData);
    constexpr data_type block_dtype = detail::make_data_type<BlockData>();

    const auto origin_dtype_size = detail::get_data_type_size(origin_info.dtype_);

    if (same_data_type && !alloc_kind_requires_copy(get_alloc_kind(origin_data), kind)) {
        refer_origin_data(origin_data,
                          origin_offset * block_dtype_size,
                          block_size,
                          block.data,
                          preserve_mutability);
    }
    else {
        if (block.data.get_count() < block_size || !block.data.has_mutable_data() ||
            alloc_kind_requires_copy(get_alloc_kind(block.data), kind)) {
            reset_array(policy, block.data, block_size, kind);
        }

        auto src_data = origin_data.get_data() + origin_offset * origin_dtype_size;
        auto dst_data = block.data.get_mutable_data();

        backend::convert_vector(policy,
                                src_data + origin_offset * block_dtype_size,
                                dst_data,
                                origin_info.dtype_,
                                block_dtype,
                                block_size);
    }
}

template <typename Policy, typename BlockData>
void pull_column_indices_impl(const Policy& policy,
                              const array<std::int64_t>& origin_column_indices,
                              const std::int64_t origin_offset,
                              const std::int64_t block_size,
                              detail::csr_block<BlockData>& block,
                              alloc_kind kind,
                              bool preserve_mutability) {
    if (!alloc_kind_requires_copy(get_alloc_kind(origin_column_indices), kind)) {
        refer_origin_data(origin_column_indices,
                          origin_offset,
                          block_size,
                          block.column_indices,
                          preserve_mutability);
    }
    else {
        reset_array(policy, block.column_indices, block_size, kind);

        const auto dtype_size = sizeof(std::int64_t);
        auto src_data = origin_column_indices.get_data() + origin_offset * dtype_size;
        auto dst_data = block.column_indices.get_mutable_data();

        backend::convert_vector(policy,
                                src_data + origin_offset * dtype_size,
                                dst_data,
                                data_type::int64,
                                data_type::int64,
                                block_size);
    }
}

template <typename Policy, typename BlockData>
void pull_row_indices_impl(const Policy& policy,
                           const array<std::int64_t>& origin_row_indices,
                           const block_info& block_info,
                           detail::csr_block<BlockData>& block,
                           alloc_kind kind,
                           bool preserve_mutability) {
    if (block.row_indices.get_count() < block_info.row_count_ + 1 ||
        !block.row_indices.has_mutable_data() ||
        alloc_kind_requires_copy(get_alloc_kind(block.row_indices), kind)) {
        reset_array(policy, block.row_indices, block_info.row_count_ + 1, kind);
    }
    if (block_info.row_offset_ == 0) {
        refer_origin_data(origin_row_indices,
                          0,
                          block_info.row_count_,
                          block.row_indices,
                          preserve_mutability);
    }
    else {
        auto src_row_indices = origin_row_indices.get_data();
        auto dst_row_indices = block.row_indices.get_mutable_data();

        for (std::int64_t i = 0; i < block_info.row_count_ + 1; i++) {
            dst_row_indices[i] = src_row_indices[block_info.row_offset_ + i] -
                                 src_row_indices[block_info.row_offset_] + 1;
        }
    }
}

template <typename Policy, typename BlockData>
void pull_csr_block_impl(const Policy& policy,
                         const csr_info& origin_info,
                         const block_info& block_info,
                         const array<byte_t>& origin_data,
                         const array<std::int64_t>& origin_column_indices,
                         const array<std::int64_t>& origin_row_indices,
                         detail::csr_block<BlockData>& block,
                         alloc_kind kind,
                         bool preserve_mutability) {
    constexpr std::int64_t block_dtype_size = sizeof(BlockData);
    constexpr data_type block_dtype = detail::make_data_type<BlockData>();

    const auto origin_dtype_size = detail::get_data_type_size(origin_info.dtype_);

    // overflows checked here
    check_origin_data(origin_data, origin_info.element_count_, origin_dtype_size, block_dtype_size);

    const std::int64_t origin_offset =
        origin_row_indices[block_info.row_offset_] - origin_row_indices[0];
    ONEDAL_ASSERT(origin_offset >= 0);

    const std::int64_t block_size =
        origin_row_indices[block_info.row_offset_ + block_info.row_count_] -
        origin_row_indices[block_info.row_offset_];
    ONEDAL_ASSERT(block_size >= 0);

    const bool same_data_type(block_dtype == origin_info.dtype_);

    pull_data_impl<Policy, BlockData>(policy,
                                      origin_info,
                                      origin_data,
                                      same_data_type,
                                      origin_offset,
                                      block_size,
                                      block,
                                      kind,
                                      preserve_mutability);

    pull_column_indices_impl<Policy, BlockData>(policy,
                                                origin_column_indices,
                                                origin_offset,
                                                block_size,
                                                block,
                                                kind,
                                                preserve_mutability);

    pull_row_indices_impl<Policy, BlockData>(policy,
                                             origin_row_indices,
                                             block_info,
                                             block,
                                             kind,
                                             preserve_mutability);
}

template <typename Policy, typename BlockData>
void csr_pull_block(const Policy& policy,
                    const csr_info& origin_info,
                    const block_info& block_info,
                    const array<byte_t>& origin_data,
                    const array<std::int64_t>& origin_column_indices,
                    const array<std::int64_t>& origin_row_indices,
                    detail::csr_block<BlockData>& block,
                    alloc_kind requested_alloc_kind,
                    bool preserve_mutability) {
    switch (origin_info.layout_) {
        case data_layout::row_major:
            pull_csr_block_impl(policy,
                                origin_info,
                                block_info,
                                origin_data,
                                origin_column_indices,
                                origin_row_indices,
                                block,
                                requested_alloc_kind,
                                preserve_mutability);
            break;
        default: throw dal::domain_error(error_msg::unsupported_data_layout());
    }
}

#define INSTANTIATE(Policy, BlockData)                                             \
    template void csr_pull_block(const Policy& policy,                             \
                                 const csr_info& origin_info,                      \
                                 const block_info& block_info,                     \
                                 const array<byte_t>& origin_data,                 \
                                 const array<std::int64_t>& origin_column_indices, \
                                 const array<std::int64_t>& origin_row_indices,    \
                                 detail::csr_block<BlockData>& block,              \
                                 alloc_kind requested_alloc_kind,                  \
                                 bool preserve_mutability);

#ifdef ONEDAL_DATA_PARALLEL
#define INSTANTIATE_ALL_POLICIES(Data)             \
    INSTANTIATE(detail::default_host_policy, Data) \
    INSTANTIATE(detail::data_parallel_policy, Data)
#else
#define INSTANTIATE_ALL_POLICIES(Data) INSTANTIATE(detail::default_host_policy, Data)
#endif

INSTANTIATE_ALL_POLICIES(float)
INSTANTIATE_ALL_POLICIES(double)
INSTANTIATE_ALL_POLICIES(std::int32_t)

} // namespace oneapi::dal::backend
