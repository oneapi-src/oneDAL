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

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/chunked_array.hpp"

/*#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/copy_convert.hpp"
#include "oneapi/dal/backend/primitives/common_convert.hpp"*/

#include "oneapi/dal/detail/memory.hpp"
#include "oneapi/dal/detail/threading.hpp"

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/backend/common_kernels.hpp"
#include "oneapi/dal/table/backend/heterogen_kernels.hpp"

namespace oneapi::dal::backend {

template <typename Meta, typename Data>
inline std::int64_t get_column_count(const Meta& meta, const Data& data) {
    const auto result = meta.get_feature_count();
#ifdef ONEDAL_ENABLE_ASSERT
    ONEDAL_ASSERT(result == data.get_count());
#endif // ONEDAL_ENABLE_ASSERT
    return result;
}

template <typename Meta, typename Data>
inline std::int64_t get_row_count(std::int64_t column_count, const Meta& meta, const Data& data) {
    auto get_count = [&](std::int64_t c) -> std::int64_t {
        const auto& column = data[c];
        const auto dtype = meta.get_data_type(c);
        return detail::get_element_count(dtype, column);
    };

    ONEDAL_ASSERT(0l < column_count);
    const auto result = get_count(0l);

#ifdef ONEDAL_ENABLE_ASSERT
    for (std::int64_t c = 1l; c < column_count; ++c) {
        ONEDAL_ASSERT(get_count(c) == result);
    }
#endif // ONEDAL_ENABLE_ASSERT

    return result;
}

std::int64_t heterogen_column_count(const table_metadata& meta,
                                    const heterogen_data& data) {
    return get_column_count(meta, data);
}

std::int64_t heterogen_row_count(std::int64_t column_count,
                                 const table_metadata& meta,
                                 const heterogen_data& data) {
    return get_row_count(column_count, meta, data);
}

std::pair<std::int64_t, std::int64_t> heterogen_shape(const table_metadata& meta,
                                      const heterogen_data& data) {
    const auto column_count = heterogen_column_count(meta, data);
    const auto row_count = heterogen_row_count(column_count, meta, data);
    return std::pair<std::int64_t, std::int64_t>{ row_count, column_count };
}

std::int64_t heterogen_row_count(const table_metadata& meta,
                                 const heterogen_data& data) {
    return heterogen_shape(meta, data).first;
}

template <typename Policy>
struct heterogen_dispatcher {};

template <typename Meta, typename Data>
std::int64_t get_row_size(const Meta& meta, const Data& data) {
    std::int64_t acc = 0l;

    const auto col_count = get_column_count(meta, data);
    for (std::int64_t col = 0l; col < col_count; ++col) {
        const auto dtype = meta.get_data_type(col);
        acc += detail::get_data_type_size(dtype);
    }

    return acc;
}

template <typename Meta, typename Data>
std::int64_t propose_row_block_size(const Meta& meta, const Data& data) {
    constexpr std::int64_t estimation = 100'000'000;
    const auto row_size = get_row_size(meta, data);
    return estimation / row_size;
}

heterogen_data heterogen_row_slice(const range& rows_range,
                                   const table_metadata& meta,
                                   const heterogen_data& data) {
    const auto col_count = get_column_count(meta, data);
    const auto row_count = get_row_count(col_count, meta, data);

    const auto rows = rows_range.normalize_range(row_count);
    const auto first = rows.start_idx, last = rows.end_idx;

    auto result = heterogen_data::empty(col_count);
    auto* const res_ptr = result.get_mutable_data();

    detail::threader_for_int64(col_count, [&](std::int64_t col) {
        const auto dtype = meta.get_data_type(col);
        const auto elem_size = detail::get_data_type_size(dtype);

        const auto last_byte = detail::check_mul_overflow(elem_size, last);
        const auto first_byte = detail::check_mul_overflow(elem_size, first);

        auto column = chunked_array<dal::byte_t>::make(data[col]);
        auto slice = column.get_slice(first_byte, last_byte);

        res_ptr[col] = std::move(slice);
    });

    return result;
}

template <>
struct heterogen_dispatcher<detail::default_host_policy> {
    using policy_t = detail::default_host_policy;

    template <typename Type>
    static void pull(const policy_t& policy,
                     const table_metadata& meta,
                     const heterogen_data& data,
                     array<Type>& block_data,
                     const range& rows_range,
                     alloc_kind requested_alloc_kind) {
        /*const auto col_count = get_column_count(meta, data);
        const auto row_count = get_row_count(col_count, meta, data);
        const auto [first, last] = rows_range.normalize_range(row_count);

        ONEDAL_ASSERT(first < last);
        const auto copy_count = last - first;

        const auto row_size = get_row_size(meta, data);
        const auto block = propose_row_block_size(meta, data);

        const auto block_size = detail::check_mul_overflow(block, row_size);

#ifdef ONEDAL_ENABLE_ASSERT
        auto full_count = detail::check_mul_overflow(copy_count, col_count);
        ONEDAL_ASSERT(full_count == block_data.get_count());
#endif // ONEDAL_ENABLE_ASSERT

        auto temp = dal::array<dal::byte_t>::empty(block_size);

        for (std::int64_t f = first; f < last; f += block) {
            const auto l = std::min(f + block, last);
            const std::int64_t len = l - f;
            ONEDAL_ASSERT(len <= block);
            ONEDAL_ASSERT(0l < len);

            auto slice = heterogen_row_slice( //
                    dal::range{f, l}, meta, data);

            copy_slice(policy, temp, slice);

        }*/
    }
};

#ifdef ONEDAL_DATA_PARALLEL

template <>
struct heterogen_dispatcher<detail::data_parallel_policy> {
    using policy_t = detail::data_parallel_policy;

    template <typename Type>
    static void pull(const policy_t& policy,
                     const table_metadata& meta,
                     const heterogen_data& data,
                     array<Type>& block_data,
                     const range& rows_range,
                     alloc_kind requested_alloc_kind) {

    }
};

#endif // ONEDAL_DATA_PARALLEL

template <typename Policy, typename Type>
void heterogen_pull_rows(const Policy& policy,
                         const table_metadata& meta,
                         const heterogen_data& data,
                         array<Type>& block_data,
                         const range& rows_range,
                         alloc_kind requested_alloc_kind) {
    heterogen_dispatcher<Policy>::template pull<Type>(policy, meta, data,
        	                block_data, rows_range, requested_alloc_kind);
}

#define INSTANTIATE(Policy, Type)                                               \
    template void heterogen_pull_rows(const Policy&,                            \
                                      const table_metadata&,                    \
                                      const heterogen_data&, \
                                      array<Type>&,                             \
                                      const range&,                             \
                                      alloc_kind);

#ifdef ONEDAL_DATA_PARALLEL
#define INSTANTIATE_ALL_POLICIES(Type)             \
    INSTANTIATE(detail::default_host_policy, Type) \
    INSTANTIATE(detail::data_parallel_policy, Type)
#else
#define INSTANTIATE_ALL_POLICIES(Type) INSTANTIATE(detail::default_host_policy, Type)
#endif

INSTANTIATE_ALL_POLICIES(float)
INSTANTIATE_ALL_POLICIES(double)
INSTANTIATE_ALL_POLICIES(std::int32_t)

} // namespace oneapi::dal::backend