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

#include "oneapi/dal/detail/memory.hpp"

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
                                    const array<detail::chunked_array_base>& data) {
    return get_column_count(meta, data);
}

std::int64_t heterogen_row_count(std::int64_t column_count,
                                 const table_metadata& meta,
                                 const array<detail::chunked_array_base>& data) {
    return get_row_count(column_count, meta, data);
}

std::pair<std::int64_t, std::int64_t> heterogen_shape(const table_metadata& meta,
                                      const array<detail::chunked_array_base>& data) {
    const auto column_count = heterogen_column_count(meta, data);
    const auto row_count = heterogen_row_count(column_count, meta, data);
    return std::pair<std::int64_t, std::int64_t>{ row_count, column_count };
}

std::int64_t heterogen_row_count(const table_metadata& meta,
                                 const array<detail::chunked_array_base>& data) {
    return heterogen_shape(meta, data).first;
}

template <typename Policy>
struct heterogen_dispatcher {};

template <>
struct heterogen_dispatcher<detail::default_host_policy> {
    using policy_t = detail::default_host_policy;

    template <typename Type>
    static void pull(const policy_t& policy,
                     const table_metadata& meta,
                     const array<detail::chunked_array_base>& data,
                     array<Type>& block_data,
                     const range& rows_range,
                     alloc_kind requested_alloc_kind) {

    }
};

#ifdef ONEDAL_DATA_PARALLEL

template <>
struct heterogen_dispatcher<detail::data_parallel_policy> {
    using policy_t = detail::data_parallel_policy;

    template <typename Type>
    static void pull(const policy_t& policy,
                     const table_metadata& meta,
                     const array<detail::chunked_array_base>& data,
                     array<Type>& block_data,
                     const range& rows_range,
                     alloc_kind requested_alloc_kind) {

    }
};

#endif // ONEDAL_DATA_PARALLEL

template <typename Policy, typename Type>
void heterogen_pull_rows(const Policy& policy,
                         const table_metadata& meta,
                         const array<detail::chunked_array_base>& data,
                         array<Type>& block_data,
                         const range& rows_range,
                         alloc_kind requested_alloc_kind) {
    heterogen_dispatcher<Policy>::template pull<Type>(policy, meta, data,
        	                block_data, rows_range, requested_alloc_kind);
}

#define INSTANTIATE(Policy, Type)                                               \
    template void heterogen_pull_rows(const Policy&,                            \
                                      const table_metadata&,                    \
                                      const array<detail::chunked_array_base>&, \
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