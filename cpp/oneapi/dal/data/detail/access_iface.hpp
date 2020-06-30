/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/data/array.hpp"

namespace oneapi::dal::detail {

struct row_block {
    range rows;

    row_block(const range& rows)
        : rows(rows) {}
};

struct column_values_block {
    std::int64_t column_index;
    range rows;

    column_values_block(std::int64_t idx, const range& rows)
        : column_index(idx),
          rows(rows) {}
};

template <typename DataType,
          typename BlockIndexType,
          typename PolicyType = detail::host_seq_policy,
          typename AllocKind = detail::host_only_alloc>
class access_pull_iface {
public:
    virtual ~access_pull_iface() {}
    virtual void pull(const PolicyType& policy,
                      array<DataType>& block, const BlockIndexType& index,
                      const AllocKind& kind) const = 0;
};

template <typename DataType,
          typename BlockIndexType,
          typename PolicyType = detail::host_seq_policy,
          typename AllocKind = detail::host_only_alloc>
class access_push_iface {
public:
    virtual ~access_push_iface() {}
    virtual void push(const PolicyType& policy,
                      const array<DataType>& block, const BlockIndexType& index,
                      const AllocKind& kind) = 0;
};

template <typename PolicyType, typename AllocKind>
struct access_iface {
    template <typename DataType, typename IndexType>
    using push_ptr_t = shared<access_push_iface<DataType, IndexType, PolicyType, AllocKind>>;

    template <typename DataType, typename IndexType>
    using pull_ptr_t = shared<access_pull_iface<DataType, IndexType, PolicyType, AllocKind>>;

    push_ptr_t<float, row_block>                     push_rows_float32;
    push_ptr_t<double, row_block>                    push_rows_float64;
    push_ptr_t<std::int32_t, row_block>              push_rows_int32;

    push_ptr_t<float, column_values_block>           push_column_float32;
    push_ptr_t<double, column_values_block>          push_column_float64;
    push_ptr_t<std::int32_t, column_values_block>    push_column_int32;

    pull_ptr_t<float, row_block>                     pull_rows_float32;
    pull_ptr_t<double, row_block>                    pull_rows_float64;
    pull_ptr_t<std::int32_t, row_block>              pull_rows_int32;

    pull_ptr_t<float, column_values_block>           pull_column_float32;
    pull_ptr_t<double, column_values_block>          pull_column_float64;
    pull_ptr_t<std::int32_t, column_values_block>    pull_column_int32;
};

using host_access_iface = access_iface<host_seq_policy, host_only_alloc>;

#ifdef ONEAPI_DAL_DATA_PARALLEL
using dpc_access_iface = access_iface<sycl::queue, sycl::usm::alloc>;
#endif

class accessible_iface {
public:
    virtual ~accessible_iface() {}

    virtual const host_access_iface& get_host_access_iface() const = 0;
#ifdef ONEAPI_DAL_DATA_PARALLEL
    virtual const dpc_access_iface& get_dpc_access_iface() const = 0;
#endif
};

template <typename DataType, typename IndexType, typename T>
struct push_access_ptr {};

template <typename DataType, typename IndexType, typename T>
struct pull_access_ptr {};

template <typename T>
struct push_access_ptr<float, row_block, T> {
    const auto& get_value(const T& obj) {
        return obj.push_rows_float32;
    }
};
template <typename T>
struct push_access_ptr<double, row_block, T> {
    const auto& get_value(const T& obj) {
        return obj.push_rows_float64;
    }
};
template <typename T>
struct push_access_ptr<std::int32_t, row_block, T> {
    const auto& get_value(const T& obj) {
        return obj.push_rows_int32;
    }
};
template <typename T>
struct push_access_ptr<float, column_values_block, T> {
    const auto& get_value(const T& obj) {
        return obj.push_column_float32;
    }
};
template <typename T>
struct push_access_ptr<double, column_values_block, T> {
    const auto& get_value(const T& obj) {
        return obj.push_column_float64;
    }
};
template <typename T>
struct push_access_ptr<std::int32_t, column_values_block, T> {
    const auto& get_value(const T& obj) {
        return obj.push_column_int32;
    }
};

template <typename T>
struct pull_access_ptr<float, row_block, T> {
    const auto& get_value(const T& obj) {
        return obj.pull_rows_float32;
    }
};
template <typename T>
struct pull_access_ptr<double, row_block, T> {
    const auto& get_value(const T& obj) {
        return obj.pull_rows_float64;
    }
};
template <typename T>
struct pull_access_ptr<std::int32_t, row_block, T> {
    const auto& get_value(const T& obj) {
        return obj.pull_rows_int32;
    }
};
template <typename T>
struct pull_access_ptr<float, column_values_block, T> {
    const auto& get_value(const T& obj) {
        return obj.pull_column_float32;
    }
};
template <typename T>
struct pull_access_ptr<double, column_values_block, T> {
    const auto& get_value(const T& obj) {
        return obj.pull_column_float64;
    }
};
template <typename T>
struct pull_access_ptr<std::int32_t, column_values_block, T> {
    const auto& get_value(const T& obj) {
        return obj.pull_column_int32;
    }
};

} // namespace oneapi::dal::detail
