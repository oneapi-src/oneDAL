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

#include "oneapi/dal/data/table_builder.hpp"

#include <stdexcept> // TODO: replace by oneDAL error handling

namespace oneapi::dal {
namespace detail {

template <typename T, typename BlockIndex>
class accessor_base {
public:
    using data_t = std::remove_const_t<T>;

public:
    static constexpr bool is_readonly = std::is_const_v<T>;

#ifdef ONEAPI_DAL_DATA_PARALLEL
    template <typename K>
    accessor_base(const K& obj)
            : host_access_(get_impl<access_provider_iface>(obj).get_access_iface_host()),
              dpc_access_(get_impl<access_provider_iface>(obj).get_access_iface_dpc()) {}
#else
    template <typename K>
    accessor_base(const K& obj)
            : host_access_(get_impl<access_provider_iface>(obj).get_access_iface_host()) {}
#endif

    template <typename Policy, typename AllocKind>
    array<data_t> pull(const Policy& policy, const BlockIndex& idx, const AllocKind& alloc) const {
        array<data_t> block;
        get_access(policy).pull(policy, block, idx, alloc);
        return block;
    }

    template <typename Policy, typename AllocKind>
    T* pull(const Policy& policy,
            array<data_t>& block,
            const BlockIndex& idx,
            const AllocKind& alloc) const {
        get_access(policy).pull(policy, block, idx, alloc);
        if constexpr (is_readonly) {
            return block.get_data();
        }
        else {
            return block.get_mutable_data();
        }
    }

    template <typename Policy>
    void push(const Policy& policy, const array<data_t>& block, const BlockIndex& idx) {
        get_access(policy).push(policy, block, idx);
    }

private:
    access_iface_host& get_access(const detail::cpu_dispatch_default&) {
        return host_access_;
    }
    const access_iface_host& get_access(const detail::cpu_dispatch_default&) const {
        return host_access_;
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    access_iface_dpc& get_access(const data_parallel_policy&) {
        return dpc_access_;
    }
    const access_iface_dpc& get_access(const data_parallel_policy&) const {
        return dpc_access_;
    }
#endif

private:
    access_iface_host& host_access_;
#ifdef ONEAPI_DAL_DATA_PARALLEL
    access_iface_dpc& dpc_access_;
#endif
};

} // namespace detail

template <typename T>
class row_accessor : private detail::accessor_base<T, detail::row_block> {
    using base = detail::accessor_base<T, detail::row_block>;

public:
    using data_t                      = typename base::data_t;
    static constexpr bool is_readonly = base::is_readonly;

public:
    template <typename K,
              typename = std::enable_if_t<is_readonly && (std::is_base_of_v<table, K> ||
                                                          std::is_base_of_v<table_builder, K>)>>
    row_accessor(const K& obj) : base(obj) {}

    row_accessor(const table_builder& b) : base(b) {}

    array<data_t> pull(const range& rows = { 0, -1 }) const {
        return pull(detail::cpu_dispatch_default{}, rows);
    }

    template <typename Policy, typename AllocKind=default_parameter_tag>
    array<data_t> pull(const Policy& policy,
                       const range& rows     = { 0, -1 },
                       const AllocKind& alloc = {}) const {
        return base::pull(policy, { rows }, alloc);
    }

    T* pull(array<data_t>& block, const range& rows = { 0, -1 }) const {
        return pull(detail::cpu_dispatch_default{}, block, rows);
    }

    template <typename Policy, typename AllocKind=default_parameter_tag>
    T* pull(const Policy& policy,
            array<data_t>& block,
            const range& rows     = { 0, -1 },
            const AllocKind& alloc = {}) const {
        return base::pull(policy, block, { rows }, alloc);
    }

    template <typename Q = T>
    std::enable_if_t<sizeof(Q) && !is_readonly> push(const array<data_t>& block,
                                                     const range& rows = { 0, -1 }) {
        push(detail::cpu_dispatch_default{}, block, rows);
    }

    template <typename Policy, typename Q = T>
    std::enable_if_t<sizeof(Q) && !is_readonly> push(const Policy& policy,
                                                     const array<data_t>& block,
                                                     const range& rows = { 0, -1 }) {
        base::push(policy, block, { rows });
    }
};

template <typename T>
class column_accessor : private detail::accessor_base<T, detail::column_values_block> {
    using base = detail::accessor_base<T, detail::column_values_block>;

public:
    using data_t                      = typename base::data_t;
    static constexpr bool is_readonly = base::is_readonly;

public:
    template <typename K,
              typename = std::enable_if_t<is_readonly && (std::is_base_of_v<table, K> ||
                                                          std::is_base_of_v<table_builder, K>)>>
    column_accessor(const K& obj) : base(obj) {}

    column_accessor(const table_builder& b) : base(b) {}

    array<data_t> pull(std::int64_t column_index, const range& rows = { 0, -1 }) const {
        return pull(detail::cpu_dispatch_default{}, column_index, rows);
    }

    template <typename Policy, typename AllocKind=default_parameter_tag>
    array<data_t> pull(const Policy& policy,
                       std::int64_t column_index,
                       const range& rows     = { 0, -1 },
                       const AllocKind& alloc = {}) const {
        return base::pull(policy, { column_index, rows }, alloc);
    }

    T* pull(array<data_t>& block, std::int64_t column_index, const range& rows = { 0, -1 }) const {
        return base::pull(detail::cpu_dispatch_default{}, block, column_index, rows);
    }

    template <typename Policy, typename AllocKind=default_parameter_tag>
    T* pull(const Policy& policy,
            array<data_t>& block,
            std::int64_t column_index,
            const range& rows     = { 0, -1 },
            const AllocKind& alloc = {}) const {
        return base::pull(policy, block, { column_index, rows }, alloc);
    }

    template <typename Q = T>
    std::enable_if_t<sizeof(Q) && !is_readonly> push(const array<data_t>& block,
                                                     std::int64_t column_index,
                                                     const range& rows = { 0, -1 }) {
        push(detail::cpu_dispatch_default{}, block, column_index, rows);
    }

    template <typename Policy, typename Q = T>
    std::enable_if_t<sizeof(Q) && !is_readonly> push(const Policy& policy,
                                                     const array<data_t>& block,
                                                     std::int64_t column_index,
                                                     const range& rows = { 0, -1 }) {
        base::push(policy, block, { column_index, rows });
    }
};

} // namespace oneapi::dal
