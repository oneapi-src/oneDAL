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

    template <typename Policy, typename AllocKind = default_parameter_tag>
    array<data_t> pull(const Policy& policy,
                       const BlockIndex& idx,
                       const AllocKind& alloc = {}) const {
        array<data_t> block;
        get_access(policy).pull(policy, block, idx, alloc);
        return block;
    }

    template <typename Policy, typename AllocKind = default_parameter_tag>
    T* pull(const Policy& policy,
            array<data_t>& block,
            const BlockIndex& idx,
            const AllocKind& alloc = {}) const {
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
    access_iface_host& get_access(const detail::default_host_policy&) {
        return host_access_;
    }
    const access_iface_host& get_access(const detail::default_host_policy&) const {
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
        return base::pull(detail::default_host_policy{}, { rows });
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    array<data_t> pull(sycl::queue& queue,
                       const range& rows             = { 0, -1 },
                       const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) const {
        return base::pull(detail::data_parallel_policy{queue}, { rows }, alloc);
    }
#endif

    T* pull(array<data_t>& block, const range& rows = { 0, -1 }) const {
        return base::pull(detail::default_host_policy{}, block, { rows });
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    T* pull(sycl::queue& queue,
            array<data_t>& block,
            const range& rows             = { 0, -1 },
            const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) const {
        return base::pull(detail::data_parallel_policy{queue}, block, { rows }, alloc);
    }
#endif

    template <typename Q = T>
    std::enable_if_t<sizeof(Q) && !is_readonly> push(const array<data_t>& block,
                                                     const range& rows = { 0, -1 }) {
        base::push(detail::default_host_policy{}, block, { rows });
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    template <typename Q = T>
    std::enable_if_t<sizeof(Q) && !is_readonly> push(sycl::queue& queue,
                                                     const array<data_t>& block,
                                                     const range& rows = { 0, -1 }) {
        base::push(detail::data_parallel_policy{queue}, block, { rows });
    }
#endif
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
        return base::pull(detail::default_host_policy{}, { column_index, rows });
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    array<data_t> pull(sycl::queue& queue,
                       std::int64_t column_index,
                       const range& rows             = { 0, -1 },
                       const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) const {
        return base::pull(detail::data_parallel_policy{queue}, { column_index, rows }, alloc);
    }
#endif

    T* pull(array<data_t>& block, std::int64_t column_index, const range& rows = { 0, -1 }) const {
        return base::pull(detail::default_host_policy{}, block, { column_index, rows });
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    T* pull(sycl::queue& queue,
            array<data_t>& block,
            std::int64_t column_index,
            const range& rows             = { 0, -1 },
            const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) const {
        return base::pull(detail::data_parallel_policy{queue}, block, { column_index, rows }, alloc);
    }
#endif

    template <typename Q = T>
    std::enable_if_t<sizeof(Q) && !is_readonly> push(const array<data_t>& block,
                                                     std::int64_t column_index,
                                                     const range& rows = { 0, -1 }) {
        base::push(detail::default_host_policy{}, block, { column_index, rows });
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    template <typename Q = T>
    std::enable_if_t<sizeof(Q) && !is_readonly> push(sycl::queue& queue,
                                                     const array<data_t>& block,
                                                     std::int64_t column_index,
                                                     const range& rows = { 0, -1 }) {
        base::push(detail::data_parallel_policy{queue}, block, { column_index, rows });
    }
#endif
};

} // namespace oneapi::dal
