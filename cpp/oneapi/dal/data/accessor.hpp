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

template <typename T, typename BlockIndexType>
struct accessor_base {
    using data_t                      = std::remove_const_t<T>;
    using pull_ptr_host_t = host_access_iface::pull_ptr_t<data_t, BlockIndexType>;
    using push_ptr_host_t = host_access_iface::push_ptr_t<data_t, BlockIndexType>;

    static constexpr bool is_readonly = std::is_const_v<T>;

    template <typename K>
    static pull_ptr_host_t get_pull_access_ptr_host(const K& obj) {
        auto access_iface = get_impl<accessible_iface>(obj).get_host_access_iface();
        using pull_access_ptr = pull_access_ptr<data_t, BlockIndexType, decltype(access_iface)>;

        return pull_access_ptr{}.get_value(access_iface);
    }

    template <typename K>
    static push_ptr_host_t get_push_access_ptr_host(const K& obj) {
        auto access_iface = get_impl<accessible_iface>(obj).get_host_access_iface();
        using push_access_ptr = push_access_ptr<data_t, BlockIndexType, decltype(access_iface)>;

        return push_access_ptr{}.get_value(access_iface);
    }

    template <typename K>
    void init_access_pointers(const K& obj) {
        pull_host_ = get_pull_access_ptr_host(obj);
        if (pull_host_ == nullptr) {
            throw std::runtime_error("object does not support pull() operator");
        }

        if constexpr (!is_readonly) {
            push_host_ = get_push_access_ptr_host(obj);

            if (push_host_ == nullptr) {
                throw std::runtime_error("object does not support push() operator");
            }
        }
    }

    pull_ptr_host_t pull_host_;
    push_ptr_host_t push_host_;
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
              typename = std::enable_if_t<is_readonly &&
                    (std::is_base_of_v<table, K> || std::is_base_of_v<table_builder, K>)>>
    row_accessor(const K& obj) {
        base::init_access_pointers(obj);
    }

    row_accessor(const table_builder& b) {
        base::init_access_pointers(b);
    }

    array<data_t> pull(const range& rows = { 0, -1 }) const {
        array<data_t> block;
        base::pull_host_->pull(detail::host_seq_policy{}, block, {rows}, detail::host_only_alloc{});
        return block;
    }

    T* pull(array<data_t>& block, const range& rows = { 0, -1 }) const {
        base::pull_host_->pull(detail::host_seq_policy{}, block, {rows}, detail::host_only_alloc{});
        if constexpr (is_readonly) {
            return block.get_data();
        }
        else {
            return block.get_mutable_data();
        }
    }

    template <typename Q = T>
    std::enable_if_t<sizeof(Q) && !is_readonly> push(const array<data_t>& block,
                                                     const range& rows = { 0, -1 }) {
        base::push_host_->push(detail::host_seq_policy{}, block, {rows}, detail::host_only_alloc{});
    }
};

template <typename T>
class column_accessor : private detail::accessor_base<T, detail::column_values_block>{
    using base = detail::accessor_base<T, detail::column_values_block>;

public:
    using data_t                      = typename base::data_t;
    static constexpr bool is_readonly = base::is_readonly;

public:
    template <typename K,
              typename = std::enable_if_t<is_readonly &&
                    (std::is_base_of_v<table, K> || std::is_base_of_v<table_builder, K>)>>
    column_accessor(const K& obj) {
        base::init_access_pointers(obj);
    }

    column_accessor(const table_builder& b) {
        base::init_access_pointers(b);
    }

    array<data_t> pull(std::int64_t column_index, const range& rows = { 0, -1 }) const {
        array<data_t> block;
        base::pull_host_->pull(detail::host_seq_policy{},
                               block, {column_index, rows},
                               detail::host_only_alloc{});
        return block;
    }

    T* pull(array<data_t>& block, std::int64_t column_index, const range& rows = { 0, -1 }) const {
        base::pull_host_->pull(detail::host_seq_policy{},
                               block, {column_index, rows},
                               detail::host_only_alloc{});
        if constexpr (is_readonly) {
            return block.get_data();
        }
        else {
            return block.get_mutable_data();
        }
    }

    template <typename Q = T>
    std::enable_if_t<sizeof(Q) && !is_readonly> push(const array<data_t>& block,
                                                     std::int64_t column_index,
                                                     const range& rows = { 0, -1 }) {
        base::push_host_->push(detail::host_seq_policy{},
                               block, {column_index, rows},
                               detail::host_only_alloc{});
    }
};

} // namespace oneapi::dal
