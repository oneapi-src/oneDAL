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

#include "oneapi/dal/table/detail/access_iface_type_traits.hpp"
#include "oneapi/dal/table/detail/access_iface_wrapper.hpp"

namespace oneapi::dal::detail {
namespace v1 {

template <typename T>
class access_wrapper_impl_host {
public:
    using policy_t = default_host_policy;

    template <typename Y>
    using alloc_t = host_allocator<Y>;

public:
    access_wrapper_impl_host(T& obj) : obj_(obj) {}

    template <typename Block>
    void pull_rows(const policy_t&,
                   Block& block,
                   const row_block& index,
                   const alloc_t<typename Block::data_t>&) const {
        if constexpr (has_pull_rows_host<T, typename Block::data_t>::value) {
            obj_.pull_rows(block, index.rows);
        }
        else {
            throw std::runtime_error(dal::detail::error_messages::pulling_rows_is_not_supported());
        }
    }

    template <typename Block>
    void pull_column(const policy_t&,
                     Block& block,
                     const column_values_block& index,
                     const alloc_t<typename Block::data_t>&) const {
        if constexpr (has_pull_column_host<T, typename Block::data_t>::value) {
            obj_.pull_column(block, index.column_index, index.rows);
        }
        else {
            throw std::runtime_error(
                dal::detail::error_messages::pulling_column_is_not_supported());
        }
    }

    template <typename Block>
    void push_rows(const policy_t&, const Block& block, const row_block& index) {
        if constexpr (has_push_rows_host<T, typename Block::data_t>::value) {
            obj_.push_rows(block, index.rows);
        }
        else {
            throw std::runtime_error(dal::detail::error_messages::pushing_rows_is_not_supported());
        }
    }

    template <typename Block>
    void push_column(const policy_t&, const Block& block, const column_values_block& index) {
        if constexpr (has_push_column_host<T, typename Block::data_t>::value) {
            obj_.push_column(block, index.column_index, index.rows);
        }
        else {
            throw std::runtime_error(
                dal::detail::error_messages::pushing_column_is_not_supported());
        }
    }

private:
    T& obj_;
};

template <typename T>
using access_wrapper_host = access_iface_wrapper<access_iface_host, access_wrapper_impl_host<T>>;

} // namespace v1

using v1::access_wrapper_impl_host;
using v1::access_wrapper_host;

} // namespace oneapi::dal::detail
