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

#ifdef ONEAPI_DAL_DATA_PARALLEL

#include "oneapi/dal/table/detail/access_iface_type_traits.hpp"
#include "oneapi/dal/table/detail/access_iface_wrapper.hpp"

#include <stdexcept> // TODO: change by oneDAL exceptions

namespace oneapi::dal::detail {

template <typename T>
class access_wrapper_impl_dpc {
public:
    using policy_t = data_parallel_policy;

    template <typename Y>
    using alloc_t = data_parallel_allocator<Y>;

public:
    access_wrapper_impl_dpc(T& obj) : obj_(obj) {}

    template <typename Block>
    void pull_rows(const policy_t& policy,
                   Block& block,
                   const row_block& index,
                   const alloc_t<typename Block::data_t>& alloc) const {
        if constexpr (has_pull_rows_dpc<T, typename Block::data_t>::value) {
            obj_.pull_rows(policy.get_queue(), block, index.rows, alloc.get_kind());
        }
        else {
            throw std::runtime_error("pulling rows is not supported for DPC++");
        }
    }

    template <typename Block>
    void pull_column(const policy_t& policy,
                     Block& block,
                     const column_values_block& index,
                     const alloc_t<typename Block::data_t>& alloc) const {
        if constexpr (has_pull_column_dpc<T, typename Block::data_t>::value) {
            obj_.pull_column(policy.get_queue(),
                             block,
                             index.column_index,
                             index.rows,
                             alloc.get_kind());
        }
        else {
            throw std::runtime_error("pulling column is not supported for DPC++");
        }
    }

    template <typename Block>
    void push_rows(const policy_t& policy, const Block& block, const row_block& index) {
        if constexpr (has_push_rows_dpc<T, typename Block::data_t>::value) {
            obj_.push_rows(policy.get_queue(), block, index.rows);
        }
        else {
            throw std::runtime_error("pushing rows is not supported for DPC++");
        }
    }

    template <typename Block>
    void push_column(const policy_t& policy, const Block& block, const column_values_block& index) {
        if constexpr (has_push_column_dpc<T, typename Block::data_t>::value) {
            obj_.push_column(policy.get_queue(), block, index.column_index, index.rows);
        }
        else {
            throw std::runtime_error("pushing column is not supported for DPC++");
        }
    }

private:
    T& obj_;
};

template <typename T>
using access_wrapper_dpc = access_iface_wrapper<access_iface_dpc, access_wrapper_impl_dpc<T>>;

} // namespace oneapi::dal::detail

#endif
