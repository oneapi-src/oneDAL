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

#include "oneapi/dal/data/detail/access_iface.hpp"
#include "oneapi/dal/data/detail/access_iface_type_traits.hpp"

#include <stdexcept> // TODO: change by oneDAL exceptions

namespace oneapi::dal::detail {

template <typename T>
class host_access_wrapper_impl {
public:
    using policy_t = host_seq_policy;
    using alloc_kind_t = host_only_alloc;

public:
    host_access_wrapper_impl(T& obj)
        : obj_(obj) {}

    template <typename Block>
    void pull_rows(const policy_t&,
                   Block& block, const row_block& index,
                   const alloc_kind_t&) const {
        if constexpr (has_pull_rows_host<T, typename Block::data_t>::value) {
            obj_.pull_rows(block, index.rows);
        } else {
            throw std::runtime_error("pulling rows is not supported");
        }
    }

    template <typename Block>
    void pull_column(const policy_t&,
                     Block& block, const column_values_block& index,
                     const alloc_kind_t&) const {
        if constexpr (has_pull_column_host<T, typename Block::data_t>::value) {
            obj_.pull_column(block, index.column_index, index.rows);
        } else {
            throw std::runtime_error("pulling column is not supported");
        }
    }

    template <typename Block>
    void push_rows(const policy_t&,
                   const Block& block, const row_block& index,
                   const alloc_kind_t&) {
        if constexpr (has_push_rows_host<T, typename Block::data_t>::value) {
            obj_.push_rows(block, index.rows);
        } else {
            throw std::runtime_error("pushing rows is not supported");
        }
    }

    template <typename Block>
    void push_column(const policy_t&,
                     const Block& block, const column_values_block& index,
                     const alloc_kind_t&) {
        if constexpr (has_push_column_host<T, typename Block::data_t>::value) {
            obj_.push_column(block, index.column_index, index.rows);
        } else {
            throw std::runtime_error("pushing column is not supported");
        }
    }

private:
    T& obj_;
};

template <typename AccessIface, typename AccessImpl>
class access_wrapper: public AccessIface,
                      public base {
public:
    using policy_t     = typename AccessImpl::policy_t;
    using alloc_kind_t = typename AccessImpl::alloc_kind_t;
    using array_f32 = array<float>;
    using array_f64 = array<double>;
    using array_i32 = array<std::int32_t>;

    access_wrapper(const AccessImpl& impl)
        : impl_(impl) {}

    virtual void pull(const policy_t& policy,
                      array_f32& block, const row_block& index,
                      const alloc_kind_t& kind) const override {
        impl_.pull_rows(policy, block, index, kind);
    }
    virtual void pull(const policy_t& policy,
                      array_f64& block, const row_block& index,
                      const alloc_kind_t& kind) const override {
        impl_.pull_rows(policy, block, index, kind);
    }
    virtual void pull(const policy_t& policy,
                      array_i32& block, const row_block& index,
                      const alloc_kind_t& kind) const override {
        impl_.pull_rows(policy, block, index, kind);
    }

    virtual void push(const policy_t& policy,
                      const array_f32& block, const row_block& index,
                      const alloc_kind_t& kind) override {
        impl_.push_rows(policy, block, index, kind);
    }
    virtual void push(const policy_t& policy,
                      const array_f64& block, const row_block& index,
                      const alloc_kind_t& kind) override {
        impl_.push_rows(policy, block, index, kind);
    }
    virtual void push(const policy_t& policy,
                      const array_i32& block, const row_block& index,
                      const alloc_kind_t& kind) override {
        impl_.push_rows(policy, block, index, kind);
    }

    virtual void pull(const policy_t& policy,
                      array_f32& block, const column_values_block& index,
                      const alloc_kind_t& kind) const override {
        impl_.pull_column(policy, block, index, kind);
    }
    virtual void pull(const policy_t& policy,
                      array_f64& block, const column_values_block& index,
                      const alloc_kind_t& kind) const override {
        impl_.pull_column(policy, block, index, kind);
    }
    virtual void pull(const policy_t& policy,
                      array_i32& block, const column_values_block& index,
                      const alloc_kind_t& kind) const override {
        impl_.pull_column(policy, block, index, kind);
    }

    virtual void push(const policy_t& policy,
                      const array_f32& block, const column_values_block& index,
                      const alloc_kind_t& kind) override {
        impl_.push_column(policy, block, index, kind);
    }
    virtual void push(const policy_t& policy,
                      const array_f64& block, const column_values_block& index,
                      const alloc_kind_t& kind) override {
        impl_.push_column(policy, block, index, kind);
    }
    virtual void push(const policy_t& policy,
                      const array_i32& block, const column_values_block& index,
                      const alloc_kind_t& kind) override {
        impl_.push_column(policy, block, index, kind);
    }
private:
    AccessImpl impl_;
};

template <typename T>
using host_access_wrapper = access_wrapper<host_access_iface, host_access_wrapper_impl<T>>;

} // namespace oneapi::dal::detail
