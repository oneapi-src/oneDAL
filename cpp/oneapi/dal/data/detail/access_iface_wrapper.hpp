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

namespace oneapi::dal::detail {

template <typename AccessIface, typename AccessImpl>
class access_iface_wrapper: public AccessIface,
                      public base {
public:
    using policy_t     = typename AccessImpl::policy_t;
    using alloc_kind_t = typename AccessImpl::alloc_kind_t;
    using array_f32 = array<float>;
    using array_f64 = array<double>;
    using array_i32 = array<std::int32_t>;

    access_iface_wrapper(const AccessImpl& impl)
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
                      const array_f32& block, const row_block& index) override {
        impl_.push_rows(policy, block, index);
    }
    virtual void push(const policy_t& policy,
                      const array_f64& block, const row_block& index) override {
        impl_.push_rows(policy, block, index);
    }
    virtual void push(const policy_t& policy,
                      const array_i32& block, const row_block& index) override {
        impl_.push_rows(policy, block, index);
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
                      const array_f32& block, const column_values_block& index) override {
        impl_.push_column(policy, block, index);
    }
    virtual void push(const policy_t& policy,
                      const array_f64& block, const column_values_block& index) override {
        impl_.push_column(policy, block, index);
    }
    virtual void push(const policy_t& policy,
                      const array_i32& block, const column_values_block& index) override {
        impl_.push_column(policy, block, index);
    }
private:
    AccessImpl impl_;
};

} // namespace oneapi::dal::detail
