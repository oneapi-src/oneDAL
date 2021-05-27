/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::backend {

/// This class is needed for compatibility with the oneDAL 2021.1.
/// This should be removed in 2022.1.
class compat_host_accessor : public detail::access_iface_host {
public:
    explicit compat_host_accessor(detail::pull_rows_iface* pull_rows = nullptr,
                                  detail::pull_column_iface* pull_column = nullptr)
            : pull_rows_iface_(pull_rows),
              pull_column_iface_(pull_column) {}

    void pull(const detail::default_host_policy& policy,
              array_f32& block,
              const detail::row_block& rb,
              const alloc_f32& alloc) const override {
        template_pull(policy, block, rb, alloc);
    }

    void pull(const detail::default_host_policy& policy,
              array_f64& block,
              const detail::row_block& rb,
              const alloc_f64& alloc) const override {
        template_pull(policy, block, rb, alloc);
    }

    void pull(const detail::default_host_policy& policy,
              array_i32& block,
              const detail::row_block& rb,
              const alloc_i32& alloc) const override {
        template_pull(policy, block, rb, alloc);
    }

    void pull(const detail::default_host_policy& policy,
              array_f32& block,
              const detail::column_values_block& cb,
              const alloc_f32& alloc) const override {
        template_pull(policy, block, cb, alloc);
    }

    void pull(const detail::default_host_policy& policy,
              array_f64& block,
              const detail::column_values_block& cb,
              const alloc_f64& alloc) const override {
        template_pull(policy, block, cb, alloc);
    }

    void pull(const detail::default_host_policy& policy,
              array_i32& block,
              const detail::column_values_block& cb,
              const alloc_i32& alloc) const override {
        template_pull(policy, block, cb, alloc);
    }

    void push(const detail::default_host_policy&,
              const array_f32&,
              const detail::row_block&) override {
        throw internal_error{ detail::error_messages::pushing_rows_is_not_supported() };
    }

    void push(const detail::default_host_policy&,
              const array_f64&,
              const detail::row_block&) override {
        throw internal_error{ detail::error_messages::pushing_rows_is_not_supported() };
    }

    void push(const detail::default_host_policy&,
              const array_i32&,
              const detail::row_block&) override {
        throw internal_error{ detail::error_messages::pushing_rows_is_not_supported() };
    }

    void push(const detail::default_host_policy&,
              const array_f32&,
              const detail::column_values_block&) override {
        throw internal_error{ detail::error_messages::pushing_column_is_not_supported() };
    }

    void push(const detail::default_host_policy&,
              const array_f64&,
              const detail::column_values_block&) override {
        throw internal_error{ detail::error_messages::pushing_column_is_not_supported() };
    }

    void push(const detail::default_host_policy&,
              const array_i32&,
              const detail::column_values_block&) override {
        throw internal_error{ detail::error_messages::pushing_column_is_not_supported() };
    }

private:
    template <typename T>
    void template_pull(const detail::default_host_policy& policy,
                       dal::v1::array<T>& block,
                       const detail::row_block& rb,
                       const detail::host_allocator<T>&) const {
        if (pull_rows_iface_) {
            auto block_v2 = block.v2();
            pull_rows_iface_->pull_rows(policy, block_v2, rb.rows);
            block.reset(block_v2);
        }
    }

    template <typename T>
    void template_pull(const detail::default_host_policy& policy,
                       dal::v1::array<T>& block,
                       const detail::column_values_block& rb,
                       const detail::host_allocator<T>&) const {
        if (pull_rows_iface_) {
            auto block_v2 = block.v2();
            pull_column_iface_->pull_column(policy, block_v2, rb.column_index, rb.rows);
            block.reset(block_v2);
        }
    }

    detail::pull_rows_iface* pull_rows_iface_;
    detail::pull_column_iface* pull_column_iface_;
};

#ifdef ONEDAL_DATA_PARALLEL
/// This class is needed for compatibility with the oneDAL 2021.1.
/// This should be removed in 2022.1.
class compat_dpc_accessor : public detail::access_iface_dpc {
public:
    explicit compat_dpc_accessor(detail::pull_rows_iface* pull_rows = nullptr,
                                 detail::pull_column_iface* pull_column = nullptr)
            : pull_rows_iface_(pull_rows),
              pull_column_iface_(pull_column) {}

    void pull(const detail::data_parallel_policy& policy,
              array_f32& block,
              const detail::row_block& rb,
              const alloc_f32& alloc) const override {
        template_pull(policy, block, rb, alloc);
    }

    void pull(const detail::data_parallel_policy& policy,
              array_f64& block,
              const detail::row_block& rb,
              const alloc_f64& alloc) const override {
        template_pull(policy, block, rb, alloc);
    }

    void pull(const detail::data_parallel_policy& policy,
              array_i32& block,
              const detail::row_block& rb,
              const alloc_i32& alloc) const override {
        template_pull(policy, block, rb, alloc);
    }

    void pull(const detail::data_parallel_policy& policy,
              array_f32& block,
              const detail::column_values_block& cb,
              const alloc_f32& alloc) const override {
        template_pull(policy, block, cb, alloc);
    }

    void pull(const detail::data_parallel_policy& policy,
              array_f64& block,
              const detail::column_values_block& cb,
              const alloc_f64& alloc) const override {
        template_pull(policy, block, cb, alloc);
    }

    void pull(const detail::data_parallel_policy& policy,
              array_i32& block,
              const detail::column_values_block& cb,
              const alloc_i32& alloc) const override {
        template_pull(policy, block, cb, alloc);
    }

    void push(const detail::data_parallel_policy&,
              const array_f32&,
              const detail::row_block&) override {
        throw internal_error{ detail::error_messages::pushing_rows_is_not_supported_for_dpc() };
    }

    void push(const detail::data_parallel_policy&,
              const array_f64&,
              const detail::row_block&) override {
        throw internal_error{ detail::error_messages::pushing_rows_is_not_supported_for_dpc() };
    }

    void push(const detail::data_parallel_policy&,
              const array_i32&,
              const detail::row_block&) override {
        throw internal_error{ detail::error_messages::pushing_rows_is_not_supported_for_dpc() };
    }

    void push(const detail::data_parallel_policy&,
              const array_f32&,
              const detail::column_values_block&) override {
        throw internal_error{ detail::error_messages::pushing_column_is_not_supported_for_dpc() };
    }

    void push(const detail::data_parallel_policy&,
              const array_f64&,
              const detail::column_values_block&) override {
        throw internal_error{ detail::error_messages::pushing_column_is_not_supported_for_dpc() };
    }

    void push(const detail::data_parallel_policy&,
              const array_i32&,
              const detail::column_values_block&) override {
        throw internal_error{ detail::error_messages::pushing_column_is_not_supported_for_dpc() };
    }

private:
private:
    template <typename T>
    void template_pull(const detail::data_parallel_policy& policy,
                       dal::v1::array<T>& block,
                       const detail::row_block& rb,
                       const detail::data_parallel_allocator<T>& alloc) const {
        if (pull_rows_iface_) {
            auto block_v2 = block.v2();
            pull_rows_iface_->pull_rows(policy, block_v2, rb.rows, alloc.get_kind());
            block.reset(block_v2);
        }
    }

    template <typename T>
    void template_pull(const detail::data_parallel_policy& policy,
                       dal::v1::array<T>& block,
                       const detail::column_values_block& rb,
                       const detail::data_parallel_allocator<T>& alloc) const {
        if (pull_rows_iface_) {
            auto block_v2 = block.v2();
            pull_column_iface_->pull_column(policy,
                                            block_v2,
                                            rb.column_index,
                                            rb.rows,
                                            alloc.get_kind());
            block.reset(block_v2);
        }
    }

    detail::pull_rows_iface* pull_rows_iface_;
    detail::pull_column_iface* pull_column_iface_;
};
#endif

/// This class is needed for compatibility with the oneDAL 2021.1.
/// This should be removed in 2022.1.
class compat_accessor : public base {
public:
    compat_accessor() = default;

#ifdef ONEDAL_DATA_PARALLEL
    explicit compat_accessor(detail::pull_rows_iface* pull_rows,
                             detail::pull_column_iface* pull_column)
            : host_acc_(pull_rows, pull_column),
              dpc_acc_(pull_rows, pull_column) {}
#else
    explicit compat_accessor(detail::pull_rows_iface* pull_rows,
                             detail::pull_column_iface* pull_column)
            : host_acc_(pull_rows, pull_column) {}
#endif

    compat_host_accessor& get_host_accessor() {
        return host_acc_;
    }

#ifdef ONEDAL_DATA_PARALLEL
    compat_dpc_accessor& get_dpc_accessor() {
        return dpc_acc_;
    }
#endif

private:
    compat_host_accessor host_acc_;

#ifdef ONEDAL_DATA_PARALLEL
    compat_dpc_accessor dpc_acc_;
#endif
};

} // namespace oneapi::dal::backend
