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

namespace oneapi::dal::detail {

template <typename T, typename DataType>
class access_pull_rows_wrapper_host :
    public access_pull_iface<DataType, row_block, host_seq_policy, host_only_alloc> {
public:
    access_pull_rows_wrapper_host(T& impl)
        : impl_(impl) {}

    virtual void pull(const host_seq_policy& policy,
                      array<DataType>& block, const row_block& index,
                      const host_only_alloc& kind) const override {
        impl_.pull_rows(block, index.rows);
    }

private:
    T& impl_;
};

template <typename T, typename DataType>
class access_pull_column_wrapper_host :
    public access_pull_iface<DataType, column_values_block, host_seq_policy, host_only_alloc> {
public:
    access_pull_column_wrapper_host(T& impl)
        : impl_(impl) {}

    virtual void pull(const host_seq_policy& policy,
                      array<DataType>& block, const column_values_block& index,
                      const host_only_alloc& kind) const override {
        impl_.pull_column(block, index.column_index, index.rows);
    }

private:
    T& impl_;
};

template <typename T, typename DataType>
class access_push_rows_wrapper_host :
    public access_push_iface<DataType, row_block, host_seq_policy, host_only_alloc> {
public:
    access_push_rows_wrapper_host(T& impl)
        : impl_(impl) {}

    virtual void push(const host_seq_policy& policy,
                      const array<DataType>& block, const row_block& index,
                      const host_only_alloc& kind) override {
        impl_.push_rows(block, index.rows);
    }

private:
    T& impl_;
};

template <typename T, typename DataType>
class access_push_column_wrapper_host :
    public access_push_iface<DataType, column_values_block, host_seq_policy, host_only_alloc> {
public:
    access_push_column_wrapper_host(T& impl)
        : impl_(impl) {}

    virtual void push(const host_seq_policy& policy,
                      const array<DataType>& block, const column_values_block& index,
                      const host_only_alloc& kind) override {
        impl_.push_column(block, index.column_index, index.rows);
    }

private:
    T& impl_;
};

template <typename DataType, typename IndexType>
using host_pull_ptr_t = host_access_iface::pull_ptr_t<DataType, IndexType>;

template <typename DataType, typename IndexType>
using host_push_ptr_t = host_access_iface::push_ptr_t<DataType, IndexType>;

template <typename T>
host_access_iface make_host_access_iface(T& obj) {
    host_access_iface iface;

    if constexpr (has_pull_rows_host<T, float>::value) {
        iface.pull_rows_float32 = host_pull_ptr_t<float, row_block>(
            new access_pull_rows_wrapper_host<T, float>(obj));
    }
    if constexpr (has_pull_rows_host<T, double>::value) {
        iface.pull_rows_float64 = host_pull_ptr_t<double, row_block>(
            new access_pull_rows_wrapper_host<T, double>(obj));
    }
    if constexpr (has_pull_rows_host<T, std::int32_t>::value) {
        iface.pull_rows_int32 = host_pull_ptr_t<std::int32_t, row_block>(
            new access_pull_rows_wrapper_host<T, std::int32_t>(obj));
    }
    if constexpr (has_pull_column_host<T, float>::value) {
        iface.pull_column_float32 = host_pull_ptr_t<float, column_values_block>(
            new access_pull_column_wrapper_host<T, float>(obj));
    }
    if constexpr (has_pull_column_host<T, double>::value) {
        iface.pull_column_float64 = host_pull_ptr_t<double, column_values_block>(
            new access_pull_column_wrapper_host<T, double>(obj));
    }
    if constexpr (has_pull_column_host<T, std::int32_t>::value) {
        iface.pull_column_int32 = host_pull_ptr_t<std::int32_t, column_values_block>(
            new access_pull_column_wrapper_host<T, std::int32_t>(obj));
    }

    if constexpr (has_push_rows_host<T, float>::value) {
        iface.push_rows_float32 = host_push_ptr_t<float, row_block>(
            new access_push_rows_wrapper_host<T, float>(obj));
    }
    if constexpr (has_push_rows_host<T, double>::value) {
        iface.push_rows_float64 = host_push_ptr_t<double, row_block>(
            new access_push_rows_wrapper_host<T, double>(obj));
    }
    if constexpr (has_push_rows_host<T, std::int32_t>::value) {
        iface.push_rows_int32 = host_push_ptr_t<std::int32_t, row_block>(
            new access_push_rows_wrapper_host<T, std::int32_t>(obj));
    }
    if constexpr (has_push_column_host<T, float>::value) {
        iface.push_column_float32 = host_push_ptr_t<float, column_values_block>(
            new access_push_column_wrapper_host<T, float>(obj));
    }
    if constexpr (has_push_column_host<T, double>::value) {
        iface.push_column_float64 = host_push_ptr_t<double, column_values_block>(
            new access_push_column_wrapper_host<T, double>(obj));
    }
    if constexpr (has_push_column_host<T, std::int32_t>::value) {
        iface.push_column_int32 = host_push_ptr_t<std::int32_t, column_values_block>(
            new access_push_column_wrapper_host<T, std::int32_t>(obj));
    }

    return iface;
}

} // namespace oneapi::dal::detail
