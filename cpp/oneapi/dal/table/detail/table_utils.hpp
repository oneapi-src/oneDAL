/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/table/detail/table_iface.hpp"

namespace oneapi::dal::detail {
namespace v1 {

ONEDAL_EXPORT homogen_table_iface* get_homogen_table_iface_impl(table_iface* table);
ONEDAL_EXPORT pull_rows_iface* get_pull_rows_iface_impl(table_iface* table);
ONEDAL_EXPORT pull_column_iface* get_pull_column_iface_impl(table_iface* table);
ONEDAL_EXPORT pull_csr_block_iface* get_pull_csr_block_iface_impl(table_iface* table);

ONEDAL_EXPORT pull_rows_iface* get_pull_rows_iface_impl(table_builder_iface* table_builder);
ONEDAL_EXPORT push_rows_iface* get_push_rows_iface_impl(table_builder_iface* table_builder);
ONEDAL_EXPORT pull_column_iface* get_pull_column_iface_impl(table_builder_iface* table_builder);
ONEDAL_EXPORT push_column_iface* get_push_column_iface_impl(table_builder_iface* table_builder);

template <typename Object>
inline std::shared_ptr<homogen_table_iface> get_homogen_table_iface(Object&& obj) {
    const auto pimpl = pimpl_accessor{}.get_pimpl(std::forward<Object>(obj));
    auto homogen_iface_ptr = get_homogen_table_iface_impl(pimpl.get());
    return std::shared_ptr<homogen_table_iface>{ pimpl, homogen_iface_ptr };
}

template <typename Object>
inline std::shared_ptr<pull_rows_iface> get_pull_rows_iface(Object&& obj) {
    const auto pimpl = pimpl_accessor{}.get_pimpl(std::forward<Object>(obj));
    return std::shared_ptr<pull_rows_iface>{ pimpl, get_pull_rows_iface_impl(pimpl.get()) };
}

template <typename Object>
inline std::shared_ptr<push_rows_iface> get_push_rows_iface(Object&& obj) {
    const auto pimpl = pimpl_accessor{}.get_pimpl(std::forward<Object>(obj));
    return std::shared_ptr<push_rows_iface>{ pimpl, get_push_rows_iface_impl(pimpl.get()) };
}

template <typename Object>
inline std::shared_ptr<pull_column_iface> get_pull_column_iface(Object&& obj) {
    const auto pimpl = pimpl_accessor{}.get_pimpl(std::forward<Object>(obj));
    return std::shared_ptr<pull_column_iface>{ pimpl, get_pull_column_iface_impl(pimpl.get()) };
}

template <typename Object>
inline std::shared_ptr<push_column_iface> get_push_column_iface(Object&& obj) {
    const auto pimpl = pimpl_accessor{}.get_pimpl(std::forward<Object>(obj));
    return std::shared_ptr<push_column_iface>{ pimpl, get_push_column_iface_impl(pimpl.get()) };
}

template <typename Object>
inline std::shared_ptr<pull_csr_block_iface> get_pull_csr_block_iface(Object&& obj) {
    const auto pimpl = pimpl_accessor{}.get_pimpl(std::forward<Object>(obj));
    return std::shared_ptr<pull_csr_block_iface>{ pimpl,
                                                  get_pull_csr_block_iface_impl(pimpl.get()) };
}

} // namespace v1

using v1::get_homogen_table_iface;
using v1::get_pull_column_iface;
using v1::get_push_column_iface;
using v1::get_pull_rows_iface;
using v1::get_push_rows_iface;
using v1::get_pull_csr_block_iface;

} // namespace oneapi::dal::detail
