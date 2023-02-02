/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/table/backend/csr_table_impl.hpp"

namespace oneapi::dal {
namespace v1 {

std::int64_t csr_table::kind() {
    return 10;
}

std::int64_t csr_table::get_non_zero_count() const {
    const auto& impl = detail::cast_impl<detail::csr_table_iface>(*this);
    return impl.get_non_zero_count();
}
sparse_indexing csr_table::get_indexing() const {
    const auto& impl = detail::cast_impl<detail::csr_table_iface>(*this);
    return impl.get_indexing();
}

csr_table::csr_table() : csr_table(new backend::csr_table_impl{}) {}

const void* csr_table::get_data() const {
    const auto& impl = detail::cast_impl<detail::csr_table_iface>(*this);
    return impl.get_data().get_data();
}

const std::int64_t* csr_table::get_column_indices() const {
    const auto& impl = detail::cast_impl<detail::csr_table_iface>(*this);
    return impl.get_column_indices().get_data();
}

const std::int64_t* csr_table::get_row_offsets() const {
    const auto& impl = detail::cast_impl<detail::csr_table_iface>(*this);
    return impl.get_row_offsets().get_data();
}

template <typename Policy>
void csr_table::init_impl(const Policy& policy,
               const dal::array<byte_t>& data,
               const dal::array<std::int64_t>& column_indices,
               const dal::array<std::int64_t>& row_offsets,
               std::int64_t column_count,
               const data_type& dtype,
               sparse_indexing indexing) {
    table::init_impl(
        new backend::csr_table_impl{ data, column_indices, row_offsets, column_count, dtype, indexing });
}

#define INSTANTIATE(Policy)                                                                          \
    template ONEDAL_EXPORT void csr_table::init_impl(const Policy&,                                  \
                                                     const dal::array<byte_t>& data,                 \
                                                     const dal::array<std::int64_t>& column_indices, \
                                                     const dal::array<std::int64_t>& row_offsets,    \
                                                     std::int64_t column_count,                      \
                                                     const data_type& dtype,                         \
                                                     sparse_indexing indexing);

INSTANTIATE(detail::default_host_policy)

} // namespace v1
} // namespace oneapi::dal
