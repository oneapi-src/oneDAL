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

#include "oneapi/dal/table/detail/csr.hpp"
#include "oneapi/dal/table/backend/csr_table_impl.hpp"

namespace oneapi::dal::detail {
namespace v1 {

std::int64_t csr_table::kind() {
    return 10;
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

const std::int64_t* csr_table::get_row_indices() const {
    const auto& impl = detail::cast_impl<detail::csr_table_iface>(*this);
    return impl.get_row_indices().get_data();
}

void csr_table::init_impl(std::int64_t column_count,
                          std::int64_t row_count,
                          const dal::array<byte_t>& data,
                          const dal::array<std::int64_t>& column_indices,
                          const dal::array<std::int64_t>& row_indices,
                          const data_type& dtype,
                          csr_indexing indexing) {
    table::init_impl(new backend::csr_table_impl(column_count,
                                                 row_count,
                                                 data,
                                                 column_indices,
                                                 row_indices,
                                                 dtype,
                                                 indexing));
}

} // namespace v1
} // namespace oneapi::dal::detail
