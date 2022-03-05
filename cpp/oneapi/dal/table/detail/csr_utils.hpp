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

/// Retrieves original data from CSR table including information about
/// mutability. If the table was created from custom implementation, returns
/// empty array.
///
/// @param[in] t Table to extract the data from
/// @return      Original array of bytes held in the table.
///              If the table was created from mutable data object,
///              this array contains mutable data.
inline dal::array<byte_t> get_original_data(const table& t) {
    return detail::cast_impl<detail::csr_table_iface>(t).get_data();
}

/// Retrieves original column indices from CSR table including information about
/// mutability. If the table was created from custom implementation, returns
/// empty array.
///
/// @param[in] t Table to extract the column indices from
/// @return      Original array of int64_t held in the table.
///              If the table was created from mutable column indices object,
///              this array contains mutable column indices.
inline dal::array<std::int64_t> get_original_column_indices(const table& t) {
    return detail::cast_impl<detail::csr_table_iface>(t).get_column_indices();
}

/// Retrieves original row indices from CSR table including information about
/// mutability. If the table was created from custom implementation, returns
/// empty array.
///
/// @param[in] t Table to extract the row offsets from
/// @return      Original array of int64_t held in the table.
///              If the table was created from mutable row offsets object,
///              this array contains mutable row offsets.
inline dal::array<std::int64_t> get_original_row_offsets(const table& t) {
    return detail::cast_impl<detail::csr_table_iface>(t).get_row_offsets();
}

} // namespace v1

using v1::get_original_data;
using v1::get_original_column_indices;
using v1::get_original_row_offsets;

} // namespace oneapi::dal::detail
