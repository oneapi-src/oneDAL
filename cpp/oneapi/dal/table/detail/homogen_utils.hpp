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

#include "oneapi/dal/table/homogen.hpp"

namespace oneapi::dal::detail {
namespace v1 {

/// Retrieves original data from homogen table including information about
/// mutability. If the table was created from custom implementation, returns
/// empty array.
///
/// @param[in] t Table to extract the data from
/// @return      Original array of bytes held in the table.
///              If the table was created from mutable data object,
///              this array contains mutable data.
inline dal::array<byte_t> get_original_data(const homogen_table& t) {
    return detail::cast_impl<detail::homogen_table_iface>(t).get_data();
}

} // namespace v1

using v1::get_original_data;

} // namespace oneapi::dal::detail
