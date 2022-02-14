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

#include "oneapi/dal/array.hpp"

namespace oneapi::dal::backend {
namespace v1 {

/// Helper functionality for table accessors
///
/// @tparam T The type of data values in blocks returned by the accessor.
///           Should be const-qualified for read-only access.
template <typename T>
class accessor_impl {
public:
    using data_t = std::remove_const_t<T>;
    static constexpr bool is_readonly = std::is_const_v<T>;

    static T* get_block_data(const dal::array<data_t>& block) {
        if constexpr (is_readonly) {
            return block.get_data();
        }
        return block.get_mutable_data();
    }
};

} // namespace v1

using v1::accessor_impl;

} // namespace oneapi::dal::backend
