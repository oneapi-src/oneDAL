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

#include "oneapi/dal/util/detail/type_traits.hpp"

namespace oneapi::dal {
namespace v1 {

template <typename T>
inline constexpr bool check_mask_flag(T mask, T flag) {
    using U = std::underlying_type_t<T>;
    return (static_cast<U>(mask) & static_cast<U>(flag)) > 0;
}

template <typename T>
inline constexpr T bitwise_and(T lhs_mask, T rhs_mask) {
    using U = std::underlying_type_t<T>;
    return static_cast<T>(static_cast<U>(lhs_mask) & static_cast<U>(rhs_mask));
}

template <typename T>
inline constexpr T bitwise_or(T lhs_mask, T rhs_mask) {
    using U = std::underlying_type_t<T>;
    return static_cast<T>(static_cast<U>(lhs_mask) | static_cast<U>(rhs_mask));
}

} // namespace v1

using v1::check_mask_flag;
using v1::bitwise_and;
using v1::bitwise_or;

} // namespace oneapi::dal
