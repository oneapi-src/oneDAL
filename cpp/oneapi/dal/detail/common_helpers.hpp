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

#include "oneapi/dal/common.hpp"

namespace dal::detail {

template <typename T>
constexpr auto make_data_type_impl() {
    if constexpr (std::is_same_v<std::int32_t, T>) {
        return data_type::int32;
    } else if constexpr (std::is_same_v<std::int64_t, T>) {
        return data_type::int64;
    } else if constexpr (std::is_same_v<std::uint32_t, T>) {
        return data_type::uint32;
    } else if constexpr (std::is_same_v<std::uint64_t, T>) {
        return data_type::uint64;
    } else if constexpr (std::is_same_v<float, T>) {
        return data_type::float32;
    } else if constexpr (std::is_same_v<double, T>) {
        return data_type::float64;
    } else {
        static_assert("unknown data type");
    }

    return data_type::float32; // should never come here
}

} // namespace dal::detail
