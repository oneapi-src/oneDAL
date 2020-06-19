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

#include <type_traits>

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common_helpers.hpp"

namespace oneapi::dal {

template <typename T>
constexpr data_type make_data_type() {
    return detail::make_data_type_impl<std::decay_t<T>>();
}

constexpr std::int64_t get_data_type_size(data_type t) {
    if (t == data_type::float32) {
        return sizeof(float);
    }
    else if (t == data_type::float64) {
        return sizeof(double);
    }
    else if (t == data_type::int32) {
        return sizeof(int32_t);
    }
    else if (t == data_type::int64) {
        return sizeof(int64_t);
    }
    else if (t == data_type::uint32) {
        return sizeof(uint32_t);
    }
    else if (t == data_type::uint64) {
        return sizeof(uint64_t);
    }
    return 0;
}

constexpr bool is_floating_point(data_type t) {
    if (t == data_type::bfloat16 ||
        t == data_type::float32 ||
        t == data_type::float64) {
            return true;
    } else {
        return false;
    }
}

template <data_type t>
struct integral_data_type { };

template <>
struct integral_data_type<data_type::float32> {
    using type = float;
};

template <>
struct integral_data_type<data_type::float64> {
    using type = double;
};

template <>
struct integral_data_type<data_type::int32> {
    using type = int32_t;
};

template <>
struct integral_data_type<data_type::int64> {
    using type = int64_t;
};

template <>
struct integral_data_type<data_type::uint32> {
    using type = uint32_t;
};

template <>
struct integral_data_type<data_type::uint64> {
    using type = uint64_t;
};

template <data_type t>
using integral_data_type_t = typename integral_data_type<t>::type;

} // namespace oneapi::dal
