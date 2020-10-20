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

#include "oneapi/dal/exceptions.hpp"

#include <type_traits>

namespace oneapi::dal::detail {

#ifdef ONEAPI_DAL_DEBUG
#include <cassert>
#define oneapi_dal_assert(cond) assert(cond)
#else
#define oneapi_dal_assert(cond)
#endif

template <typename Exception, typename... Args>
inline void throw_on_false(bool condition, Args&&... args) {
    static_assert(std::is_base_of_v<dal::exception, Exception>,
                  "Exception type shall be derived from dal::exception");

    if (!condition) {
        throw Exception(std::forward<Args>(args)...);
    }
}

template <typename Data>
inline void throw_on_sum_overflow(const Data& first, const Data& second) {
    static_assert(std::is_integral_v<Data>,
                  "The check requires integral operands");

    volatile Data tmp = first + second;
    tmp -= first;
    if (tmp != second) {
        throw range_error("overflow found in sum of two values");
    }
}

template <typename Data>
inline void throw_on_mul_overflow(const Data& first, const Data& second) {
    static_assert(std::is_integral_v<Data>,
                  "The check requires integral operands");

    if (first != 0 && second != 0) {
        volatile Data tmp = first * second;
        tmp /= first;
        if (tmp != second) {
            throw range_error("overflow found in multiplication of two values");
        }
    }
}

} // namespace oneapi::dal::detail
