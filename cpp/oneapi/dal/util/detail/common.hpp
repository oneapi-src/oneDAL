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

template <typename Data>
inline void throw_if_sum_overflow(const Data& first, const Data& second) {
    volatile Data tmp = first + second;
    tmp -= second;
    if (tmp != second) {
        throw range_error("overflow found in sum of two values");
    }
}

} // namespace oneapi::dal::detail
