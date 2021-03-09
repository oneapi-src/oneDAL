/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

namespace oneapi::dal::detail {
namespace v1 {

template <typename T>
class array_via_policy {
public:
    array_via_policy() = delete;

    template <typename... Args>
    static array<T> wrap(const default_host_policy& policy, Args&&... args) {
        return array<T>{ std::forward<Args>(args)... };
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename... Args>
    static array<T> wrap(const data_parallel_policy& policy, Args&&... args) {
        return array<T>{ policy.get_queue(), std::forward<Args>(args)... };
    }
#endif
};

} // namespace v1

using v1::array_via_policy;

} // namespace oneapi::dal::detail
