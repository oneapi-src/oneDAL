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

#include <cstring>

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/dispatcher.hpp"
#include "oneapi/dal/policy.hpp"

namespace oneapi::dal::detail {

struct host_only_alloc {
    host_only_alloc() {}
    host_only_alloc(const default_parameter_tag&) {}
};

template <typename T>
inline T* malloc(const cpu_dispatch_default&, std::int64_t count, const host_only_alloc& kind) {
    return new T[count];
}

template <typename T>
inline void free(const cpu_dispatch_default&, T* pointer) {
    delete[] pointer;
}

inline void memset(const cpu_dispatch_default&, void* dest, std::int32_t value, std::int64_t size) {
    // TODO: is not safe since std::memset accepts size as size_t
    // TODO: can be optimized in future
    std::memset(dest, value, size);
}

inline void memcpy(const cpu_dispatch_default&, void* dest, const void* src, std::int64_t size) {
    // TODO: is not safe since std::memset accepts size as size_t
    // TODO: can be optimized in future
    std::memcpy(dest, src, size);
}

template <typename T>
inline void fill(const cpu_dispatch_default&, T* dest, std::int64_t count, const T& value) {
    // TODO: can be optimized in future
    for (std::int64_t i = 0; i < count; i++) {
        dest[i] = value;
    }
}

} // namespace oneapi::dal::detail
