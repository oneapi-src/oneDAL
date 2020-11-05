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
#include "oneapi/dal/detail/policy.hpp"

namespace oneapi::dal::detail {
namespace v1 {

template <typename T>
inline T* malloc(const default_host_policy&, std::int64_t count) {
    return new T[count];
}

template <typename T>
inline void free(const default_host_policy&, T* pointer) {
    delete[] pointer;
}

inline void memset(const default_host_policy&, void* dest, std::int32_t value, std::int64_t size) {
    // TODO: is not safe since std::memset accepts size as size_t
    // TODO: can be optimized in future
    std::memset(dest, value, size);
}

inline void memcpy(const default_host_policy&, void* dest, const void* src, std::int64_t size) {
    // TODO: is not safe since std::memset accepts size as size_t
    // TODO: can be optimized in future
    std::memcpy(dest, src, size);
}

template <typename T>
inline void fill(const default_host_policy&, T* dest, std::int64_t count, const T& value) {
    // TODO: can be optimized in future
    for (std::int64_t i = 0; i < count; i++) {
        dest[i] = value;
    }
}

template <typename T>
class host_allocator {
public:
    T* allocate(std::int64_t n) const {
        return malloc<T>(default_host_policy{}, n);
    }
    void deallocate(T* p, std::int64_t n) const {
        free(default_host_policy{}, p);
    }
};

} // namespace v1

using v1::malloc;
using v1::free;
using v1::memset;
using v1::memcpy;
using v1::fill;
using v1::host_allocator;

} // namespace oneapi::dal::detail
