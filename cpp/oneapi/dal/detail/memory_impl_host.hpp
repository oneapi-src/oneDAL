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

ONEDAL_EXPORT void* malloc(const default_host_policy&, std::size_t size);
ONEDAL_EXPORT void* calloc(const default_host_policy&, std::size_t size);
ONEDAL_EXPORT void free(const default_host_policy&, void* pointer);
ONEDAL_EXPORT void memset(const default_host_policy&,
                          void* dest,
                          std::int32_t value,
                          std::int64_t size);
ONEDAL_EXPORT void memcpy(const default_host_policy&,
                          void* dest,
                          const void* src,
                          std::int64_t size);

template <typename T>
inline T* malloc(const default_host_policy& policy, std::int64_t count) {
    ONEDAL_ASSERT_MUL_OVERFLOW(std::size_t, sizeof(T), count);
    const std::size_t bytes_count = sizeof(T) * count;
    return static_cast<T*>(malloc(policy, bytes_count));
}

template <typename T>
inline T* calloc(const default_host_policy& policy, std::int64_t count) {
    ONEDAL_ASSERT_MUL_OVERFLOW(std::size_t, sizeof(T), count);
    const std::size_t bytes_count = sizeof(T) * count;
    return static_cast<T*>(calloc(policy, bytes_count));
}

template <typename T>
inline void free(const default_host_policy& policy, T* pointer) {
    using mutable_t = std::remove_const_t<T>;
    free(policy, reinterpret_cast<void*>(const_cast<mutable_t*>(pointer)));
}

template <typename T>
inline void fill(const default_host_policy& policy, T* dest, std::int64_t count, const T& value) {
    ONEDAL_ASSERT(dest != nullptr);
    ONEDAL_ASSERT(count > 0);

    for (std::int64_t i = 0l; i < count; ++i) {
        ::new (dest + i) T{ value };
    }
}

template <typename T>
class host_allocator {
public:
    host_allocator() {}
    host_allocator(const host_policy& policy) {}
    host_allocator(const default_host_policy& policy) {}

    T* allocate(std::int64_t n) const {
        return malloc<T>(default_host_policy{}, n);
    }
    void deallocate(T* p, std::int64_t n) const {
        free(default_host_policy{}, p);
    }
};

} // namespace v1

using v1::malloc;
using v1::calloc;
using v1::free;
using v1::fill;
using v1::memset;
using v1::memcpy;
using v1::host_allocator;

} // namespace oneapi::dal::detail
