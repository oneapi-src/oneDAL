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

#include <cstring>

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/policy.hpp"

namespace oneapi::dal::detail {
namespace v1 {

ONEDAL_EXPORT void* malloc_impl_host(const default_host_policy&, std::int64_t size);
ONEDAL_EXPORT void free_impl_host(const default_host_policy&, void* pointer);
ONEDAL_EXPORT void fill_impl_host(const default_host_policy&,
                                  void* dest,
                                  std::int64_t size,
                                  const void* pattern,
                                  std::int64_t pattern_size);

template <typename T>
T* malloc(const default_host_policy& policy, std::int64_t count) {
    const std::int64_t bytes_count = sizeof(T) * count;
    ONEDAL_ASSERT(bytes_count > count);
    return static_cast<T*>(malloc_impl_host(policy, bytes_count));
}

template <typename T>
void free(const default_host_policy& policy, T* pointer) {
    free_impl_host(policy, pointer);
}

ONEDAL_EXPORT void memset(const default_host_policy&,
                          void* dest,
                          std::int32_t value,
                          std::int64_t size);
ONEDAL_EXPORT void memcpy(const default_host_policy&,
                          void* dest,
                          const void* src,
                          std::int64_t size);

template <typename T>
void fill(const default_host_policy& policy, T* dest, std::int64_t count, const T& value) {
    const std::int64_t bytes_count = sizeof(T) * count;
    ONEDAL_ASSERT(bytes_count > count);
    fill_impl_host(policy, dest, bytes_count, &value, sizeof(T));
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
