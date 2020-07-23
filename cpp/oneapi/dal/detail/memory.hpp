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
#include "oneapi/dal/policy.hpp"

namespace oneapi::dal::detail {

struct host_only_alloc {};

template <typename T>
inline T* malloc(const host_policy&, std::int64_t count, host_only_alloc) {
    return new T[count];
}

#ifdef ONEAPI_DAL_DATA_PARALLEL
template <typename T>
inline T* malloc(const data_parallel_policy& policy, std::int64_t count, sycl::usm::alloc kind) {
    auto& queue  = policy.get_queue();
    auto device  = queue.get_device();
    auto context = queue.get_context();
    // TODO: is not safe since sycl::memset accepts count as size_t
    return sycl::malloc<T>(count, device, context, kind);
}
#endif

template <typename T>
inline void free(const host_policy&, T* pointer) {
    delete[] pointer;
}

#ifdef ONEAPI_DAL_DATA_PARALLEL
template <typename T>
inline void free(const data_parallel_policy& policy, T* pointer) {
    sycl::free(pointer, policy.get_queue());
}
#endif

inline void memset(const host_policy&, void* dest, std::int32_t value, std::int64_t size) {
    // TODO: is not safe since std::memset accepts size as size_t
    // TODO: can be optimized in future
    std::memset(dest, value, size);
}

#ifdef ONEAPI_DAL_DATA_PARALLEL
inline void memset(const data_parallel_policy& policy,
                   void* dest,
                   std::int32_t value,
                   std::int64_t size) {
    // TODO: is not safe since queue.memset accepts size as size_t
    auto event = policy.get_queue().memset(dest, value, size);
    event.wait();
}
#endif

inline void memcpy(const host_policy&, void* dest, const void* src, std::int64_t size) {
    // TODO: is not safe since std::memset accepts size as size_t
    // TODO: can be optimized in future
    std::memcpy(dest, src, size);
}

#ifdef ONEAPI_DAL_DATA_PARALLEL
inline void memcpy(const data_parallel_policy& policy,
                   void* dest,
                   const void* src,
                   std::int64_t size) {
    // TODO: is not safe since queue.memcpy accepts size as size_t
    auto event = policy.get_queue().memcpy(dest, src, size);
    event.wait();
}
#endif

template <typename T>
inline void fill(const host_policy&, T* dest, std::int64_t count, const T& value) {
    // TODO: can be optimized in future
    for (std::int64_t i = 0; i < count; i++) {
        dest[i] = value;
    }
}

#ifdef ONEAPI_DAL_DATA_PARALLEL
template <typename T>
inline void fill(const data_parallel_policy& policy, T* dest, std::int64_t count, const T& value) {
    // TODO: can be optimized in future
    auto& queue = policy.get_queue();
    auto event  = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<class oneapi_dal_memory_fill>(sycl::range<1>(count), [=](sycl::id<1> idx) {
            dest[idx[0]] = value;
        });
    });

    event.wait();
}
#endif

template <typename T, typename Policy>
class default_delete {
public:
    explicit default_delete(const Policy& policy) : policy_(policy) {}

    void operator()(T* data) {
        detail::free(policy_, data);
    }

private:
    std::remove_reference_t<Policy> policy_;
};

} // namespace oneapi::dal::detail
