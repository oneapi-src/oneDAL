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

#ifdef ONEDAL_DATA_PARALLEL
#include <cstring>

#include "oneapi/dal/detail/common_dpc.hpp"
#include "oneapi/dal/detail/policy.hpp"

namespace oneapi::dal::detail {
namespace v1 {

template <typename T>
inline T* malloc(const data_parallel_policy& policy,
                 std::int64_t count,
                 const sycl::usm::alloc& alloc) {
    auto& queue = policy.get_queue();
    auto device = queue.get_device();
    auto context = queue.get_context();
    // TODO: is not safe since sycl::memset accepts count as size_t
    return sycl::malloc<T>(count, device, context, alloc);
}

template <typename T>
inline void free(const data_parallel_policy& policy, T* pointer) {
    using mutable_t = std::remove_const_t<T>;
    sycl::free(const_cast<mutable_t*>(pointer), policy.get_queue());
}

inline void memset(const data_parallel_policy& policy,
                   void* dest,
                   std::int32_t value,
                   std::int64_t size) {
    // TODO: is not safe since queue.memset accepts size as size_t
    auto event = policy.get_queue().memset(dest, value, size);
    event.wait();
}

inline void memcpy(const data_parallel_policy& policy,
                   void* dest,
                   const void* src,
                   std::int64_t size) {
    // TODO: is not safe since queue.memcpy accepts size as size_t
    auto event = policy.get_queue().memcpy(dest, src, size);
    event.wait();
}

template <typename T>
inline void fill(const data_parallel_policy& policy, T* dest, std::int64_t count, const T& value) {
    // TODO: can be optimized in future
    auto& queue = policy.get_queue();
    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<class oneapi_dal_memory_fill>(sycl::range<1>(count), [=](sycl::id<1> idx) {
            dest[idx[0]] = value;
        });
    });

    event.wait();
}

template <typename T>
class data_parallel_allocator {
public:
    data_parallel_allocator(const data_parallel_policy& policy,
                            sycl::usm::alloc kind = sycl::usm::alloc::shared)
            : policy_(policy),
              kind_(kind) {}

    T* allocate(std::int64_t n) const {
        return malloc<T>(policy_, n, kind_);
    }

    void deallocate(T* p, std::int64_t n) const {
        free(policy_, p);
    }

    sycl::usm::alloc get_kind() const {
        return kind_;
    }

private:
    data_parallel_policy policy_;
    sycl::usm::alloc kind_;
};

} // namespace v1

using v1::malloc;
using v1::free;
using v1::memset;
using v1::memcpy;
using v1::fill;
using v1::data_parallel_allocator;

} // namespace oneapi::dal::detail

#endif // ONEDAL_DATA_PARALLEL
