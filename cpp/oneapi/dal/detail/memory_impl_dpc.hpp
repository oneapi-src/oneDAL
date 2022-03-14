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

#include "oneapi/dal/detail/policy.hpp"

namespace oneapi::dal::detail {
namespace v1 {

ONEDAL_EXPORT void* malloc(const data_parallel_policy&, std::size_t size, const sycl::usm::alloc&);
ONEDAL_EXPORT void free(const data_parallel_policy&, void* pointer);
ONEDAL_EXPORT void memset(const data_parallel_policy& policy,
                          void* dest,
                          std::int32_t value,
                          std::int64_t size);
ONEDAL_EXPORT void memcpy(const data_parallel_policy& policy,
                          void* dest,
                          const void* src,
                          std::int64_t size);
ONEDAL_EXPORT void memcpy_usm2host(const data_parallel_policy& policy,
                                   void* dest_host,
                                   const void* src_usm,
                                   std::int64_t size);
ONEDAL_EXPORT void memcpy_host2usm(const data_parallel_policy& policy,
                                   void* dest_usm,
                                   const void* src_host,
                                   std::int64_t size);
ONEDAL_EXPORT bool is_known_usm_pointer_type(const data_parallel_policy& policy,
                                             const void* pointer);

template <typename T>
inline T* malloc(const data_parallel_policy& policy,
                 std::int64_t count,
                 const sycl::usm::alloc& alloc) {
    ONEDAL_ASSERT_MUL_OVERFLOW(std::size_t, sizeof(T), count);
    const std::size_t bytes_count = sizeof(T) * count;
    return static_cast<T*>(malloc(policy, bytes_count, alloc));
}

template <typename T>
inline void free(const data_parallel_policy& policy, T* pointer) {
    using mutable_t = std::remove_const_t<T>;
    free(policy, reinterpret_cast<void*>(const_cast<mutable_t*>(pointer)));
}

template <typename T>
inline void fill(const data_parallel_policy& policy, T* dest, std::int64_t count, const T& value) {
    ONEDAL_ASSERT(is_known_usm_pointer_type(policy, dest));
    ONEDAL_ASSERT(count > 0);

    auto& queue = policy.get_queue();
    queue.fill(dest, value, count).wait_and_throw();
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
using v1::memcpy_usm2host;
using v1::memcpy_host2usm;
using v1::is_known_usm_pointer_type;
using v1::fill;
using v1::data_parallel_allocator;

} // namespace oneapi::dal::detail

#endif // ONEDAL_DATA_PARALLEL
