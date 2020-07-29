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

#include "oneapi/dal/detail/common_dpc.hpp"
#include "oneapi/dal/policy.hpp"

namespace oneapi::dal::detail {
#ifdef ONEAPI_DAL_DATA_PARALLEL

class data_parallel_alloc {
public:
    data_parallel_alloc()
        : data_parallel_alloc(default_parameter_tag{}) {}

    data_parallel_alloc(const default_parameter_tag&)
        : kind_(sycl::usm::alloc::shared) {}

    data_parallel_alloc(const sycl::usm::alloc kind)
        : kind_(kind) {}

    const sycl::usm::alloc& get_kind() const {
        return kind_;
    }

private:
    sycl::usm::alloc kind_;
};

template <typename T>
inline T* malloc(const data_parallel_policy& policy,
                 std::int64_t count,
                 const data_parallel_alloc& alloc = {}) {
    auto& queue  = policy.get_queue();
    auto device  = queue.get_device();
    auto context = queue.get_context();
    // TODO: is not safe since sycl::memset accepts count as size_t
    return sycl::malloc<T>(count, device, context, alloc.get_kind());
}

template <typename T>
inline void free(const data_parallel_policy& policy, T* pointer) {
    sycl::free(pointer, policy.get_queue());
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
    auto event  = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<class oneapi_dal_memory_fill>(sycl::range<1>(count), [=](sycl::id<1> idx) {
            dest[idx[0]] = value;
        });
    });

    event.wait();
}
#endif

} // namespace oneapi::dal::detail
