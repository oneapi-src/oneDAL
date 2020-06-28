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

namespace oneapi::dal::detail {

#ifdef ONEAPI_DAL_DATA_PARALLEL
template <typename T>
inline T* malloc(sycl::queue& queue, std::int64_t count, sycl::usm::alloc kind) {
    auto device = queue.get_device();
    auto context = queue.get_context();
    // TODO: is not safe since sycl::memset accepts count as size_t
    return sycl::malloc<T>(count, device, context, kind);
}

template <typename T>
inline void free(sycl::queue& queue, T* pointer) {
    sycl::free(pointer, queue);
}

inline void memset(sycl::queue& queue, void* dest, std::int32_t value, std::int64_t size) {
    // TODO: is not safe since queue.memset accepts size as size_t
    auto event = queue.memset(dest, value, size);
    event.wait();
}

inline void memcpy(sycl::queue& queue, void* dest, const void* src, std::int64_t size) {
    // TODO: is not safe since queue.memcpy accepts size as size_t
    auto event = queue.memcpy(dest, src, size);
    event.wait();
}

template <typename T>
inline void fill(sycl::queue& queue, T* dest, std::int64_t count, const T& value) {
    // TODO: can be optimized in future
    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<class oneapi_dal_memory_fill>(sycl::range<1>(count),
        [=](sycl::id<1> idx) {
            dest[idx[0]] = value;
        });
    });

    event.wait();
}
#endif

} // namespace oneapi::dal::detail
