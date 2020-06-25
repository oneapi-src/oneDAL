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

#include "oneapi/dal/detail/common_dpc.hpp"

namespace oneapi::dal::detail {

template <typename T, typename Policy, typename AllocKind>
T* malloc(const Policy&, std::int64_t count, AllocKind);

template <typename T, typename Policy>
void free(const Policy&, T* pointer);

template <typename T, typename Policy>
void fill(const Policy&, T* dest, std::int64_t count, const T& value);

template <typename T>
T* malloc(const host_policy&, std::int64_t count, host_only_alloc) {
    return new T[count];
}

template <typename T>
void free(const host_policy&, T* pointer) {
    delete[] pointer;
}

template <typename T>
void fill(const host_policy&, T* dest, std::int64_t count, const T& value) {
    for (std::int64_t i = 0; i < count; i++) {
        dest[i] = value;
    }
}

#ifdef ONEAPI_DAL_DATA_PARALLEL
template <typename T>
T* malloc(const sycl::queue& queue, std::int64_t count, sycl::usm::alloc kind) {
    auto device = queue.get_device();
    auto context = queue.get_context();
    return sycl::malloc<T>(count, device, context, kind);
}

template <typename T>
void free(const sycl::queue& queue, T* pointer) {
    sycl::free(pointer, queue);
}

template <typename T>
void fill(const sycl::queue& queue, T* dest, std::int64_t count, const T& value) {
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
