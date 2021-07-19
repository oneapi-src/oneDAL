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

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/backend/memory.hpp"

namespace oneapi::dal::backend {

#ifdef ONEDAL_DATA_PARALLEL
template <typename T>
inline std::tuple<array<T>, sycl::event> to_device(sycl::queue& q, const array<T>& ary) {
    ONEDAL_ASSERT(ary.get_count() > 0);

    if (ary.get_queue().has_value()) {
        auto ary_q = ary.get_queue().value();
        check_if_same_context(q, ary_q);
        if (is_same_device(q, ary_q) && is_device_usm(ary)) {
            return { ary, sycl::event{} };
        }
        else {
            const auto ary_device = array<T>::empty(q, ary.get_count(), sycl::usm::alloc::device);
            const auto event =
                copy<T>(q, ary_device.get_mutable_data(), ary.get_data(), ary.get_count());
            return { ary_device, event };
        }
    }
    else {
        const auto ary_device = array<T>::empty(q, ary.get_count(), sycl::usm::alloc::device);
        const auto event =
            copy_host2usm<T>(q, ary_device.get_mutable_data(), ary.get_data(), ary.get_count());
        return { ary_device, event };
    }
}

template <typename T>
inline std::tuple<array<T>, sycl::event> to_host(const array<T>& ary) {
    ONEDAL_ASSERT(ary.get_count() > 0);

    if (!ary.get_queue().has_value()) {
        return { ary, sycl::event{} };
    }

    ONEDAL_ASSERT(ary.get_queue().has_value());
    auto q = ary.get_queue().value();

    const auto ary_host = array<T>::empty(q, ary.get_count());
    const auto event =
        copy_usm2host<T>(q, ary_host.get_mutable_data(), ary.get_data(), ary.get_count());
    return { ary_host, event };
}

template <typename T>
inline array<T> to_device_sync(sycl::queue& q, const array<T>& ary) {
    auto [ary_device, event] = to_device(q, ary);
    event.wait_and_throw();
    return ary_device;
}

template <typename T>
inline array<T> to_host_sync(const array<T>& ary) {
    auto [ary_host, event] = to_host(ary);
    event.wait_and_throw();
    return ary_host;
}

sycl::event gather_device2host(sycl::queue& q,
                               void* dst_host,
                               const void* src_device,
                               std::int64_t block_count,
                               std::int64_t src_stride_in_bytes,
                               std::int64_t block_size_in_bytes,
                               const event_vector& deps = {});

sycl::event scatter_host2device(sycl::queue& q,
                                void* dst_device,
                                const void* src_host,
                                std::int64_t block_count,
                                std::int64_t dst_stride_in_bytes,
                                std::int64_t block_size_in_bytes,
                                const event_vector& deps = {});
#endif

} // namespace oneapi::dal::backend
