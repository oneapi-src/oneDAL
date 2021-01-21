/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifdef ONEDAL_DATA_PARALLEL
#include "oneapi/dal/detail/memory_impl_dpc.hpp"

namespace oneapi::dal::detail::v1 {

void* malloc_impl_dpc(const data_parallel_policy& policy,
                      std::int64_t size,
                      const sycl::usm::alloc& alloc) {
    auto& queue = policy.get_queue();
    auto device = queue.get_device();
    auto context = queue.get_context();

    auto ptr = sycl::malloc(detail::integral_cast<std::size_t>(size), device, context, alloc);
    if (ptr == nullptr) {
        if (alloc == sycl::usm::alloc::shared || alloc == sycl::usm::alloc::host) {
            throw dal::host_bad_alloc();
        }
        else if (alloc = sycl::usm::alloc::device) {
            throw dal::device_bad_alloc();
        }
        else {
            throw dal::invalid_argument(detail::error_messages::unknown_usm_pointer_type());
        }
    }
    return ptr;
}

void free_impl_dpc(const data_parallel_policy& policy, void* pointer) {
    sycl::free(pointer, policy.get_queue());
}

void fill_impl_dpc(const data_parallel_policy& policy,
                   void* dest,
                   std::int64_t size,
                   const void* pattern,
                   std::int64_t pattern_size) {
    // TODO: can be optimized in future

    auto dest_bytes = static_cast<std::uint8_t*>(dest);
    auto pattern_bytes = static_cast<const std::uint8_t*>(pattern);

    if (size < 0) {
        throw dal::invalid_argument(detail::error_messages::dst_size_leq_zero());
    }

    if (pattern_size < 0) {
        throw dal::invalid_argument(detail::error_messages::src_size_leq_zero());
    }

    if (size % pattern_size != 0) {
        throw dal::invalid_argument(
            detail::error_messages::fill_pattern_size_divides_src_size_with_remainder());
    }

    auto& queue = policy.get_queue();
    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<class oneapi_dal_memory_fill>(sycl::range<1>(size), [=](sycl::id<1> idx) {
            dest_bytes[idx[0]] = pattern_bytes[idx[0] % pattern_size];
        });
    });

    event.wait();
}

void memset(const data_parallel_policy& policy, void* dest, std::int32_t value, std::int64_t size) {
    auto event = policy.get_queue().memset(dest, value, detail::integral_cast<std::size_t>(size));
    event.wait();
}

void memcpy(const data_parallel_policy& policy, void* dest, const void* src, std::int64_t size) {
    auto event = policy.get_queue().memcpy(dest, src, detail::integral_cast<std::size_t>(size));
    event.wait();
}

} // namespace oneapi::dal::detail::v1

#endif // ONEDAL_DATA_PARALLEL
