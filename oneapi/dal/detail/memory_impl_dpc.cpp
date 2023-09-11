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

#include "oneapi/dal/detail/memory_impl_dpc.hpp"
#include "oneapi/dal/backend/memory.hpp"

namespace oneapi::dal::detail {

void* malloc(const data_parallel_policy& policy, std::size_t size, const sycl::usm::alloc& alloc) {
    return backend::malloc(policy.get_queue(), size, alloc);
}

void free(const data_parallel_policy& policy, void* pointer) {
    return backend::free(policy.get_queue(), pointer);
}

void memset(const data_parallel_policy& policy, void* dest, std::int32_t value, std::int64_t size) {
    ONEDAL_ASSERT(is_known_usm_pointer_type(policy, dest));

    auto& queue = policy.get_queue();
    queue.memset(dest, value, detail::integral_cast<std::size_t>(size)).wait_and_throw();
}

void memcpy(const data_parallel_policy& policy, void* dest, const void* src, std::int64_t size) {
    auto& queue = policy.get_queue();
    backend::memcpy(queue, dest, src, integral_cast<std::size_t>(size)).wait_and_throw();
}

void memcpy_usm2host(const data_parallel_policy& policy,
                     void* dest_host,
                     const void* src_usm,
                     std::int64_t size) {
    auto& queue = policy.get_queue();
    backend::memcpy_usm2host(queue,
                             (byte_t*)dest_host,
                             (byte_t*)src_usm,
                             integral_cast<std::size_t>(size))
        .wait_and_throw();
}

void memcpy_host2usm(const data_parallel_policy& policy,
                     void* dest_usm,
                     const void* src_host,
                     std::int64_t size) {
    auto& queue = policy.get_queue();
    backend::memcpy_host2usm(queue,
                             (byte_t*)dest_usm,
                             (byte_t*)src_host,
                             integral_cast<std::size_t>(size))
        .wait_and_throw();
}

bool is_known_usm_pointer_type(const data_parallel_policy& policy, const void* pointer) {
    return backend::is_known_usm(policy.get_queue(), pointer);
}

} // namespace oneapi::dal::detail
