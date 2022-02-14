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

#include "oneapi/dal/backend/transfer.hpp"
#include <algorithm>

namespace oneapi::dal::backend {

sycl::event gather_device2host(sycl::queue& q,
                               void* dst_host,
                               const void* src_device,
                               std::int64_t block_count,
                               std::int64_t src_stride_in_bytes,
                               std::int64_t block_size_in_bytes,
                               const event_vector& deps) {
    ONEDAL_ASSERT(dst_host);
    ONEDAL_ASSERT(src_device);
    ONEDAL_ASSERT(block_count > 0);
    ONEDAL_ASSERT(src_stride_in_bytes > 0);
    ONEDAL_ASSERT(block_size_in_bytes > 0);
    ONEDAL_ASSERT(src_stride_in_bytes >= block_size_in_bytes);
    ONEDAL_ASSERT(is_known_usm(q, src_device));
    ONEDAL_ASSERT_MUL_OVERFLOW(std::int64_t, block_count, block_size_in_bytes);

    const auto gathered_device_unique =
        make_unique_usm_device(q, block_count * block_size_in_bytes);

    auto gather_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);

        const byte_t* src_byte = reinterpret_cast<const byte_t*>(src_device);
        byte_t* gathered_byte = reinterpret_cast<byte_t*>(gathered_device_unique.get());

        const std::int64_t required_local_size = 256;
        const std::int64_t local_size = std::min(down_pow2(block_count), required_local_size);
        const auto range = make_multiple_nd_range_1d(block_count, local_size);

        cgh.parallel_for(range, [=](sycl::nd_item<1> id) {
            const auto i = id.get_global_id();
            if (i < block_count) {
                // TODO: Unroll for optimization
                for (int j = 0; j < block_size_in_bytes; j++) {
                    gathered_byte[i * block_size_in_bytes + j] =
                        src_byte[i * src_stride_in_bytes + j];
                }
            }
        });
    });

    auto copy_event = memcpy_usm2host(q,
                                      dst_host,
                                      gathered_device_unique.get(),
                                      block_count * block_size_in_bytes,
                                      { gather_event });

    // We need to wait until gather kernel is completed to deallocate
    // `gathered_device_unique`
    copy_event.wait_and_throw();

    return sycl::event{};
}

sycl::event scatter_host2device(sycl::queue& q,
                                void* dst_device,
                                const void* src_host,
                                std::int64_t block_count,
                                std::int64_t dst_stride_in_bytes,
                                std::int64_t block_size_in_bytes,
                                const event_vector& deps) {
    ONEDAL_ASSERT(dst_device);
    ONEDAL_ASSERT(src_host);
    ONEDAL_ASSERT(block_count > 0);
    ONEDAL_ASSERT(dst_stride_in_bytes > 0);
    ONEDAL_ASSERT(block_size_in_bytes > 0);
    ONEDAL_ASSERT(dst_stride_in_bytes >= block_size_in_bytes);
    ONEDAL_ASSERT(is_known_usm(q, dst_device));
    ONEDAL_ASSERT_MUL_OVERFLOW(std::int64_t, block_count, block_size_in_bytes);

    const auto gathered_device_unique =
        make_unique_usm_device(q, block_count * block_size_in_bytes);

    auto copy_event = memcpy_host2usm(q,
                                      gathered_device_unique.get(),
                                      src_host,
                                      block_count * block_size_in_bytes,
                                      deps);

    auto scatter_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(copy_event);

        byte_t* gathered_byte = reinterpret_cast<byte_t*>(gathered_device_unique.get());
        byte_t* dst_byte = reinterpret_cast<byte_t*>(dst_device);

        const std::int64_t required_local_size = 256;
        const std::int64_t local_size = std::min(down_pow2(block_count), required_local_size);
        const auto range = make_multiple_nd_range_1d(block_count, local_size);

        cgh.parallel_for(range, [=](sycl::nd_item<1> id) {
            const auto i = id.get_global_id();
            if (i < block_count) {
                // TODO: Unroll for optimization
                for (int j = 0; j < block_size_in_bytes; j++) {
                    dst_byte[i * dst_stride_in_bytes + j] =
                        gathered_byte[i * block_size_in_bytes + j];
                }
            }
        });
    });

    // We need to wait until scatter kernel is completed to deallocate
    // `gathered_device_unique`
    scatter_event.wait_and_throw();

    return sycl::event{};
}

} // namespace oneapi::dal::backend
