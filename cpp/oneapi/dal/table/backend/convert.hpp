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

#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/backend/memory.hpp"

namespace oneapi::dal::backend {

void convert_vector(const detail::default_host_policy& policy,
                    const void* src,
                    void* dst,
                    data_type src_type,
                    data_type dst_type,
                    std::int64_t element_count);

void convert_vector(const detail::default_host_policy& policy,
                    const void* src,
                    void* dst,
                    data_type src_type,
                    data_type dst_type,
                    std::int64_t src_stride,
                    std::int64_t dst_stride,
                    std::int64_t element_count);

void convert_matrix(const detail::default_host_policy& policy,
                    const void* src,
                    void* dst,
                    data_type src_type,
                    data_type dst_type,
                    const std::int64_t src_row_stride,
                    const std::int64_t dst_row_stride,
                    const std::int64_t src_col_stride,
                    const std::int64_t dst_col_stride,
                    const std::int64_t dst_row_count,
                    const std::int64_t dst_col_count);

/// Shifts values in the array `arr` of type `arr_type` by the value pointed
/// to by the `shift` pointer.
/// Pseudocode:
///     for i in range 0, ..., element_count - 1:
///         arr[i] = arr[i] + shift[0]
void shift_array_values(const detail::default_host_policy& policy,
                        void* arr,
                        data_type arr_type,
                        const std::int64_t element_count,
                        const void* shift);

#ifdef ONEDAL_DATA_PARALLEL

void convert_vector(const detail::data_parallel_policy& policy,
                    const void* src,
                    void* dst,
                    data_type src_type,
                    data_type dst_type,
                    std::int64_t element_count);

void convert_vector(const detail::data_parallel_policy& policy,
                    const void* src,
                    void* dst,
                    data_type src_type,
                    data_type dst_type,
                    std::int64_t src_stride,
                    std::int64_t dst_stride,
                    std::int64_t element_count);

/// Converts matrix of `src_type` to matrix of `dst_type` and copy memory to
/// device in case 'dst' is accesible on the device.
void convert_matrix(const detail::data_parallel_policy& policy,
                    const void* src,
                    void* dst,
                    data_type src_type,
                    data_type dst_type,
                    const std::int64_t src_row_stride,
                    const std::int64_t dst_row_stride,
                    const std::int64_t src_col_stride,
                    const std::int64_t dst_col_stride,
                    const std::int64_t dst_row_count,
                    const std::int64_t dst_col_count);

/// Converts array of `src_type` to array of `dst_type` on device represented by
/// `q` assuming `src` and `dst` are accesible on the device.
sycl::event convert_vector_device2device(sycl::queue& q,
                                         const void* src,
                                         void* dst,
                                         data_type src_type,
                                         data_type dst_type,
                                         std::int64_t src_stride,
                                         std::int64_t dst_stride,
                                         std::int64_t element_count,
                                         const event_vector& deps = {});

/// Converts array of `src_type` to array of `dst_type` on device represented by
/// `q` assuming `src_device` is accesible on the device and `dst_host` is
/// accessible only on host.
sycl::event convert_vector_device2host(sycl::queue& q,
                                       const void* src_device,
                                       void* dst_host,
                                       data_type src_type,
                                       data_type dst_type,
                                       std::int64_t src_stride,
                                       std::int64_t dst_stride,
                                       std::int64_t element_count,
                                       const event_vector& deps = {});

/// Converts array of `src_type` to array of `dst_type` on device represented by
/// `q` assuming `src_host` is accesible only on host and `dst_device` is
/// accessible on the device.
sycl::event convert_vector_host2device(sycl::queue& q,
                                       const void* src_host,
                                       void* dst_device,
                                       data_type src_type,
                                       data_type dst_type,
                                       std::int64_t src_stride,
                                       std::int64_t dst_stride,
                                       std::int64_t element_count,
                                       const std::vector<sycl::event>& deps = {});

/// Shifts values in the array `arr` of type `arr_type` by the value pointed
/// to by the `shift` pointer.
/// Pseudocode:
///     for i in range 0, ..., element_count - 1:
///         arr[i] = arr[i] + shift[0]
/// Array `arr` can be allocated either on device or accessible only on host
sycl::event shift_array_values(const detail::data_parallel_policy& policy,
                               void* arr,
                               data_type arr_type,
                               const std::int64_t element_count,
                               const void* shift,
                               const event_vector& deps = {});

#endif

} // namespace oneapi::dal::backend
