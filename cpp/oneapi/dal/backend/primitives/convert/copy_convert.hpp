/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/backend/primitives/convert/common.hpp"

namespace oneapi::dal::backend::primitives {

void copy_convert(const detail::host_policy& policy,
                  const dal::array<data_type>& input_types,
                  const dal::array<dal::byte_t>& input_data,
                  const shape_t& input_shape,
                  data_type output_type,
                  dal::array<dal::byte_t>& output_data,
                  const shape_t& output_strides);

void copy_convert(const detail::host_policy& policy,
                  const dal::array<std::int64_t>& input_offsets,
                  const dal::array<data_type>& input_types,
                  const dal::array<dal::byte_t>& input_data,
                  const shape_t& input_shape,
                  data_type output_type,
                  dal::array<dal::byte_t>& output_data,
                  const shape_t& output_strides);

void copy_convert(detail::host_policy& policy,
                  const std::int64_t* input_offsets,
                  const data_type* input_types,
                  const dal::byte_t* input_data,
                  const shape_t& input_shape,
                  data_type output_type,
                  dal::byte_t* output_data,
                  const shape_t& output_strides);

template <typename CpuType>
void copy_convert(const std::int64_t* input_offsets,
                  const data_type* input_types,
                  const dal::byte_t* input_data,
                  const shape_t& input_shape,
                  data_type output_type,
                  dal::byte_t* output_data,
                  const shape_t& output_strides);

#ifdef ONEDAL_DATA_PARALLEL

template <typename Type>
inline sycl::event copy_convert(sycl::queue& queue,
                                const dal::array<data_type>& types,
                                const dal::array<dal::byte_t>& input_data,
                                const shape_t& input_shape,
                                dal::array<Type>& output_data,
                                const shape_t& output_strides,
                                const std::vector<sycl::event>& deps = {}) {
    constexpr auto output_type = detail::make_data_type<Type>();
    return copy_convert(queue,
                        input_types,
                        input_data,
                        input_shape,
                        output_type,
                        output_data,
                        output_strides,
                        deps);
}

sycl::event copy_convert(sycl::queue& queue,
                         const dal::array<data_type>& types,
                         const dal::array<dal::byte_t>& input_data,
                         const shape_t& input_shape,
                         data_type output_type,
                         dal::array<dal::byte_t>& output_data,
                         const shape_t& output_strides,
                         const std::vector<sycl::event>& deps = {});

#endif

} // namespace oneapi::dal::backend::primitives
