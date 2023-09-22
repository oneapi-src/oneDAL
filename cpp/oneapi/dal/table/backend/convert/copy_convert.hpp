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

#include "oneapi/dal/table/backend/convert/common.hpp"
#include "oneapi/dal/table/backend/convert/common_convert.hpp"

namespace oneapi::dal::backend {

void copy_convert(const detail::host_policy& policy,
                  const dal::array<data_type>& input_types,
                  const dal::array<dal::byte_t>& input_data,
                  const shape_t& input_shape,
                  data_type output_type,
                  dal::array<dal::byte_t>& output_data,
                  const shape_t& output_strides);

void copy_convert(const detail::host_policy& policy,
                  const dal::array<const dal::byte_t*>& inp_pointers,
                  const dal::array<data_type>& inp_types,
                  const dal::array<std::int64_t>& inp_strides,
                  const dal::array<dal::byte_t*>& out_pointers,
                  const dal::array<data_type>& out_types,
                  const dal::array<std::int64_t>& out_strides,
                  const shape_t& shape);

void copy_convert(const detail::host_policy& policy,
                  const dal::byte_t* const* inp_pointers,
                  const data_type* inp_types,
                  const std::int64_t* inp_strides,
                  dal::byte_t* const* out_pointers,
                  const data_type* out_types,
                  const std::int64_t* out_strides,
                  const shape_t& shape);

template <typename CpuType>
void copy_convert(const detail::host_policy& policy,
                  const dal::byte_t* const* inp_pointers,
                  const data_type* inp_types,
                  const std::int64_t* inp_strides,
                  dal::byte_t* const* out_pointers,
                  const data_type* out_types,
                  const std::int64_t* out_strides,
                  const shape_t& shape);

#ifdef ONEDAL_DATA_PARALLEL

sycl::event copy_convert(const detail::data_parallel_policy& policy,
                         const dal::array<data_type>& input_types,
                         const dal::array<dal::byte_t>& input_data,
                         const shape_t& input_shape,
                         data_type output_type,
                         dal::array<dal::byte_t>& output_data,
                         const shape_t& output_strides,
                         const std::vector<sycl::event>& deps = {});

sycl::event copy_convert(const detail::data_parallel_policy& policy,
                         const dal::array<const dal::byte_t*>& inp_pointers,
                         const dal::array<data_type>& inp_types,
                         const dal::array<std::int64_t>& inp_strides,
                         const dal::array<dal::byte_t*>& out_pointers,
                         const dal::array<data_type>& out_types,
                         const dal::array<std::int64_t>& out_strides,
                         const shape_t& shape,
                         const std::vector<sycl::event>& deps = {});

sycl::event copy_convert(const detail::data_parallel_policy& policy,
                         const dal::byte_t* const* inp_pointers,
                         const data_type* inp_types,
                         const std::int64_t* inp_strides,
                         dal::byte_t* const* out_pointers,
                         const data_type* out_types,
                         const std::int64_t* out_strides,
                         const shape_t& shape,
                         const std::vector<sycl::event>& deps = {});

sycl::event copy_convert(sycl::queue& queue,
                         const dal::byte_t* const* inp_pointers,
                         const data_type* inp_types,
                         const std::int64_t* inp_strides,
                         dal::byte_t* const* out_pointers,
                         const data_type* out_types,
                         const std::int64_t* out_strides,
                         const shape_t& shape,
                         const std::vector<sycl::event>& deps = {});

sycl::event copy_convert(sycl::queue& queue,
                         const dal::byte_t* const* inp_pointers,
                         data_type inp_type,
                         const std::int64_t* inp_strides,
                         dal::byte_t* const* out_pointers,
                         data_type out_type,
                         const std::int64_t* out_strides,
                         const shape_t& shape,
                         const std::vector<sycl::event>& deps = {});

#endif

} // namespace oneapi::dal::backend
