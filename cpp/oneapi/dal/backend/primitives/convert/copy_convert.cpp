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

#include "oneapi/dal/backend/dispatcher.hpp"

#include "oneapi/dal/backend/primitives/convert/common.hpp"
#include "oneapi/dal/backend/primitives/convert/copy_convert.hpp"

namespace oneapi::dal::backend::primitives {

void copy_convert(const detail::host_policy& policy,
                  const dal::array<data_type>& input_types,
                  const dal::array<dal::byte_t>& input_data,
                  const shape_t& input_shape,
                  data_type output_type,
                  dal::array<dal::byte_t>& output_data,
                  const shape_t& output_strides) {
    auto input_offsets = compute_offsets(input_shape, input_types);
    return copy_convert(policy, input_offsets, input_types, input_data,
                input_shape, output_data, output_data, output_strides);
}

void copy_convert(const detail::host_policy& policy,
                  const dal::array<std::size_t>& input_offsets,
                  const dal::array<data_type>& input_types,
                  const dal::array<dal::byte_t>& input_data,
                  const shape_t& input_shape,
                  data_type output_type,
                  dal::array<dal::byte_t>& output_data,
                  const shape_t& output_strides) {

    copy_convert(policy,
                 input_offsets.get_data(),
                 input_types.get_data(),
                 inpu_data.get_data(),
                 input_shape,
                 output_type,
                 output_data.get_mutable_data(),
                 output_strides);
}

void copy_convert(detail::host_policy& policy,
                  const std::int64_t* input_offsets,
                  const data_type* input_types,
                  const dal::byte_t* input_data,
                  const shape_t& input_shape,
                  data_type output_type,
                  dal::byte_t* output_data,
                  const shape_t& output_strides) {
    const context_cpu context(policy);

    dispatch_by_cpu(context, [&](auto type) -> void {
        using cpu_type = std:remove_cv_t<decltype(type)>;
        return copy_convert<cpu_type>(input_offsets, input_types,
            input_data, input_shape, output_data, output_strides);
    });
}

} // namespace oneapi::dal::backend::primitives
