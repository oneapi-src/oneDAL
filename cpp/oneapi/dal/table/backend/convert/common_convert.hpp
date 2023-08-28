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

#include <type_traits>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/common.hpp"

#include "oneapi/dal/backend/common.hpp"

#include "oneapi/dal/table/backend/convert/common.hpp"

namespace oneapi::dal::backend {

template <bool mut, typename Pointer = std::conditional_t<mut, dal::byte_t*, const dal::byte_t*>>
dal::array<Pointer> compute_pointers(const dal::array<dal::byte_t>& data,
                                     const dal::array<std::int64_t>& offsets);

dal::array<std::int64_t> compute_output_offsets(data_type output_type,
                                                const shape_t& input_shape,
                                                const shape_t& output_strides);

dal::array<std::int64_t> compute_input_offsets(const shape_t& input_shape,
                                               const data_type* input_types);

dal::array<std::int64_t> compute_input_offsets(const shape_t& input_shape,
                                               const dal::array<data_type>& input_types);

} // namespace oneapi::dal::backend
