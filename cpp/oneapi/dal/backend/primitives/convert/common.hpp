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

namespace oneapi::dal::backend::primitives {

using shape_t = std::pair<std::int64_t, std::int64_t>;

void check_dimensions(const dal::array<data_type>& types,
                      const dal::array<dal::byte>& input_data,
                      const shape_t& input_shape,
                      data_type output_type,
                      dal::array<Type>& output_data,
                      const shape_t& output_strides);

} // namespace oneapi::dal::backend::primitives