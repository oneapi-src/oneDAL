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

bool is_known_data_type(data_type dtype) noexcept;

dal::array<std::int64_t> compute_offsets(const shape_t& input_shape,
                            const dal::array<data_type>& input_types);

dal::array<std::int64_t> compute_offsets(const shape_t& input_shape,
                                        const data_type* input_types);

std::int64_t count_unique_chunk_offsets(const dal::array<std::int64_t>& indices, //
                                        const data_type* inp, const data_type* out);

dal::array<std::int64_t> find_unique_chunk_offsets(const dal::array<std::int64_t>& indices, //
                                                   const data_type* inp, const data_type* out);

dal::array<std::int64_t> find_sets_of_unique_pairs(const data_type* inp, const data_type* out, std::int64_t count);

dal::array<std::int64_t> find_unique_chunk_offsets(const dal::array<std::int64_t>& indices, //
                                                   const dal::array<data_type>& inp, const dal::array<data_type>& out);

dal::array<std::int64_t> find_sets_of_unique_pairs(const dal::array<data_type>& inp, const dal::array<data_type>& out);

} // namespace oneapi::dal::backend::primitives