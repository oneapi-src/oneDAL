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
#include "oneapi/dal/common.hpp"

#include "oneapi/dal/backend/common.hpp"

namespace oneapi::dal::backend::primitives {

using shape_t = std::pair<std::int64_t, std::int64_t>;

bool is_known_data_type(data_type dtype) noexcept;

template <typename Index, typename Type>
void extract_by_indices_impl(const Index* indices, const Type* values,
                                    Type* output, std::int64_t count);

dal::array<dal::byte_t> extract_by_indices(const dal::byte_t* indices, data_type indices_type,
                            const std::byte* values, data_type values_type, std::int64_t count);

template <typename Index, typename Type>
inline dal::array<Type> extract_by_indices(const Index* indices,
                        const Type* values, std::int64_t count) {
    constexpr auto val_type = detail::make_data_type<Type>();
    constexpr auto idx_type = detail::make_data_type<Index>();

    const auto* val_ptr = reinterpret_cast<const dal::byte_t*>(values);
    const auto* idx_ptr = reinterpret_cast<const dal::byte_t*>(indices);

    auto raw_res = extract_by_indices(idx_ptr, idx_type, val_ptr, val_type, count);

    dal::byte_t* const raw_ptr = raw_res.get_mutable_data();
    auto* const res_ptr = reinterpret_cast<Type* const>(raw_ptr);

    return dal::array<Type>(raw_res, res_ptr, count);
}

template <typename Index, typename Type>
inline dal::array<Type> extract_by_indices(const dal::array<Index>& indices,
                                           const dal::array<Type>& values) {
    const std::int64_t count = indices.get_count();
    return extract_by_indices(indices.get_data(), values.get_data(), count);
}

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