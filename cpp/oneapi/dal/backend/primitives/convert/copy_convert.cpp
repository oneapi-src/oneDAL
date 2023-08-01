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

#include <utility>
#include <numeric>
#include <algorithm>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/primitives/convert/common.hpp"
#include "oneapi/dal/backend/primitives/convert/copy_convert.hpp"

namespace oneapi::dal::backend::primitives {

template <bool mut = true>
auto compute_pointers(const dal::array<dal::byte_t>& data,
                      const dal::array<std::int64_t>& offsets) {
    const std::int64_t count = offsets.get_count();
    const std::int64_t* const raw_offsets = offsets.get_data();
    using ptr_t = std::conditional_t<mut, dal::byte_t*, const dal::byte_t*>;

    ptr_t source = nullptr;
    if constexpr (mut)
        source = data.get_mutable_data();
    else
        source = data.get_data();

    auto pointers = dal::array<ptr_t>::empty(count);
    ptr_t* raw_pointers = pointers.get_mutable_data();

    PRAGMA_IVDEP
    for (std::int64_t row = 0l; row < count; ++row) {
        raw_pointers[row] = source + raw_offsets[row];
    }

    return pointers;
}

auto compute_output_offsets(data_type output_type,
                            const shape_t& input_shape,
                            const shape_t& output_strides) {
    const auto [row_count, col_count] = input_shape;
    const auto [row_stride, col_stride] = output_strides;
    const auto type_size = detail::get_data_type_size(output_type);
    const std::int64_t row_stride_in_bytes = type_size * row_stride;

    auto offsets = dal::array<std::int64_t>::empty(row_count);
    detail::check_mul_overflow(row_count, row_stride_in_bytes);
    std::int64_t* const raw_offsets = offsets.get_mutable_data();

    PRAGMA_IVDEP
    for (std::int64_t row = 0l; row < row_count; ++row) {
        raw_offsets[row] = row * row_stride_in_bytes;
    }

    return offsets;
}

auto compute_input_offsets(const shape_t& input_shape,
                           const data_type* input_types) {
    ONEDAL_ASSERT(input_types != nullptr);
    const auto [row_count, col_count] = input_shape;
    ONEDAL_ASSERT((0l < row_count) && (0l < col_count));
    auto offsets = dal::array<std::int64_t>::empty(row_count);
    std::int64_t* const offsets_ptr = offsets.get_mutable_data();

    std::int64_t offset = 0l;
    for (std::int64_t row = 0l; row < row_count; ++row) {
        offsets_ptr[row] = offset;

        const data_type dtype = input_types[row];
        ONEDAL_ASSERT(is_known_data_type(dtype));

        auto raw_size = detail::get_data_type_size(dtype);
        auto row_size = detail::check_mul_overflow(raw_size, col_count);
        offset = detail::check_sum_overflow(offset, row_size);
    }

    return offsets;
}

auto compute_input_offsets(const shape_t& input_shape,
                           const dal::array<data_type>& input_types) {
    return compute_input_offsets(input_shape, input_types.get_data());
}

void copy_convert(const detail::host_policy& policy,
                  const dal::array<data_type>& input_types,
                  const dal::array<dal::byte_t>& input_data,
                  const shape_t& input_shape,
                  data_type output_type,
                  dal::array<dal::byte_t>& output_data,
                  const shape_t& output_strides) {
    const auto [row_count, col_count] = input_shape;
    const auto [row_stride, col_stride] = input_shape;

    auto input_offsets = compute_input_offsets(input_shape, input_types);
    auto output_offsets = compute_output_offsets(output_type, input_shape, output_strides);

    auto input_pointers = compute_pointers<false>(input_data, input_offsets);
    auto output_pointers = compute_pointers<true>(output_data, output_offsets);

    auto output_types = dal::array<data_type>::full(row_count, output_type);
    auto output_strides_arr = dal::array<std::int64_t>::full(row_count, col_stride);
    auto input_strides = dal::array<std::int64_t>::full(row_count, std::int64_t{ 1l });

    return copy_convert(policy, input_pointers, input_types, input_strides,
            output_pointers, output_types, output_strides_arr, input_shape);
}

void copy_convert(const detail::host_policy& policy,
                  const dal::array<const dal::byte_t*>& inp_pointers,
                  const dal::array<data_type>& inp_types,
                  const dal::array<std::int64_t>& inp_strides,
                  const dal::array<dal::byte_t*>& out_pointers,
                  const dal::array<data_type>& out_types,
                  const dal::array<std::int64_t>& out_strides,
                  const shape_t& shape) {

    return copy_convert(policy,
                        inp_pointers.get_data(),
                        inp_types.get_data(),
                        inp_strides.get_data(),
                        out_pointers.get_data(),
                        out_types.get_data(),
                        out_strides.get_data(),
                        shape);
}

void copy_convert(const detail::host_policy& policy,
                  const dal::byte_t* const* inp_pointers,
                  const data_type* inp_types,
                  const std::int64_t* inp_strides,
                  dal::byte_t* const* out_pointers,
                  const data_type* out_types,
                  const std::int64_t* out_strides,
                  const shape_t& shape) {
    const context_cpu context(policy);

    return dispatch_by_cpu(context, [&](auto type) -> void {
        using cpu_type = std::remove_cv_t<decltype(type)>;
        return copy_convert<cpu_type>(policy, inp_pointers, inp_types,
            inp_strides, out_pointers, out_types, out_strides, shape);
    });
}

} // namespace oneapi::dal::backend::primitives
