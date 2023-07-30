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

#include "oneapi/dal/backend/dispatcher.hpp"

#include "oneapi/dal/backend/primitives/convert/common.hpp"

namespace oneapi::dal::backend::primitives {

/*template <typename DtypeProvider>
void check_array_against_type(const shape_t& array_shape,
                              std::int64_t size_in_bytes,
                              DtypeProvider&& dtype_provider) {
#ifdef ONEDAL_ENABLE_ASSERT
    std::int64_t offset = 0l;
    auto [row_count, col_count] = array_shape;
    for (std::int64_t r = 0l; r < row_count; ++r) {
        const data_type dtype = dtype_provider(r);
        const auto tsize = detail::get_data_type_size(dtype);
        const auto row_size = detail::check_mul_overflow(tsize, col_count);

        offset = detail::check_sum_overflow(offset, row_size);
        ONEDAL_OFFSET(offset <= size_in_bytes);
    }
#endif // ONEDAL_ENABLE_ASSERT
}

void check_dimensions(const dal::array<data_type>& input_types,
                      const dal::array<dal::byte>& input_data,
                      const shape_t& input_shape,
                      data_type output_type,
                      dal::array<dal::byte>& output_data,
                      const shape_t& output_strides) {
    /* Checking input array*/ {
        auto dtype_provider = [&](std::int64_t r) { return input_types[r]; };
        const std::int64_t input_size = input_data.get_size_in_bytes();
        check_array_against_type(input_shape, input_size, dtype_provider);
    }

    /* Checking output array*/ {
        auto dtype_provider = [](std::int64_t) { return output_type; };
        const std::int64_t output_size = output_data.get_size_in_bytes();
        check_array_against_type(input_shape, output_size, dtype_provider);
    }
}*/

bool is_known_data_type(data_type dtype) noexcept {
    const auto op = [](auto type) { return true; };
    const auto unknown = [](data_type dt) { return false; };
    return detail::dispatch_by_data_type(dtype, op, unknown);
}

dal::array<std::int64_t> compute_offsets(const shape_t& input_shape,
                            const dal::array<data_type>& input_types) {
    return compute_offsets(input_shape, input_types.get_data());
}

dal::array<std::int64_t> compute_offsets(const shape_t& input_shape,
                                         const data_type* input_types) {
    ONEDAL_ASSERT(input_types != nullptr);
    const auto [row_count, col_count] = input_shape;
    ONEDAL_ASSERT((0l < row_count) && (0l < col_count));
    auto result = dal::array<std::int64_t>::empty(row_count);
    std::int64_t* const result_ptr = result.get_mutable_data();

    std::int64_t offset = 0l;
    for (std::int64_t row = 0l; row < row_count; ++row) {
        data_type dtype = input_types[row_signed];
        ONEDAL_ASSERT(is_known_data_type(dtype));

        auto raw_size = detail::get_size_by_data_type(dtype);
        auto row_size = detail::check_mul_overflow(raw_size, col_count);
        offset = detail::check_sum_overflow(offset, row_size);
        result_ptr[row] = offset;
    }

    return result;
}

} // namespace oneapi::dal::backend::primitives