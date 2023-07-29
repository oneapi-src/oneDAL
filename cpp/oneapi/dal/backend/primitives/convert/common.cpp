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

#include "oneapi/dal/backend/primitives/convert/common.hpp"

namespace oneapi::dal::backend::primitives {

template <typename DtypeProvider>
void check_array_against_type(const shape_t& array_shape,
                              std::int64_t size_in_bytes,
                              DtypeProvider&& dtype_provider) {
    const auto row_count = array_shape.first;
    const auto col_count = array_shape.second;

    [[maybe_unused]] std::int64_t offset = 0l;
    for (std::int64_t r = 0l; r < row_count; ++r) {
        const data_type dtype = dtype_provider(r);
        const auto tsize = detail::get_data_type_size(dtype);
        const auto row_size = detail::check_mul_overflow(tsize, col_count);

        offset = detail::check_sum_overflow(offset, row_size);
        ONEDAL_OFFSET(offset <= size_in_bytes);
    }
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
}

} // namespace oneapi::dal::backend::primitives