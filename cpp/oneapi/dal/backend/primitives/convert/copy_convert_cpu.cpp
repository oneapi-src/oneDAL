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

#include "oneapi/dal/detail/threading.hpp"

#include "oneapi/dal/backend/dispatcher.hpp"

#include "oneapi/dal/backend/primitives/convert/common.hpp"
#include "oneapi/dal/backend/primitives/convert/copy_convert.hpp"

namespace oneapi::dal::backend::primitives {

template <typename CpuType, typename OutputType, typename InputType>
struct copy_converter_impl {
    using out_t = OutputType;
    using inp_t = InputType;

    static void run(out_t* out, const inp_t* inp,
                    std::int64_t count, std::int64_t stride) {
        ONEDAL_ASSERT(out != nullptr);
        ONEDAL_ASSERT(inp != nullptr);
        ONEDAL_ASSERT(0l <= count);
        ONEDAL_ASSERT(0l < stride);

        if (stride == 1l) {
            contiguous(out, inp, count);
        }
        else {
            strided(out, inp, count, stride);
        }
    }

    // TODO: Use template specialization
    // in case of the same type
    static void contiguous(out_t* const out,
            const inp_t* const inp, std::int64_t count) noexcept {

        for (std::int64_t i = 0l; i < count; ++i) {
            out[i] = static_cast<out_t>(inp[i]);
        }
    }

    static void strided(out_t* const out,
            const inp_t* const inp, std::int64_t count,
            std::int64_t stride) noexcept {

        for (std::int64_t i = 0l; i < count; ++i) {
            out[i * stride] = static_cast<out_t>(inp[i]);
        }
    }
};

/// @brief Computes offsets
inline auto decompose_index(std::int64_t idx, std::int64_t block) {
    ONEDAL_ASSERT(0l < block);

    const std::int64_t row = i / block;
    const std::int64_t col = i - row * block

    ONEDAL_ASSERT(0l <= row);
    ONEDAL_ASSERT(0l <= col);
    ONEDAL_ASSERT(col < block);
    return std::make_pair(row, col);
}

/// @brief Converts index of thread to the index
///        of the first element in data block
template <typename Index>
inline auto get_fisrt_index(const Index& idx, std::int64_t block,
                            std::int64_t col_count) {
    auto [row, col] = idx;
    ONEDAL_ASSERT(0l <= row);
    ONEDAL_ASSERT(0l <= col);
    ONEDAL_ASSERT(col < block);

    auto col_block = col_count / block;
    ONEDAL_ASSERT(0l < col_block);

    auto first_idx = col * col_block;
    ONEDAL_ASSERT(first_idx < col_count);

    return std::make_pair(row, first_idx);
}

/// @brief Converts index of thread to the
///        sentinel in data block
template <typename Index>
inline auto get_last_index(const Index& idx, std::int64_t block,
                           std::int64_t col_count) {
    auto [row, col] = idx;
    ONEDAL_ASSERT(0l <= row);
    ONEDAL_ASSERT(0l <= col);
    ONEDAL_ASSERT(col < block);

    auto col_block = col_count / block;
    ONEDAL_ASSERT(0l < col_block);

    const auto wannabe_idx = (col + 1l) * col_block;
    auto last_idx = std::min(wannabe_idx, col_count);
    ONEDAL_ASSERT(last_idx <= col_count);

    return std::make_pair(row, last_idx);
}

template <typename Index>
inline auto get_input_offset(const Index& first_idx, data_type type,
                             const std::size_t* const offsets) {
    auto [row, first_col] = first_idx;
    const auto data_size = detail::get_data_type_size(type);
    const auto row_offset = (row == 0l) ? 0l : offsets[row - 1l];

    return row_offset + data_size * first_col;
}

template <typename Index>
inline auto get_input_offset(const Index& first_idx,
                             const data_type* types,
                             const std::size_t* offsets) {
    auto [row, _first_col] = first_idx;
    return get_input_offset(first_idx, types[row], offsets);
}

template <typename Index, typename Strides>
inline auto get_input_offset(const Index& first_idx,
                data_type type, const Strides& strides) {
    auto [row, col] = first_idx;
    auto [row_str, col_str] = strides;
    auto size = detail::get_data_type_size(type);
    return (row * row_str + col * col_str) * size;
}

template <typename CpuType, typename Shape>
inline auto get_threading_policy(const Shape& workload_shape,
                                 std::int64_t native_thread_count) {
    // TODO: dispatching by CpuType
    constexpr std::int64_t max_vec = 16l;

    auto [row_count, col_count] = workload_shape;
    auto lcm = std::lcm(row_count, native_thread_count);
    const std::int64_t lcm_per_row = lcm / row_count;

    // Should almost always be true for
    if (lcm_per_row * max_vec <= col_count) {
        return std::make_pair(row_count, lcm_per_row);
    }
    else {
        const auto thr_per_row = (native_thread_count / row_count) //
                            + bool(native_thread_count % row_count);
        return std::make_pair(row_count, thr_per_row);
    }
}

template <typename CpuType>
void copy_convert(const std::int64_t* input_offsets,
                  const data_types* input_types,
                  const dal::byte* input_data,
                  const shape_t& input_shape,
                  data_type output_type,
                  dal::byte* output_data,
                  const shape_t& output_strides) {

    const auto [row_count, col_count] = input_shape;
    const auto native_thread_count = detail::threader_get_max_threads();
    const auto threading = get_threading_policy<CpuType>(input_shape, native_thread_count);

    const auto [row_threads, col_threads] = threading;
    const auto thread_count = detail::check_mul_overflow(row_threads, col_threads);

    detail::threader_for_int64(thread_count, [&](std::int64_t i) -> void {
        const auto idx = decompose_index(i, col_threads);
        const auto [row_idx, col_idx] = idx;
        ONEDAL_ASSERT(row_idx < row_threads);

        const auto first = get_fisrt_index(idx, col_threads, col_count);
        const auto last = get_last_index(idx, col_threads, col_count);

        [[maybe_unused]] auto [f_row, f_col] = first, [l_row, l_col] = last;
        ONEDAL_ASSERT((0l <= f_col) && (f_col <= l_col) && (l_col <= col_count));
        ONEDAL_ASSERT((row_idx == f_row) && (row_idx == l_row));

        const auto inp_offset = get_input_offset(first, input_types, input_offsets);
        const auto out_offset = get_output_offset(first, output_type, output_strides);

        // Preparation for an actual kernel
        const auto* const raw_inp = input_data + inp_offset;
        auto* const raw_out = output_data + out_offset;

        const std::int64_t stride = output_strides.second;
        const std::int64_t count = l_col - f_col;

        // Dispatch parameters
        const data_type input_type = input_types[row_idx];
        const std::array<data_type, 2ul> dtypes{ output_type, input_type };
        dispatch_by_data_types<2ul>(dtypes.data(), [=](auto output, auto input) {
            using out_t = std::remove_cv_t<decltype(output)>;
            using inp_t = std::remove_cv_t<decltype(input)>;

            const auto* const inp_ptr = reinterpret_cast<const inp_t*>(raw_inp);
            auto* const out_ptr = reinterpret_cast<out_t*>(raw_out);

            copy_converter_impl<CpuType, out_t, inp_t>::run(inp_ptr, out_ptr, count, stride);
        });
    });
}

} // namespace oneapi::dal::backend::primitives
