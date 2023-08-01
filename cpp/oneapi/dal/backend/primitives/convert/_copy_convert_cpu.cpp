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

template <typename CpuType, typename OutputType, typename InputType>
struct copy_converter_impl {
    using out_t = OutputType;
    using inp_t = InputType;

    static void run(out_t* out, std::int64_t out_stride,
                    const inp_t* inp, std::int64_t inp_stride,
                    std::int64_t count) {
        ONEDAL_ASSERT(0l < out_stride);
        ONEDAL_ASSERT(0l < out_stride);
        ONEDAL_ASSERT(out != nullptr);
        ONEDAL_ASSERT(inp != nullptr);
        ONEDAL_ASSERT(0l <= count);

        if (out_stride == 1l && inp_stride == 1l) {
             return contiguous(out, inp, count);
        }

        if (out_stride == 1l && inp_stride != 1l) {
             return semi_strided(out, out_stride, inp, count);
        }

        if (out_stride != 1l && inp_stride == 1l) {
             return semi_strided(out, inp, inp_stride, count);
        }

        return strided(out, out_stride, inp, inp_stride, count);
    }

    // TODO: Optimize with memcpy
    static void contiguous(out_t* out, const inp_t* inp, std::int64_t count) {

        for (std::int64_t i = 0l; i < count; ++i) {
            out[i] = static_cast<out_t>(inp[i]);
        }
    }

    static void semi_strided(out_t* out, std::int64_t out_stride,
                            const inp_t* inp, std::int64_t count) {

        for (std::int64_t i = 0l; i < count; ++i) {
            const std::int64_t out_offset = i * out_stride;
            out[out_offset] = static_cast<out_t>(inp[i]);
        }
    }

    // TODO: Optimize with gather instructions
    static void semi_strided(out_t* out, const inp_t* inp,
                    std::int64_t inp_stride, std::int64_t count) {

        for (std::int64_t i = 0l; i < count; ++i) {
            const std::int64_t inp_offset = i * inp_stride;
            out[i] = static_cast<out_t>(inp[inp_offset]);
        }
    }

    static void strided(out_t* out, std::int64_t out_stride,
            const inp_t* inp, std::int64_t inp_stride, std::int64_t count) {

        for (std::int64_t i = 0l; i < count; ++i) {
            const std::int64_t out_offset = i * out_stride;
            const std::int64_t inp_offset = i * inp_stride;
            out[out_offset] = static_cast<out_t>(inp[inp_offset]);
        }
    }
};

template <typename CpuType, typename InpType, typename OutType>
inline auto propose_block_size(const detail::host_policy& policy) {
    constexpr std::int64_t out_size = sizeof(OutType);
    constexpr std::int64_t l1_estimation = 16'384l;
    return std::max(128l, l1_estimation / out_size);
}

template <typename CpuType, typename InpType, typename OutType>
void copy_convert(const detail::host_policy& policy,
                  const InpType* inp_ptr,
                  std::int64_t inp_str,
                  OutType* out_ptr,
                  std::int64_t out_str,
                  std::int64_t count) {
    const auto count_s = detail::integral_cast<std::size_t>(count);
    const auto block_size = propose_block_size<CpuType, InpType, OutType>(policy);
    const auto block_size_s = detail::integral_cast<std::size_t>(block_size);

    detail::threader_for_blocked_size(count_s, block_size_s,
    [=](std::size_t f, std::size_t l) -> void {
        const auto first = detail::integral_cast<std::int64_t>(f);
        const auto last = detail::integral_cast<std::int64_t>(l);
        const std::int64_t count = last - first;

        copy_converter_impl<CpuType, OutType, InpType>::run(
            out_ptr, out_str, inp_ptr, inp_str, count);
    });
}

template <typename CpuType>
void copy_convert(const detail::host_policy& policy,
                  const dal::byte_t** inp_ptrs,
                  const data_type* inp_types,
                  const std::int64_t* inp_strs,
                  dal::byte_t** out_ptrs,
                  const data_type* out_types,
                  const std::int64_t* out_strs,
                  const shape_t& shape) {

    const std::int64_t row_count = shape.first;
    const std::int64_t col_count = shape.second;

    detail::threader_for_int64(row_count,
    [&](std::int64_t i) -> void {
        auto* out_raw_ptr = out_ptrs[i];
        const auto* inp_raw_ptr = inp_ptrs[i];

        const auto out_str = out_strs[i];
        const auto inp_str = inp_strs[i];

        const auto out_type = out_types[i];
        const auto inp_type = inp_types[i];

        backend::multi_dispatch_by_data_type(
        [&](auto out, auto inp) -> void {
            using input_t = std::decay_t<decltype(inp_type)>;
            using output_t = std::decay_t<decltype(out_type)>;

            auto* out_ptr = reinterpret_cast<output_t*>(out_raw_ptr);
            const auto* inp_ptr = reinterpret_cast<const input_t*>(inp_raw_ptr);

            copy_convert<CpuType>(policy, inp_ptr, inp_str,
                                  out_ptr, out_str, col_count);
        }, out_type, inp_type);
    });
}


template void copy_convert<__CPU_TAG__>(
                  const detail::host_policy& policy,
                  const dal::byte_t** inp_ptrs,
                  const data_type* inp_types,
                  const std::int64_t* inp_strs,
                  dal::byte_t** out_ptrs,
                  const data_type* out_types,
                  const std::int64_t* out_strs,
                  const shape_t& shape);

} // namespace oneapi::dal::backend::primitives