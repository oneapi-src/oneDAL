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

#include "oneapi/dal/detail/array_utils.hpp"

#include "oneapi/dal/backend/common.hpp"

#include "oneapi/dal/backend/primitives/convert/common.hpp"
#include "oneapi/dal/backend/primitives/convert/copy_convert.hpp"

namespace oneapi::dal::backend::primitives {

sycl::event copy_convert(sycl::queue& queue,
                         const dal::byte_t* const* inp_pointers,
                         const data_type* inp_types,
                         const std::int64_t* inp_strides,
                         dal::byte_t* const* out_pointers,
                         const data_type* out_types,
                         const std::int64_t* out_strides,
                         const shape_t& shape,
                         const std::vector<sycl::event>& deps) {
    sycl::event::wait_and_throw(deps);
    const detail::data_parallel_policy policy{ queue };

    const auto [row_count, col_count] = shape;

    const dal::array<std::int64_t> unique_indices = //
        find_sets_of_unique_pairs(inp_types, out_types, row_count);
    const std::int64_t* const unique_indices_ptr = unique_indices.get_data();

    const dal::array<std::int64_t> chunk_offsets = //
        find_unique_chunk_offsets(unique_indices, inp_types, out_types);

    const dal::array<std::int64_t> inp_strides_host = //
        extract_by_indices(unique_indices_ptr, inp_strides, row_count);
    auto inp_strides_device = detail::copy(policy, inp_strides_host);

    const dal::array<std::int64_t> out_strides_host = //
        extract_by_indices(unique_indices_ptr, out_strides, row_count);
    auto out_strides_device = detail::copy(policy, out_strides_host);

    const dal::array<const dal::byte_t*> inp_pointers_host = //
        extract_by_indices(unique_indices_ptr, inp_pointers, row_count);
    auto inp_pointers_device = detail::copy(policy, inp_pointers_host);

    const dal::array<dal::byte_t*> out_pointers_host = //
        extract_by_indices(unique_indices_ptr, out_pointers, row_count);
    auto out_pointers_device = detail::copy(policy, out_pointers_host);

    std::int64_t first = 0l;
    sycl::event last_event{};

    const std::int64_t chunk_count = chunk_offsets.get_count();
    for (std::int64_t chunk = 0l; chunk < chunk_count; ++chunk) {
        const auto last = chunk_offsets[chunk];
        const auto chunk_len = last - first;
        ONEDAL_ASSERT(0l < chunk_len);

        const auto first_idx = unique_indices_ptr[first];
        const data_type inp_type = inp_types[first_idx];
        const data_type out_type = out_types[first_idx];

        auto inp_ptrs_slice = inp_pointers_device.get_slice(first, last);
        auto out_ptrs_slice = out_pointers_device.get_slice(first, last);
        auto inp_strs_slice = inp_strides_device.get_slice(first, last);
        auto out_strs_slice = out_strides_device.get_slice(first, last);

        const auto* const inp_ptrs = inp_ptrs_slice.get_data();
        const auto* const out_ptrs = out_ptrs_slice.get_data();
        const auto* const inp_strs = inp_strs_slice.get_data();
        const auto* const out_strs = out_strs_slice.get_data();

        const auto chunk_shape = std::make_pair(chunk_len, col_count);

        last_event = copy_convert(queue, inp_ptrs, inp_type, inp_strs,
            out_ptrs, out_type, out_strs, chunk_shape, { last_event });


        first = last;
    }

    return last_event;

}

} // namespace oneapi::dal::backend::primitives