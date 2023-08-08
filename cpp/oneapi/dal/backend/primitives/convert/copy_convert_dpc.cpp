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

#include "oneapi/dal/backend/common.hpp"

#include "oneapi/dal/backend/primitives/convert/common.hpp"
#include "oneapi/dal/backend/primitives/convert/copy_convert.hpp"

namespace oneapi::dal::backend::primitives {

/*sycl::event copy_convert(sycl::queue& queue,
                         const dal::byte_t* const* inp_pointers,
                         const data_type* inp_types,
                         const std::int64_t* inp_strides,
                         dal::byte_t* const* out_pointers,
                         const data_type* out_types,
                         const std::int64_t* out_strides,
                         const shape_t& shape,
                         const std::vector<sycl::event>& deps) {
    const auto [row_count, col_count] = shape;

    const dal::array<std::int64_t> unique_indices = //
        find_sets_of_unique_pairs(inp_types, out_types, row_count);
    const std::int64_t* const unique_indices_ptr = unique_indices.get_data();

    const dal::array<std::int64_t> chunk_offsets = //
        find_unique_chunk_offsets(unique_indices, inp_types, out_types);
    const std::int64_t* const chunk_offsets_ptr = chunk_offsets.get_data();

    const dal::array<std::int64_t> inp_strides_host = //
        extract_by_indices(unique_indices_ptr, inp_strides, row_count);

    const dal::array<std::int64_t> out_strides_host = //
        extract_by_indices(unique_indices_ptr, out_strides, row_count);

    const dal::array<const dal::byte_t*> inp_pointers_host = //
        extract_by_indices(unique_indices_ptr, inp_pointers, row_count);

    const dal::array<dal::byte_t*> out_pointers_host = //
        extract_by_indices(unique_indices_ptr, out_pointers, row_count);

    const std::int64_t chunk_count = chunk_offsets.get_count();
    for (std::int64_t chunk = 0l; chunk < chunk_count; ++chunk) {
        const data_type inp_type =
        const data_type out_type =
        sycl::event copy_event = copy_convert(queue, )
    }

}*/

} // namespace oneapi::dal::backend::primitives