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

#include "oneapi/dal/array.hpp"

#include "oneapi/dal/backend/common.hpp"

#include "oneapi/dal/backend/primitives/convert/copy_convert.hpp"

namespace oneapi::dal::backend::primitives {

/*template <typename InpType, typename OutType>
auto propose_range(const sycl::queue& queue, const shape_t& shape) {
    const auto [row_count, col_count] = shape;
    const auto raw_wg = propose_wg_size(queue);
    const auto count = std::max(raw_wg, col_count);
    return std::make_pair(row_count, count);
}

template <typename InpType, typename OutType>
void copy_convert(sycl::queue& queue,
                  const InpType* const* inp_pointers,
                  const std::int64_t* inp_strides,
                  OutType* const* out_pointers,
                  const std::int64_t* out_strides,
                  const shape_t& shape,
                  const std::vector<sycl::event>& deps) {
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);

        const auto range = poropose_range<InpType, OutType>(queue, shape);
        const auto range_2d = make_range_2d(range.first, range.second);

        const std::int64_t col_count = shape.second;
        const std::int64_t wi_per_row = range.second;
        h.paralell_for(range_2d, [=](sycl::id<2> idx) -> void {
            const std::int64_t row = idx[0];
            OutType* const out_ptr = out_pointers[row];
            const InpType* const inp_ptr = inp_pointers[row];

            for (std::int64_t col = idx[1]; col < col_count; col += wi_per_row) {
                const std::int64_t out_offset = col * out_str;
                const std::int64_t inp_offset = col * inp_str;
                out_ptr[out_offset] = static_cast<OutType>(inp_ptr[inp_offset]);
            }
        });
    });
}

auto func_unique_pairs(const data_type* inp, const data_type* out, std::int64_t count) {

}*/

} // namespace oneapi::dal::backend::primitives