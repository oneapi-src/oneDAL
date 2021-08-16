/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/selection/kselect_by_rows_base.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

// Performs k-selection for k == 1
template <typename Float>
class kselect_by_rows_single_col : public kselect_by_rows_base<Float> {
public:
    kselect_by_rows_single_col() {}
    sycl::event operator()(sycl::queue& queue,
                           const ndview<Float, 2>& data,
                           std::int64_t k,
                           ndview<Float, 2>& selection,
                           ndview<std::int32_t, 2>& indices,
                           const event_vector& deps) override {
        return select<true, true>(queue, data, selection, indices, deps);
    }
    sycl::event operator()(sycl::queue& queue,
                           const ndview<Float, 2>& data,
                           std::int64_t k,
                           ndview<Float, 2>& selection,
                           const event_vector& deps) override {
        ndarray<std::int32_t, 2> dummy;
        return select<true, false>(queue, data, selection, dummy, deps);
    }
    sycl::event operator()(sycl::queue& queue,
                           const ndview<Float, 2>& data,
                           std::int64_t k,
                           ndview<std::int32_t, 2>& indices,
                           const event_vector& deps) override {
        ndarray<Float, 2> dummy;
        return select<false, true>(queue, data, dummy, indices, deps);
    }

private:
    template <bool selection_out, bool indices_out>
    sycl::event select(sycl::queue& queue,
                       const ndview<Float, 2>& data,
                       ndview<Float, 2>& selection,
                       ndview<std::int32_t, 2>& indices,
                       const event_vector& deps = {}) {
        ONEDAL_ASSERT(!indices_out || indices.get_shape()[0] == data.get_shape()[0]);
        ONEDAL_ASSERT(!indices_out || indices.get_shape()[1] == 1);
        ONEDAL_ASSERT(!selection_out || selection.get_shape()[0] == data.get_shape()[0]);
        ONEDAL_ASSERT(!selection_out || selection.get_shape()[1] == 1);

        const std::int64_t col_count = data.get_dimension(1);
        const std::int64_t row_count = data.get_dimension(0);
        const std::int64_t inp_stride = data.get_leading_stride();
        [[maybe_unused]] const std::int64_t out_ids_stride = indices.get_leading_stride();
        [[maybe_unused]] const std::int64_t out_dst_stride = selection.get_leading_stride();

        const std::int64_t wg_size =
            get_scaled_wg_size_per_row(queue, col_count, preffered_wg_size);

        const Float* data_ptr = data.get_data();
        [[maybe_unused]] Float* selection_ptr =
            selection_out ? selection.get_mutable_data() : nullptr;
        [[maybe_unused]] std::int32_t* indices_ptr =
            indices_out ? indices.get_mutable_data() : nullptr;
        const auto fp_max = detail::limits<Float>::max();

        auto event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.parallel_for(
                make_multiple_nd_range_2d({ wg_size, row_count }, { wg_size, 1 }),
                [=](sycl::nd_item<2> item) {
                    auto sg = item.get_sub_group();
                    const std::uint32_t sg_id = sg.get_group_id()[0];
                    const std::uint32_t wg_id = item.get_global_id(1);
                    const std::uint32_t sg_num = sg.get_group_range()[0];
                    const std::uint32_t sg_global_id = wg_id * sg_num + sg_id;
                    if (sg_global_id >= row_count)
                        return;
                    const std::uint32_t in_offset = sg_global_id * inp_stride;
                    [[maybe_unused]] const std::int32_t offset_ids_out = sg_global_id * out_ids_stride;
                    [[maybe_unused]] const std::int32_t offset_dst_out = sg_global_id * out_dst_stride;

                    const std::uint32_t local_id = sg.get_local_id()[0];
                    const std::uint32_t local_range = sg.get_local_range()[0];

                    std::int32_t index = -1;
                    Float value = fp_max;
                    for (std::uint32_t i = local_id; i < col_count; i += local_range) {
                        const Float cur_val = data_ptr[in_offset + i];
                        if (cur_val < value) {
                            index = i;
                            value = cur_val;
                        }
                    }

                    sg.barrier();

                    const Float final_value = reduce(sg, value, sycl::ONEAPI::minimum<Float>());
                    const bool present = (final_value == value);
                    const std::int32_t pos =
                        exclusive_scan(sg, present ? 1 : 0, sycl::ONEAPI::plus<std::int32_t>());
                    const bool owner = present && pos == 0;
                    const std::int32_t final_index =
                        -reduce(sg, owner ? -index : 1, sycl::ONEAPI::minimum<std::int32_t>());

                    if (local_id == 0) {
                        if constexpr (indices_out) {
                            indices_ptr[offset_ids_out] = final_index;
                        }
                        if constexpr (selection_out) {
                            selection_ptr[offset_dst_out] = final_value;
                        }
                    }
                });
        });
        return event;
    }
    static constexpr std::int64_t preffered_wg_size = 128;
};
#endif

} // namespace oneapi::dal::backend::primitives
