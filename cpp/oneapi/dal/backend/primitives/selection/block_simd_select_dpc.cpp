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

#include <limits>

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/selection/block_simd_select.hpp"

namespace oneapi::dal::backend::primitives {

constexpr uint32_t preffered_wg_size = 128;

template <typename Float, uint32_t simd_width, bool selection_out, bool indices_out>
sycl::event block_simd_select(sycl::queue& queue,
                              const ndview<Float, 2>& block,
                              std::int64_t k,
                              ndview<Float, 2>& selection,
                              ndview<int, 2>& indices,
                              const event_vector& deps) {
    if constexpr (selection_out) {
        ONEDAL_ASSERT(block.get_dimension(1) == selection.get_dimension(1));
        ONEDAL_ASSERT(selection.get_dimension(0) == k);
        ONEDAL_ASSERT(selection.has_mutable_data());
    }
    if constexpr (indices_out) {
        ONEDAL_ASSERT(block.get_dimension(1) == indices.get_dimension(1));
        ONEDAL_ASSERT(indices.get_dimension(0) == k);
        ONEDAL_ASSERT(indices.has_mutable_data());
    }

    const auto sg_sizes = queue.get_device().get_info<sycl::info::device::sub_group_sizes>();
    ONEDAL_ASSERT(!sg_sizes.empty());
    auto result = std::max_element(sg_sizes.begin(), sg_sizes.end());
    ONEDAL_ASSERT(result != sg_sizes.end());
    // TODO: overflow check
    const uint32_t sg_max_size = static_cast<int>(*result);
    const uint32_t expected_sg_num = preffered_wg_size / sg_max_size;

    const std::int64_t nx = block.get_dimension(1);
    const std::int64_t ny = block.get_dimension(0);
    sycl::range<2> global(
        preffered_wg_size,
        up_multiple(ny, static_cast<std::int64_t>(expected_sg_num)) / expected_sg_num);
    sycl::range<2> local(preffered_wg_size, 1);

    sycl::nd_range<2> nd_range2d(global, local);

    const Float* block_ptr = block.get_data();
    Float* selection_ptr = selection_out ? selection.get_mutable_data() : nullptr;
    int* indices_ptr = indices_out ? indices.get_mutable_data() : nullptr;

    auto fp_max = detail::limits<Float>::max();

    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range2d, [=](sycl::nd_item<2> item) {
            auto sg = item.get_sub_group();
            const uint32_t sg_id = sg.get_group_id()[0];
            const uint32_t wg_id = item.get_global_id(1);
            const uint32_t sg_num = sg.get_group_range()[0];
            const uint32_t sg_global_id = wg_id * sg_num + sg_id;
            if (sg_global_id >= ny)
                return;
            const uint32_t in_offset = sg_global_id * nx;
            const uint32_t out_offset = sg_global_id * k;

            const uint32_t local_id = sg.get_local_id()[0];
            const uint32_t local_range = sg.get_local_range()[0];

            Float values[simd_width];
            uint32_t private_indices[simd_width];

            for (uint32_t i = 0; i < simd_width; i++) {
                values[i] = fp_max;
                private_indices[i] = -1;
            }

            for (uint32_t i = local_id; i < nx; i += local_range) {
                Float cur_val = block_ptr[in_offset + i];
                uint32_t index = i;
                int pos = -1;

                pos = values[k - 1] > cur_val ? k - 1 : pos;
                for (uint32_t j = k - 2; j >= 0; j--) {
                    bool do_shift = values[j] > cur_val;
                    pos = do_shift ? j : pos;
                    values[j + 1] = do_shift ? values[j] : values[j + 1];
                    private_indices[j + 1] = do_shift ? private_indices[j] : private_indices[j + 1];
                }

                if (pos != -1) {
                    values[pos] = cur_val;
                    private_indices[pos] = index;
                }
            }
            sg.barrier();

            int bias = 0;
            Float final_values[simd_width];
            int final_indices[simd_width];
            for (uint32_t i = 0; i < k; i++) {
                Float min_val = reduce(sg, values[bias], sycl::ONEAPI::minimum());
                bool present = (min_val == values[bias]);
                int pos = exclusive_scan(sg, present ? 1 : 0, std::plus<int>());
                bool owner = present && pos == 0;
                final_indices[i] =
                    -reduce(sg, owner ? -private_indices[bias] : 1, sycl::ONEAPI::minimum());
                final_values[i] = min_val;
                bias += owner ? 1 : 0;
            }
            if constexpr (selection_out) {
                for (uint32_t i = local_id; i < nx; i += local_range) {
                    indices_ptr[out_offset + i] = final_indices[i];
                }
            }
            if constexpr (indices_out) {
                for (uint32_t i = local_id; i < nx; i += local_range) {
                    selection_ptr[out_offset + i] = final_values[i];
                }
            }
        });
    });
    return event;
}

#define INSTANTIATE(F, simd_width, selection_out, indices_out)                              \
    template ONEDAL_EXPORT sycl::event                                                      \
    block_simd_select<F, simd_width, selection_out, indices_out>(sycl::queue & queue,       \
                                                                 const ndview<F, 2>& block, \
                                                                 std::int64_t k,            \
                                                                 ndview<F, 2>& selection,   \
                                                                 ndview<int, 2>& indices,   \
                                                                 const event_vector& deps);

#define INSTANTIATE_FLOAT(simd_width, selection_out, indices_out) \
    INSTANTIATE(float, simd_width, selection_out, indices_out)    \
    INSTANTIATE(double, simd_width, selection_out, indices_out)

#define INSTANTIATE_SIMD_WIDTH(simd_width)     \
    INSTANTIATE_FLOAT(simd_width, true, false) \
    INSTANTIATE_FLOAT(simd_width, false, true) \
    INSTANTIATE_FLOAT(simd_width, true, true)

INSTANTIATE_SIMD_WIDTH(16)
INSTANTIATE_SIMD_WIDTH(32)
INSTANTIATE_SIMD_WIDTH(64)
INSTANTIATE_SIMD_WIDTH(128)

} // namespace oneapi::dal::backend::primitives
