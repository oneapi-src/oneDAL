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

#include "oneapi/dal/backend/primitives/selection/block_select_single_pass.hpp"

namespace oneapi::dal::backend::primitives {


template <typename Float, bool selected_out, bool indices_out>
sycl::event block_select_single_pass(sycl::queue& queue,
                 const ndview<Float, 2>& block,
                 std::int64_t k,
                 ndview<Float, 2>& selected,
                 ndview<int, 2>& indices,
                 const event_vector& deps) {
    ONEDAL_ASSERT(block.get_dimension(1) == selected.get_dimension(1));
    ONEDAL_ASSERT(block.get_dimension(1) == indices.get_dimension(1));
    ONEDAL_ASSERT(indices.get_dimension(0) == k);
    ONEDAL_ASSERT(selected.get_dimension(0) == k);
    ONEDAL_ASSERT(indices.has_mutable_data());
    ONEDAL_ASSERT(selection.has_mutable_data());

    sycl::event::wait_and_throw(deps);

    const Float* block_ptr = block.get_data();
    Float* selected_ptr = selected_out ? selected.get_mutable_data() : nullptr;
    int* indices_ptr = indices_out ? indices.get_mutable_data() : nullptr;

    auto nx = block.get_dimension(1);
    auto ny = block.get_dimension(0);

    auto fp_max = std::numeric_limits<Float>::max();

    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for_work_group({ny}, {nx}, [=](sycl::group<1> grp) {
            const int in_offset = grp.get_id(0) * nx;
            const int out_offset = grp.get_id(0) * k;

            Float values[32];
            int private_indices[32];

            grp.parallel_for_work_item([&](sycl::h_item<2> item) {

                for(int i = 0; i < 32; i++) {
                    values[i] = fp_max;
                    private_indices[i] = -1;
                }

                for(int i = item.get_physical_local_id(0); i < nx; i += item.get_physical_local_range(0)) {
                    Float cur_val = block_ptr[in_offset + i];
                    int index             = i;
                    int pos               = -1;

                    for (int j = k - 1; j > -1; j--)
                    {
                        bool do_shift = values[j] > cur_val;
                        pos           = do_shift ? j : pos;
                        if (j < k - 1)
                        {
                            values[j + 1]  = do_shift ? values[j] : values[j + 1];
                            private_indices[j + 1] = do_shift ? private_indices[j] : private_indices[j + 1];
                        }
                    }
                    if (pos != -1)
                    {
                        values[pos]  = cur_val;
                        private_indices[pos] = index;
                    }
                }
            });
            grp.parallel_for_work_item([&](sycl::h_item<2> item) {

                int bias = 0;
                Float final_values[32];
                int final_indices[32];
                for (int i = 0; i < k; i++)
                {
                    Float min_val = reduce_over_group(grp, values[bias], std::min<Float>());
                    bool present            = (min_val == values[bias]);
                    int pos                 = exclusive_scan_over_group(grp, present ? 1 : 0, std::plus<Float>());
                    bool owner              = present && pos == 0;
                    final_indices[i]        = - reduce_over_group(grp, owner ? -private_indices[bias] : 1, std::min<Float>());
                    final_values[i]         = min_val;
                    bias += owner ? 1 : 0;
                }
                if constexpr (selected_out) {
                    for(int i = item.get_local_id(0); i < nx; i += grp.get_local_range(0))
                    {
                        indices_ptr[out_offset + i] = final_indices[i];
                    }
                }
                if constexpr (indices_out) {
                    for(int i = item.get_local_id(0); i < nx; i += grp.get_local_range(0))
                    {
                        selected_ptr[out_offset + i] = final_values[i];
                    }
                }
            });
        });
    });
    return event;
}

#define INSTANTIATE(F, selected_out, indices_out)                                                    \
    template ONEDAL_EXPORT sycl::event block_select_single_pass<F, selected_out, indices_out>(sycl::queue & queue,       \
                 const ndview<F, 2>& block, \
                 std::int64_t k, \
                 ndview<F, 2>& selected, \
                 ndview<int, 2>& indices, \
                 const event_vector& deps);

#define INSTANTIATE_FLOAT(selected_out, indices_out) \
    INSTANTIATE(float, selected_out, indices_out)    \
    INSTANTIATE(double, selected_out, indices_out)

INSTANTIATE_FLOAT(true, false)
INSTANTIATE_FLOAT(false, true)
INSTANTIATE_FLOAT(true, true)


} // namespace oneapi::dal::backend::primitives
