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

#include "oneapi/dal/backend/primitives/blas/single_pass_select.hpp"

namespace oneapi::dal::backend::primitives {


template <typename Float, bool selected_out, bool indices_out>
sycl::event block_select(sycl::queue& queue,
                 ndview<Float, 2>& block,
                 std::int64_t k,
                 std::int64_t register_width,
                 ndview<Float, 2>& selected,
                 ndview<int, 2>& indices,
                 const event_vector& deps) {
    ONEDAL_ASSERT(block.get_dimension(1) == selected.get_dimension(1));
    ONEDAL_ASSERT(block.get_dimension(1) == indices.get_dimension(1));
    ONEDAL_ASSERT(indices.get_dimension(0) == k);
    ONEDAL_ASSERT(selected.get_dimension(0) == k);
    ONEDAL_ASSERT(indices.has_mutable_data());
    ONEDAL_ASSERT(selection.has_mutable_data());
    ONEDAL_ASSERT(k <= register_width);

    sycl::event::wait_and_throw(deps);

    const Float* block_ptr = block.get_data();
    Float* selected_ptr = selected_out ? selected.get_mutable_data() : nullptr;
    int* indices_ptr = indices_out ? indices.get_mutable_data() : nullptr;

    auto nx = block.get_dimension(1);
    auto ny = block.get_dimension(0);

    auto fp_max = std::numeric_limits<Float>::max();

    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for_workgroup({n}, [=](group<1> grp) {
            const int in_offset = grp.get_global_id(0) * nx;
            const int out_offset = grp.get_global_id(0) * k;

            Float values[register_width];
            int private_indices[register_width];

            grp.parallel_for_work_item(range<2>(16,1), [&](h_item<2> item) {
                sub_group sg = item.get_sub_group();
                if(sg.get_group_id() > 0) return;

                for(int i = local_id; i < register_width; i += local_size) {
                    values[i] = fp_max;
                    private_indices[i] = -1;
                }

                for(int i = sg.get_local_id(); i < nx; i += sg.get_local_range()) {
                    Float cur_val = block_ptr[offset + i];
                    int index             = i;
                    int pos               = -1;

                    for (int j = k - 1; j > -1; j--)
                    {
                        bool do_shift = values[j] > value;
                        pos           = do_shift ? j : pos;
                        if (j < array_size - 1)
                        {
                            values[j + 1]  = do_shift ? values[j] : values[j + 1];
                            private_indices[j + 1] = do_shift ? private_indices[j] : private_indices[j + 1];
                        }
                    }
                    if (pos != -1)
                    {
                        values[pos]  = value;
                        private_indices[pos] = index;
                    }
                }
            }
            grp.parallel_for_work_item(range<2>(16,1), [&](h_item<2> item) {
                sub_group sg = item.get_sub_group();
                if(sg.get_group_id() > 0) return;

                int bias = 0;
                Float final_values[register_width];
                int final_indices[register_width];
                for (int i = 0; i < k; i++)
                {
                    algorithmFPType min_val = sg.reduce(values[bias], minimum);
                    bool present            = (min_val == values[bias]);
                    int pos                 = sg.exclusive_scan(present ? 1 : 0, plus);
                    bool owner              = present && pos == 0;
                    final_indices[i]        = -sg.reduce(owner ? -private_indices[bias] : 1, minimum);
                    final_values[i]         = min_val;
                    bias += owner ? 1 : 0;
                }
                if constexpr (selected_out) {
                    for(int i = sg.get_local_id(); i < nx; i += sg.get_local_range())
                    {
                        indices_ptr[out_offset + i] = final_indices[i];
                    }
                }
                if constexpr (indices_out) {
                    for(int i = sg.get_local_id(); i < nx; i += sg.get_local_range())
                    {
                        selected_ptr[out_offset + i] = final_values[i];
                    }
                }
            }
        });
    });
    return event;
}

} // namespace oneapi::dal::backend::primitives
