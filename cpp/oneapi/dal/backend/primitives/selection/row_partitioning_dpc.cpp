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

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/selection/row_partitioning.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

int constexpr preffered_sg_size = 16;

sycl::nd_range<2> get_row_partitioning_range(std::int64_t row_count, std::int64_t col_count)
{
    sycl::range<2> global(preffered_sg_size, row_count);
    sycl::range<2> local(preffered_sg_size, 1);
    sycl::nd_range<2> nd_range2d(global, local);
    return nd_range2d;
}

template<typename Float>
int kernel_row_partitioning(sycl::nd_item<2> item,
                                           Float* values,
                                           int* indices,
                                           int partition_start,
                                           int partition_end,
                                           Float pivot) {
    auto sg = item.get_sub_group();
    const int local_id = sg.get_local_id()[0];
    const int local_size = sg.get_local_range()[0];

    const int seq_size = partition_end - partition_start;
    const int last_group_size = seq_size % local_size;
    int full_group_size = (seq_size / local_size) * local_size;
    full_group_size += last_group_size > 0 ? local_size : 0; 
    int split_index = 0;


    for (int i = partition_start + local_id; i < partition_start + full_group_size; i += local_size) {
        sg.barrier();
        bool inside = i < partition_end;
        Float cur_value = inside ? values[i] : 0.0;
        int cur_index = i < partition_end ? indices[i] : -1;
        int is_small = cur_value < pivot ? 1 : 0;
        const int num_of_small = reduce(sg, is_small && inside ? 1 : 0, sycl::ONEAPI::plus<int>());
        int min_ind = reduce(sg, i, sycl::ONEAPI::minimum<int>());
        if (num_of_small > 0) {
            const int pos_in_group_small = exclusive_scan(sg, is_small && inside ? 1 : 0, std::plus<int>());
            const int pos_in_group_great = num_of_small + exclusive_scan(sg, is_small || !inside ? 0 : 1, std::plus<int>());
            const int pos_small = partition_start + split_index + pos_in_group_small;
            const int pos_great = min_ind + pos_in_group_great;
            const int pos = is_small ? pos_small : pos_great;  
            const int pos_to_move = pos < min_ind ? min_ind + num_of_small - 1 - pos_in_group_small : -1;
            if(inside) {
                if (is_small) {
                    Float value_to_move = values[pos];
                    int index_to_move = indices[pos];
                    if (pos_to_move > -1) {
                        values[pos_to_move] = value_to_move;
                        indices[pos_to_move] = index_to_move;
                    }
                }
                values[pos] = cur_value;
                indices[pos] = cur_index;
            }
        }
        split_index += num_of_small;
    }
    split_index = -reduce(sg, -split_index, sycl::ONEAPI::minimum<Float>());
    return split_index + partition_start;
}

#define INSTANTIATE(F)                                          \
    template int kernel_row_partitioning<F>( \
                            sycl::nd_item<2> item, \
                            F* values, \
                            int* indices, \
                            int partition_start,\
                            int partition_end, \
                            F pivot);

INSTANTIATE(float);
INSTANTIATE(double);

#endif

} // namespace oneapi::dal::backend::primitives
