/* file: select_indexed.cl */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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

/*
//++
//  Implementation of quick select kernels.
//--
*/

#ifndef __SELECT_INDEXED_CL__
#define __SELECT_INDEXED_CL__

#define DECLARE_SOURCE(name, src) static const char * name = #src;

DECLARE_SOURCE(
    quick_select_simd,

    algorithmFPType get_rnd(__global const algorithmFPType * rnd_seq, int rnd_eriod, int * count) {
        algorithmFPType ret = rnd_seq[(*count)++];
        if (*count >= rnd_eriod)
        {
            *count = 0;
        }
        return ret;
    }

    void partition_by_values(__global algorithmFPType * values, __global int * indices, int partition_start, int partition_end, int local_id,
                             int local_size, algorithmFPType pivot, int * split_index_ptr, int * great_total_ptr) {
        int full_size       = partition_end - partition_start;
        int last_group_size = full_size % local_size;
        int full_group_size = full_size - last_group_size;

        for (int i = partition_start + local_id; i < partition_end; i += local_size)
        {
            sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
            algorithmFPType cur_value  = values[i];
            int cur_index              = indices[i];
            unsigned char is_small     = cur_value < pivot ? 1 : 0;
            unsigned char num_of_great = sub_group_reduce_add(cur_value > pivot ? 1 : 0);
            unsigned char num_of_small = sub_group_reduce_add(is_small);
            int min_ind                = sub_group_reduce_min(i);
            if (num_of_small > 0)
            {
                unsigned char pos_in_group_small = sub_group_scan_exclusive_add(is_small);
                unsigned char pos_in_group_great = sub_group_scan_exclusive_add(is_small > 0 ? 0 : 1);
                int cur_size                     = i > full_group_size - 1 ? last_group_size : local_size;
                if (is_small)
                {
                    algorithmFPType value_to_move = values[partition_start + *split_index_ptr + pos_in_group_small];
                    int index_to_move             = indices[partition_start + *split_index_ptr + pos_in_group_small];

                    values[partition_start + *split_index_ptr + pos_in_group_small]  = cur_value;
                    indices[partition_start + *split_index_ptr + pos_in_group_small] = cur_index;
                    if (partition_start + *split_index_ptr + pos_in_group_small < min_ind)
                    {
                        values[min_ind + cur_size - 1 - pos_in_group_small]  = value_to_move;
                        indices[min_ind + cur_size - 1 - pos_in_group_small] = index_to_move;
                    }
                }
                else
                {
                    values[min_ind + cur_size - 1 - pos_in_group_great]  = cur_value;
                    indices[min_ind + cur_size - 1 - pos_in_group_great] = cur_index;
                }
            }
            sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
            *split_index_ptr += num_of_small;
            *great_total_ptr += num_of_great;
        }
        *split_index_ptr = -sub_group_reduce_min(-(*split_index_ptr));
        *great_total_ptr = -sub_group_reduce_min(-(*great_total_ptr));
    }

    __kernel void quick_select_group(__global algorithmFPType * in_values, __global int * in_indices, __global algorithmFPType * out_values,
                                     __global int * out_indices, __global const algorithmFPType * rnd_seq, int RndPeriod, int N, int NLast, int K,
                                     int BlockOffset) {
        const int row_id     = get_global_id(0) * get_num_sub_groups() + get_sub_group_id();
        const int local_id   = get_local_id(1);
        const int local_size = get_sub_group_size();
        const int row_number = get_global_size(0);

        if (row_id >= row_number)
        {
            return;
        }

        N = (row_id == get_global_size(0) - 1) ? NLast : N;

        const int offset_in  = row_id * BlockOffset;
        const int offset_out = row_id * K;
        int partition_start  = 0;
        int partition_end    = N;
        int rnd_count        = 0;

        __global algorithmFPType * values = &in_values[offset_in];
        __global int * indices            = &in_indices[offset_in];

        for (int i = partition_start + local_id; i < partition_end; i += local_size)
        {
            indices[i] = i;
        }

        int iteration_count = 0;
        while (1)
        {
            iteration_count++;
            int split_index             = 0;
            const algorithmFPType rnd   = get_rnd(rnd_seq, RndPeriod, &rnd_count);
            int pos                     = (int)(rnd * (partition_end - partition_start - 1));
            pos                         = pos < 0 ? 0 : pos;
            const algorithmFPType pivot = values[partition_start + pos];
            int num_of_great            = 0;
            partition_by_values(values, indices, partition_start, partition_end, local_id, local_size, pivot, &split_index, &num_of_great);

            if ((partition_start + split_index) == K || (!split_index && !num_of_great))
            {
                break;
            }
            if (partition_start + split_index > K)
            {
                partition_end = partition_start + split_index;
            }
            if (partition_start + split_index < K)
            {
                partition_start += split_index;
            }
            if (iteration_count > N)
            {
                break;
            }
        }
        for (int i = local_id; i < K; i += local_size)
        {
            out_values[offset_out + i]  = values[i];
            out_indices[offset_out + i] = indices[i];
        }
    }

);

DECLARE_SOURCE(
    direct_select_simd, __kernel void direct_select_group(__global const algorithmFPType * values_in, __global algorithmFPType * values_out,
                                                          __global int * indices_out, int N, int NL, int BlockOffset, algorithmFPType FPMax) {
        const int local_size    = get_sub_group_size();
        const int sub_group_num = get_num_sub_groups();
        const int M             = get_global_size(0);
        const int global_id     = get_global_id(0) * sub_group_num + get_sub_group_id();

        if (global_id >= M)
        {
            return;
        }

        const int local_id = get_sub_group_local_id();

        const __global algorithmFPType * finput = &values_in[global_id * BlockOffset];

        if (global_id == get_global_size(0) - 1)
        {
            N = NL;
        }

        const int array_size = __K__;
        int indices[array_size];
        for (int j = 0; j < array_size; j++)
        {
            indices[j] = -1;
        }

        algorithmFPType values[array_size];
        for (int j = 0; j < array_size; j++)
        {
            values[j] = FPMax;
        }

        for (int i = local_id; i < N; i += local_size)
        {
            algorithmFPType value = finput[i];
            int index             = i;
            int pos               = -1;

            for (int j = array_size - 1; j > -1; j--)
            {
                bool do_shift = values[j] > value;
                pos           = do_shift ? j : pos;
                if (j < array_size - 1)
                {
                    values[j + 1]  = do_shift ? values[j] : values[j + 1];
                    indices[j + 1] = do_shift ? indices[j] : indices[j + 1];
                }
            }
            if (pos != -1)
            {
                values[pos]  = value;
                indices[pos] = index;
            }
        }
        sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
        int bias = 0;
        algorithmFPType final_values[array_size];
        int final_indices[array_size];
        for (int i = 0; i < array_size; i++)
        {
            algorithmFPType min_val = sub_group_reduce_min(values[bias]);
            bool present            = (min_val == values[bias]);
            int pos                 = sub_group_scan_exclusive_add(present ? 1 : 0);
            bool owner              = present && pos == 0;
            final_indices[i]        = -sub_group_reduce_min(owner ? -indices[bias] : 1);
            final_values[i]         = min_val;
            bias += owner ? 1 : 0;
        }

        __global int * local_ind_out             = &indices_out[global_id * __K__];
        __global algorithmFPType * local_val_out = &values_out[global_id * __K__];

        for (int i = local_id; i < array_size; i += local_size)
        {
            local_ind_out[i] = final_indices[i];
        }

        for (int i = local_id; i < array_size; i += local_size)
        {
            local_val_out[i] = final_values[i];
        }
    });

#endif
