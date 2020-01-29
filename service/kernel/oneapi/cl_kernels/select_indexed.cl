/* file: select_indexed.cl */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

    algorithmFPType get_rnd(__global const algorithmFPType * rndseq, int RndPeriod, int * count) {
        algorithmFPType ret = rndseq[(*count)++];
        if (*count >= RndPeriod) *count = 0;
        return ret;
    }

void partition( __global  algorithmFPType *pdist,
                __global int *plabels,
                int start,
                int end,
                int local_id,
                int local_size,
                algorithmFPType pivot,
                int *split,
                int *total_big_num,
                bool doset) 
{
    const int global_id  = get_global_id(0);
    int full_size = end - start;
    int last_group_size = full_size % local_size;
    int full_group_size = full_size -last_group_size;
    if(doset)
        for(int i = start + local_id; i < end; i += local_size)
            plabels[i] = i;
    for(int i = start + local_id; i < end; i += local_size) 
    {
        sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
        algorithmFPType curVal = pdist[i];
        int curLabel = plabels[i];
        unsigned char flag = curVal < pivot ? 1 : 0;
        unsigned char big_num = sub_group_reduce_add(curVal > pivot ? 1 : 0);
        unsigned char incr = sub_group_reduce_add(flag);
        int min_ind = sub_group_reduce_min(i);
        if(incr > 0) 
        {
            unsigned char shift = sub_group_scan_exclusive_add(flag);
            unsigned char old_shift= sub_group_scan_exclusive_add(flag > 0 ? 0 : 1);
            int cur_size = i > full_group_size - 1 ? last_group_size : local_size;
            if(flag) 
            {
                algorithmFPType exVal = pdist[start + *split + shift];
                int exLabel = plabels[start + *split + shift];
                pdist[start + *split + shift] = curVal;
                plabels[start + *split + shift] = curLabel;
                if(start + *split + shift < min_ind) {
                    pdist[min_ind + cur_size - 1 - shift] = exVal;
                    plabels[min_ind + cur_size - 1 - shift] = exLabel;
                }
            }
        } 
        sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
        *split += incr;
        *total_big_num += big_num;
    }

__kernel void quick_select_group( __global  algorithmFPType           *distances,
                            __global  int                       *labels,
                            __global  algorithmFPType           *nbrdistances,
                            __global  int                       *nbrlabels,
                            __global  const algorithmFPType     *rndseq,
                            int RndPeriod,
                            int N,
                            int NLast,
                            int K,
                            int BlockOffset) 
{
    const int id  = get_global_id(0);
    const int global_id = id * get_num_sub_groups() + get_sub_group_id();
    const int local_id = get_local_id(1);
    const int local_size = get_sub_group_size();
    const int max_global_id = get_global_size(0);

    if(global_id >= max_global_id) 
    {
        return;
    }

    N = (global_id == get_global_size(0) - 1) ? NLast : N;

    const int offset_in = global_id * BlockOffset;
    const int offset_out = global_id * K;
    int start = 0;
    int end = N;
    int rnd_count = 0;

    __global  algorithmFPType *pdist = &distances[offset_in];
    __global  int *plabels = &labels[offset_in];
    int count = 0;
    while(1) 
    {
        count++;
        int split = 0;
        const algorithmFPType rnd = get_rnd(rndseq, RndPeriod, &rnd_count);
        int pos = (int)(rnd * (end - start - 1));
        const algorithmFPType pivot = pdist[start + pos];
        int total_big_num = 0;
        partition(pdist, plabels, start, end, local_id, local_size, pivot, &split, &total_big_num, count == 1);
        if((start + split) == K || (!split && !total_big_num))
            break;

            if (start + split > K) end = start + split;
            if (start + split < K) start += split;
            if (count > N)
            {
                break;
            }
        }
        for (int i = 0; i < K; i++)
        {
            nbrdistances[offset_out + i] = pdist[i];
            nbrlabels[offset_out + i]    = plabels[i];
        }
    }
    for(int i = 0; i < K; i++) {
        nbrdistances[offset_out + i] = pdist[i];
        nbrlabels[offset_out + i] = plabels[i];
    }
}

);

DECLARE_SOURCE(direct_select_simd,
__kernel void direct_select_group(__global  const algorithmFPType  *pdist,
                                 __global  algorithmFPType  *pdout,
                                 __global int               *plout,
                                 int N,
                                 int NL,
                                 int BlockOffset)
{
    const int local_size = get_sub_group_size();
    const int sub_group_num = get_num_sub_groups();
    const int M = get_global_size(0);
    const int global_id  = get_global_id(0) * sub_group_num + get_sub_group_id();


    if(global_id >= M)
        return;

    const int local_id = get_sub_group_local_id();

    const __global algorithmFPType *finput = &pdist[global_id * BlockOffset];
    __global int *loutput = &plout[global_id * __K__];
    __global algorithmFPType *doutput = &pdout[global_id * __K__];

    if(global_id == get_global_size(0) - 1)
        N = NL;

    const int array_size = __K__;

    int labels[array_size];
    for(int j = 0; j < array_size; j++)
        labels[j] = -1;
    algorithmFPType distances[array_size];
    for(int j = 0; j < array_size; j++)
        distances[j] = 1.0e30;

    for(int i = local_id; i < N; i += local_size) 
    {
        algorithmFPType value = finput[i]; 
        int label = i;
        int pos = -1;

        for(int j = array_size -1 ; j > -1 ; j--)
        {
            bool do_shift = distances[j] > value;
            pos = do_shift ? j : pos;
            if(j < array_size - 1) 
            {
                distances[j + 1] = do_shift ? distances[j] : distances[j + 1];
                labels[j + 1] = do_shift ? labels[j] : labels[j + 1];
            }
        }
        if(pos != -1) {
            distances[pos] = value;
            labels[pos] = label;
        }
    }
    sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
    int bias = 0;
    algorithmFPType final_distances[array_size];
    int final_labels[array_size];
    for(int i = 0; i < array_size; i++) 
    {
        algorithmFPType min_val = sub_group_reduce_min(distances[bias]);
        bool present = (min_val == distances[bias]);
        int pos = sub_group_scan_exclusive_add(present ? 1 : 0);
        bool owner = present && pos == 0;
        final_labels[i] = -sub_group_reduce_min(owner ? -labels[bias] : 1);
        final_distances[i] = min_val;
        bias += owner ? 1 : 0;
    }
    for(int i = local_id; i < array_size; i++) {
        loutput[i] = final_labels[i];
//        printf("array size %d global_id %d local_id %d final index %d\n", array_size, global_id, local_id, loutput[i]);
    }
    
    for(int i = local_id; i < array_size; i++) 
        doutput[i] = final_distances[i];
}
);

#endif
