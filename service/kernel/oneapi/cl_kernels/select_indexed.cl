/* file: select_indexed.cl */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#define DECLARE_SOURCE(name, src) static const char* name = #src;

DECLARE_SOURCE(quick_select_simd,

algorithmFPType get_rnd(   __global  const algorithmFPType *rndseq,
                int RndPeriod,
                int *count) 
{
    algorithmFPType ret = rndseq[(*count)++];
    if(*count >= RndPeriod)
        *count = 0;
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
                int *total_big_num) 
{
    const int global_id  = get_global_id(0);
    int full_size = end - start;
    int last_group_size = full_size % local_size;
    int full_group_size = full_size -last_group_size;
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
            algorithmFPType exVal = flag > 0 ? pdist[start + *split + shift] : -1.0;
            int exLabel = flag > 0 ? plabels[start + *split + shift] : -1;
            int cur_size = i > full_group_size - 1 ? last_group_size : local_size;
            if(flag) 
            {
                pdist[start + *split + shift] = curVal;
                plabels[start + *split + shift] = curLabel;
                if(start + *split + shift < min_ind) {
                    pdist[min_ind + cur_size - 1 - shift] = exVal;
                    plabels[min_ind + cur_size - 1 - shift] = exLabel;
                }
            } 
            else 
            {
                pdist[min_ind + cur_size - 1 - old_shift] = curVal;
                plabels[min_ind + cur_size - 1 - old_shift] = curLabel;
            }
        }
        sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
        *split += incr;
        *total_big_num += big_num;
    }
    *split = - sub_group_reduce_min(-(*split));
    *total_big_num = - sub_group_reduce_min(-(*total_big_num));
}

__kernel void quick_select_group( __global  algorithmFPType           *distances,
                            __global  int                       *labels,
                            __global  algorithmFPType           *nbrdistances,
                            __global  int                       *nbrlabels,
                            __global  const algorithmFPType     *rndseq,
                            int RndPeriod,
                            int N,
                            int K,
                            int BlockOffset) 
{
    const int global_id  = get_global_id(0);
    const int local_id = get_local_id(1);
    const int local_size = get_local_size(1);
    if(get_sub_group_id() > 0) 
    {
        return;
    }
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
        partition(pdist, plabels, start, end, local_id, local_size, pivot, &split, &total_big_num);
        if((start + split) == K || (!split && !total_big_num))
            break;

        if(start + split > K)
            end = start + split;
        if(start + split < K)
            start += split;
        if(count > N) 
        {
            break;
        }
    }
    for(int i = 0; i < K; i++) {
        nbrdistances[offset_out + i] = pdist[i];
        nbrlabels[offset_out + i] = plabels[i];
    }
}


);

#endif
