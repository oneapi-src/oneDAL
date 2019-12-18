/* file: radix_sort.cl */
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
//  Implementation of radix sort kernels.
//--
*/

#ifndef __RADIX_SORT_CL__
#define __RADIX_SORT_CL__

#define DECLARE_SOURCE(name, src) static const char* name = #src;

DECLARE_SOURCE(radix_sort_simd,

void swap(  __global sortedType **input,
            __global sortedType **output) 
{
    __global sortedType *tmp = *input;
    *input = *output;
    *output = tmp;
}

__kernel void radix_sort_group( __global  sortedType      *labels,
                                __global  int             *sorted,
                                __global  int             *radixbuf,
                                int N,
                                int BlockOffset) 
{

    const int global_id  = get_global_id(0);
    const int local_id = get_local_id(1);
// Code is written for a single subgroup. It's necessary to adjust the local range if idle subgoups are presented
    if(get_sub_group_id() > 0)
        return;
    const int local_size = get_sub_group_size();
    int group_aligned_size = N - N % local_size;
    int rem = N - group_aligned_size;
// radixBuf should be big enough to accumulate radix_range elements
    const int radix_range = 256;
    const int byte_range = 8;

    const int radix_count = sizeof(sortedType);
    __global sortedType *input = &labels[global_id * BlockOffset];
    __global sortedType *output = &sorted[global_id * BlockOffset];
    __global int * counters = &radixbuf[global_id * radix_range];
//  Radix sort 
   for(int i = 0; i < radix_count; i++)
   {
        __global unsigned char *cinput = (__global unsigned char *)input;
        for(int j = local_id; j < radix_range; j++)
            counters[j] = 0;
//  Count elements in sub group to write once per value
        for(int j = local_id; j < group_aligned_size + local_size; j += local_size) 
        {
            bool exists =  j < group_aligned_size || local_id < rem;
            unsigned char c = exists ? cinput[j * radix_count + i] : 0;
            int entry = -1;
            for(int k = 0; k < local_size; k++) 
            {
                bool correct = j < group_aligned_size || k < rem;
                int done = sub_group_broadcast(correct ? 0 : 1 , k);
                if(done)
                    break;
                unsigned char value = sub_group_broadcast(c, k);
                if(entry < 0 && value == c)
                    entry = k;
                int count = sub_group_reduce_add(exists && value == c ? 1 : 0);
                if(entry == local_id && entry == k) 
                {
                    counters[value] += count;
                }
            }
            sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
        }
//  Parallel scan on counters to generate offsets in place
        int offset = 0;
        for(int j = local_id; j < radix_range; j += local_size) 
        {
            int value = counters[j];
            int boundary = sub_group_scan_exclusive_add(value);
            counters[j] = offset + boundary;
            int partial_offset = sub_group_reduce_add(value);
            offset += partial_offset;
        }
        sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
       for(int j = local_id; j < group_aligned_size + local_size; j += local_size) 
       {
            bool exists =  j < group_aligned_size || local_id < rem;
            unsigned char c = exists ? cinput[j * radix_count + i] : 0;
            int local_offset = 0;
            int  done = 0;
            int entry = -1;

            for(int k = 0; k < local_size; k++) 
            {
                bool correct = j < group_aligned_size || k < rem;
                int done = sub_group_broadcast(correct ? 0 : 1 , k);
                if(done) 
                {
                    break;
                }
                int skip = sub_group_broadcast(entry < 0 ? 0 : 1, k);
                if(skip)
                    continue;
                unsigned char value = sub_group_broadcast(c, k);
                if(entry < 0 && value == c)
                    entry = k;
                int offset = sub_group_scan_exclusive_add(value == c && exists? 1 : 0);
                if(value == c) 
                {
                    local_offset = offset + counters[value];
                }
                int count = sub_group_reduce_add(value == c && exists? 1 : 0);
                if(local_id == k && entry == k) 
                {
                    counters[value] += count;
                }
            }
            sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
            if(exists)
                output[local_offset] = input[j];
        }
        swap(&input, &output);
    }
    for(int i = local_id; i < N; i += local_size) 
        output[i] = input[i];
}

);

#endif
