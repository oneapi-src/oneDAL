/* file: radix_sort.cl */
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
//  Implementation of radix sort kernels.
//--
*/

#ifndef __RADIX_SORT_CL__
#define __RADIX_SORT_CL__

#define DECLARE_SOURCE(name, src) static const char * name = #src;

DECLARE_SOURCE(
    radix_sort_simd,

    void swap(__global sortedType ** input, __global sortedType ** output) {
        __global sortedType * tmp = *input;
        *input                    = *output;
        *output                   = tmp;
    }

    uint __attribute__((overloadable)) invBits(uint x) {
        return x ^ (-(x >> 31) | 0x80000000u);
        //    return x ^ 0x80000000u;
    }

    ulong __attribute__((overloadable)) invBits(ulong x) {
        return x ^ (-(x >> 63) | 0x8000000000000000ul);
        //    return x ^ 0x8000000000000000u;
    }

    __kernel void radix_sort_group(__global sortedType * labels, __global int * sorted, __global int * radixbuf, int N, int BlockOffset) {
        const int global_id = get_global_id(0);
        const int local_id  = get_local_id(1);
        // Code is written for a single subgroup. It's necessary to adjust the local range if idle subgoups are presented
        if (get_sub_group_id() > 0) return;
        const int local_size   = get_sub_group_size();
        int group_aligned_size = N - N % local_size;
        int rem                = N - group_aligned_size;
        // radixBuf should be big enough to accumulate radix_range elements
        const int radix_range = 256;
        const int byte_range  = 8;

        const int radix_count        = sizeof(sortedType);
        __global sortedType * input  = &labels[global_id * BlockOffset];
        __global sortedType * output = &sorted[global_id * BlockOffset];
        __global int * counters      = &radixbuf[global_id * radix_range];
        //  Radix sort
        for (int i = 0; i < radix_count; i++)
        {
            __global unsigned char * cinput = (__global unsigned char *)input;
            for (int j = local_id; j < radix_range; j++) counters[j] = 0;
            //  Count elements in sub group to write once per value
            for (int j = local_id; j < group_aligned_size + local_size; j += local_size)
            {
                bool exists     = j < group_aligned_size || local_id < rem;
                unsigned char c = exists ? cinput[j * radix_count + i] : 0;
                int entry       = -1;
                for (int k = 0; k < local_size; k++)
                {
                    bool correct = j < group_aligned_size || k < rem;
                    int done     = sub_group_broadcast(correct ? 0 : 1, k);
                    if (done) break;
                    unsigned char value = sub_group_broadcast(c, k);
                    if (entry < 0 && value == c) entry = k;
                    int count = sub_group_reduce_add(exists && value == c ? 1 : 0);
                    if (entry == local_id && entry == k)
                    {
                        counters[value] += count;
                    }
                }
                sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
            }
            //  Parallel scan on counters to generate offsets in place
            int offset = 0;
            for (int j = local_id; j < radix_range; j += local_size)
            {
                int value          = counters[j];
                int boundary       = sub_group_scan_exclusive_add(value);
                counters[j]        = offset + boundary;
                int partial_offset = sub_group_reduce_add(value);
                offset += partial_offset;
            }
            sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
            for (int j = local_id; j < group_aligned_size + local_size; j += local_size)
            {
                bool exists      = j < group_aligned_size || local_id < rem;
                unsigned char c  = exists ? cinput[j * radix_count + i] : 0;
                int local_offset = 0;
                int done         = 0;
                int entry        = -1;

                for (int k = 0; k < local_size; k++)
                {
                    bool correct = j < group_aligned_size || k < rem;
                    int done     = sub_group_broadcast(correct ? 0 : 1, k);
                    if (done)
                    {
                        break;
                    }
                    int skip = sub_group_broadcast(entry < 0 ? 0 : 1, k);
                    if (skip) continue;
                    unsigned char value = sub_group_broadcast(c, k);
                    if (entry < 0 && value == c) entry = k;
                    int offset = sub_group_scan_exclusive_add(value == c && exists ? 1 : 0);
                    if (value == c)
                    {
                        local_offset = offset + counters[value];
                    }
                    int count = sub_group_reduce_add(value == c && exists ? 1 : 0);
                    if (local_id == k && entry == k)
                    {
                        counters[value] += count;
                    }
                }
                sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
                if (exists) output[local_offset] = input[j];
            }
            swap(&input, &output);
        }
        for (int i = local_id; i < N; i += local_size) output[i] = input[i];
    }

    __kernel void radixScan(const __global radixIntType * values, __global int * partialHists, int nRows, int bitOffset) {
        const int RADIX_BITS = 4;

        const int n_groups             = get_num_groups(0);
        const int n_sub_groups         = get_num_sub_groups();
        const int n_total_sub_groups   = n_sub_groups * n_groups;
        const int nElementsForSubgroup = nRows / n_total_sub_groups + !!(nRows % n_total_sub_groups);
        const int local_size           = get_sub_group_size();

        const int id           = get_local_id(0);
        const int local_id     = get_sub_group_local_id();
        const int sub_group_id = get_sub_group_id();
        const int group_id     = get_group_id(0) * n_sub_groups + sub_group_id;

        int iStart = group_id * nElementsForSubgroup;
        int iEnd   = (group_id + 1) * nElementsForSubgroup;

        if (iEnd > nRows)
        {
            iEnd = nRows;
        }

        int offset[1 << RADIX_BITS];
        const int radix_range   = 1 << RADIX_BITS;
        const int radix_range_1 = radix_range - 1;
        for (int i = 0; i < radix_range; i++)
        {
            offset[i] = 0;
        }

        for (int i = iStart + local_id; i < iEnd; i += local_size)
        {
            radixIntType data_bits = ((invBits(values[i]) >> bitOffset) & radix_range_1);
            for (int j = 0; j < radix_range; j++)
            {
                int value          = data_bits == j;
                int partial_offset = sub_group_reduce_add(value);
                offset[j] += partial_offset;
            }
        }

        if (local_id == 0)
        {
            for (int j = 0; j < radix_range; j++)
            {
                partialHists[group_id * radix_range + j] = offset[j];
            }
        }
    }

    __kernel void radixHistScan(const __global int * partialHists, __global int * partialPrefixHists, int nSubgroupSums) {
        const int RADIX_BITS = 4;

        if (get_sub_group_id() > 0) return;

        const int local_size = get_sub_group_size();
        const int local_id   = get_sub_group_local_id();

        int offset[1 << RADIX_BITS];
        const int radix_range = 1 << RADIX_BITS;
        for (int i = 0; i < radix_range; i++)
        {
            offset[i] = 0;
        }

        for (int i = local_id; i < nSubgroupSums; i += local_size)
        {
            for (int j = 0; j < radix_range; j++)
            {
                int value                               = partialHists[i * radix_range + j];
                int boundary                            = sub_group_scan_exclusive_add(value);
                partialPrefixHists[i * radix_range + j] = offset[j] + boundary;
                int partial_offset                      = sub_group_reduce_add(value);
                offset[j] += partial_offset;
            }
        }

        if (local_id == 0)
        {
            int totalSum = 0;
            for (int j = 0; j < radix_range; j++)
            {
                partialPrefixHists[nSubgroupSums * radix_range + j] = totalSum;
                totalSum += offset[j];
            }
        }
    }

    __kernel void radixReorder(const __global radixIntType * valuesSrc, const __global int * indicesSrc, const __global int * partialPrefixHists,
                               __global radixIntType * valuesDst, __global int * indicesDst, int nRows, int bitOffset) {
        const int RADIX_BITS = 4;

        const int n_groups             = get_num_groups(0);
        const int n_sub_groups         = get_num_sub_groups();
        const int n_total_sub_groups   = n_sub_groups * n_groups;
        const int nElementsForSubgroup = nRows / n_total_sub_groups + !!(nRows % n_total_sub_groups);
        const int local_size           = get_sub_group_size();

        const int id           = get_local_id(0);
        const int local_id     = get_sub_group_local_id();
        const int sub_group_id = get_sub_group_id();
        const int group_id     = get_group_id(0) * n_sub_groups + sub_group_id;

        int iStart = group_id * nElementsForSubgroup;
        int iEnd   = (group_id + 1) * nElementsForSubgroup;

        if (iEnd > nRows)
        {
            iEnd = nRows;
        }

        int offset[1 << RADIX_BITS];

        const int radix_range   = 1 << RADIX_BITS;
        const int radix_range_1 = radix_range - 1;

        for (int i = 0; i < radix_range; i++)
        {
            offset[i] = partialPrefixHists[group_id * radix_range + i] + partialPrefixHists[n_total_sub_groups * radix_range + i];
        }

        for (int i = iStart + local_id; i < iEnd; i += local_size)
        {
            radixIntType data_value = valuesSrc[i];
            radixIntType data_bits  = ((invBits(data_value) >> bitOffset) & radix_range_1);
            int pos_new             = 0;
            for (int j = 0; j < radix_range; j++)
            {
                int value    = data_bits == j;
                int boundary = sub_group_scan_exclusive_add(value);
                pos_new |= value * (offset[j] + boundary);
                int partial_offset = sub_group_reduce_add(value);
                offset[j]          = offset[j] + partial_offset;
            }
            valuesDst[pos_new]  = data_value;
            indicesDst[pos_new] = indicesSrc[i];
        }
    }

);

#endif
