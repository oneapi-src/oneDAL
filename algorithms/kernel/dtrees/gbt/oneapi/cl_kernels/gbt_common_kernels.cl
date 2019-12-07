/* file: gbt_kernels.cl */
/*******************************************************************************
* Copyright 2019 Intel Corporation
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
//  Implementation of GBT OpenCL kernels.
//--
*/

#ifndef __GBT_KERNELS_CL__
#define __GBT_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE(name, src) static const char* name = #src;

DECLARE_SOURCE(gbt_common_kernels,

__kernel void extractColumn(const __global algorithmFPType* data,
                                  __global algorithmFPType* values,
                                  __global int* indices,
                            int featureId,
                            int nFeatures,
                            int nRows)
{
    const int id = get_global_id(0);
    values[id] = data[id * nFeatures + featureId];
    indices[id] = id;
}

uint __attribute__((overloadable)) invBits(uint x) {
    return x ^ (-(x >> 31) | 0x80000000u);
//    return x ^ 0x80000000u;
}

ulong __attribute__((overloadable)) invBits(ulong x) {
    return x ^ (-(x >> 63) | 0x8000000000000000ul);
//    return x ^ 0x8000000000000000u;
}

__kernel void radixScan(const __global radixIntType* values,
                              __global int* partialHists,
                        int nRows,
                        int bitOffset)
{
    const int RADIX_BITS = 4;

    const int n_groups = get_num_groups(0);
    const int n_sub_groups = get_num_sub_groups();
    const int n_total_sub_groups = n_sub_groups * n_groups;
    const int nElementsForSubgroup = nRows / n_total_sub_groups + !!(nRows % n_total_sub_groups);
    const int local_size = get_sub_group_size();

    const int id = get_local_id(0);
    const int local_id = get_sub_group_local_id();
    const int sub_group_id = get_sub_group_id();
    const int group_id = get_group_id(0) * n_sub_groups + sub_group_id;

    int iStart = group_id * nElementsForSubgroup;
    int iEnd = (group_id + 1) * nElementsForSubgroup;

    if (iEnd > nRows)
    {
        iEnd = nRows;
    }

    int offset[1 << RADIX_BITS];
    const int radix_range = 1 << RADIX_BITS;
    const int radix_range_1 = radix_range - 1;
    for (int i = 0; i < radix_range; i++)
    {
        offset[i] = 0;
    }

    for(int i = iStart + local_id; i < iEnd; i += local_size)
    {
        radixIntType data_bits = ((invBits(values[i]) >> bitOffset) & radix_range_1);
        for (int j = 0; j < radix_range; j++)
        {
            int value = data_bits == j;
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

__kernel void radixHistScan(const __global int* partialHists,
                                  __global int* partialPrefixHists,
                            int nSubgroupSums)
{
    const int RADIX_BITS = 4;

    if (get_sub_group_id() > 0)
        return;

    const int local_size = get_sub_group_size();
    const int local_id = get_sub_group_local_id();

    int offset[1 << RADIX_BITS];
    const int radix_range = 1 << RADIX_BITS;
    for (int i = 0; i < radix_range; i++)
    {
        offset[i] = 0;
    }

    for(int i = local_id; i < nSubgroupSums; i += local_size)
    {
        for (int j = 0; j < radix_range; j++)
        {
            int value = partialHists[i * radix_range + j];
            int boundary = sub_group_scan_exclusive_add(value);
            partialPrefixHists[i * radix_range + j] = offset[j] + boundary;
            int partial_offset = sub_group_reduce_add(value);
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

__kernel void radixReorder(const __global radixIntType* valuesSrc,
                           const __global int* indicesSrc, 
                           const __global int* partialPrefixHists,
                                 __global radixIntType* valuesDst, 
                                 __global int* indicesDst,
                           int nRows,
                           int bitOffset)
{
    const int RADIX_BITS = 4;

    const int n_groups = get_num_groups(0);
    const int n_sub_groups = get_num_sub_groups();
    const int n_total_sub_groups = n_sub_groups * n_groups;
    const int nElementsForSubgroup = nRows / n_total_sub_groups + !!(nRows % n_total_sub_groups);
    const int local_size = get_sub_group_size();

    const int id = get_local_id(0);
    const int local_id = get_sub_group_local_id();
    const int sub_group_id = get_sub_group_id();
    const int group_id = get_group_id(0) * n_sub_groups + sub_group_id;

    int iStart = group_id * nElementsForSubgroup;
    int iEnd = (group_id + 1) * nElementsForSubgroup;

    if (iEnd > nRows)
    {
        iEnd = nRows;
    }

    int offset[1 << RADIX_BITS];

    const int radix_range = 1 << RADIX_BITS;
    const int radix_range_1 = radix_range - 1;

    for (int i = 0; i < radix_range; i++)
    {
        offset[i] = partialPrefixHists[group_id * radix_range + i] + partialPrefixHists[n_total_sub_groups * radix_range + i];
    }

    for(int i = iStart + local_id; i < iEnd; i += local_size)
    {
        radixIntType data_value = valuesSrc[i];
        radixIntType data_bits = ((invBits(data_value) >> bitOffset) & radix_range_1);
        int pos_new = 0;
        for (int j = 0; j < radix_range; j++)
        {
            int value = data_bits == j;
            int boundary = sub_group_scan_exclusive_add(value);
            pos_new |= value * (offset[j] + boundary);
            int partial_offset = sub_group_reduce_add(value);
            offset[j] = offset[j] + partial_offset;
        }
        valuesDst[pos_new] = data_value;
        indicesDst[pos_new] = indicesSrc[i];
    }
}

__kernel void collectBinBorders(const __global algorithmFPType* values,
                                const __global int* binOffsets,
                                      __global algorithmFPType* binBorders)
{
    const int id = get_global_id(0);
    binBorders[id] = values[binOffsets[id]];
}

__kernel void computeBins(const __global algorithmFPType* values,
                          const __global int* indices,
                          const __global algorithmFPType* binBorders,
                                __global int* bins,
                          int nRows,
                          int nBins)
{
    const int n_groups = get_num_groups(0);
    const int n_sub_groups = get_num_sub_groups();
    const int n_total_sub_groups = n_sub_groups * n_groups;
    const int nElementsForSubgroup = nRows / n_total_sub_groups + !!(nRows % n_total_sub_groups);
    const int local_size = get_sub_group_size();

    const int id = get_local_id(0);
    const int local_id = get_sub_group_local_id();
    const int sub_group_id = get_sub_group_id();
    const int group_id = get_group_id(0) * n_sub_groups + sub_group_id;

    int iStart = group_id * nElementsForSubgroup;
    int iEnd = (group_id + 1) * nElementsForSubgroup;

    if (iEnd > nRows)
    {
        iEnd = nRows;
    }

    int curBin = 0;

    for(int i = iStart + local_id; i < iEnd; i += local_size)
    {
        algorithmFPType value = values[i];
        while (binBorders[curBin] < value) curBin++;
        bins[indices[i]] = curBin;
    }
}

__kernel void storeColumn(const __global int* data,
                                __global int* fullData,
                          int featureId,
                          int nFeatures,
                          int nRows)
{
    const int id = get_global_id(0);
    fullData[id * nFeatures + featureId] = data[id];
}

);

#endif
