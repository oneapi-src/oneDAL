/* file: partition.cl */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  Implementation of partion kernels.
//--
*/

#ifndef __PARTITION_CL__
#define __PARTITION_CL__

#define DECLARE_SOURCE(name, src) static const char * name = #src;

DECLARE_SOURCE(
    kernelsPartition,

    __kernel void scan(const __global int * mask, __global int * partialSums, int nElems) {
        const int n_groups             = get_num_groups(0);
        const int n_sub_groups         = get_num_sub_groups();
        const int n_total_sub_groups   = n_sub_groups * n_groups;
        const int nElementsForSubgroup = nElems / n_total_sub_groups + !!(nElems % n_total_sub_groups);
        const int local_size           = get_sub_group_size();

        const int id           = get_local_id(0);
        const int local_id     = get_sub_group_local_id();
        const int sub_group_id = get_sub_group_id();
        const int group_id     = get_group_id(0) * n_sub_groups + sub_group_id;

        int iStart = group_id * nElementsForSubgroup;
        int iEnd   = (group_id + 1) * nElementsForSubgroup;

        if (iEnd > nElems)
        {
            iEnd = nElems;
        }

        int sum = 0;

        for (int i = iStart + local_id; i < iEnd; i += local_size)
        {
            const int value = mask[i];
            sum += sub_group_reduce_add(value);
        }

        if (local_id == 0)
        {
            partialSums[group_id] = sum;
        }
    }

    __kernel void scanIndex(const __global int * mask, const __global int * indices, __global int * partialSums, int nElems) {
        const int n_groups             = get_num_groups(0);
        const int n_sub_groups         = get_num_sub_groups();
        const int n_total_sub_groups   = n_sub_groups * n_groups;
        const int nElementsForSubgroup = nElems / n_total_sub_groups + !!(nElems % n_total_sub_groups);
        const int local_size           = get_sub_group_size();

        const int id           = get_local_id(0);
        const int local_id     = get_sub_group_local_id();
        const int sub_group_id = get_sub_group_id();
        const int group_id     = get_group_id(0) * n_sub_groups + sub_group_id;

        int iStart = group_id * nElementsForSubgroup;
        int iEnd   = (group_id + 1) * nElementsForSubgroup;

        if (iEnd > nElems)
        {
            iEnd = nElems;
        }

        int sum = 0;

        for (int i = iStart + local_id; i < iEnd; i += local_size)
        {
            const int value = mask[indices[i]];
            sum += sub_group_reduce_add(value);
        }

        if (local_id == 0)
        {
            partialSums[group_id] = sum;
        }
    }

    __kernel void sumScan(const __global int * partialSums, __global int * partialPrefixSums, __global int * totalSum, int nSubgroupSums) {
        if (get_sub_group_id() > 0) return;

        const int local_size = get_sub_group_size();
        const int local_id   = get_sub_group_local_id();

        int sum = 0;

        for (int i = local_id; i < nSubgroupSums; i += local_size)
        {
            int value            = partialSums[i];
            int boundary         = sub_group_scan_exclusive_add(value);
            partialPrefixSums[i] = sum + boundary;
            sum += sub_group_reduce_add(value);
        }

        if (local_id == 0)
        {
            totalSum[0]                      = sum;
            partialPrefixSums[nSubgroupSums] = sum;
        }
    }

    __kernel void reorder(const __global int * mask, const __global algorithmFPType * data, __global algorithmFPType * outData,
                          const __global int * partialPrefixSums, int nElems) {
        const int n_groups             = get_num_groups(0);
        const int n_sub_groups         = get_num_sub_groups();
        const int n_total_sub_groups   = n_sub_groups * n_groups;
        const int nElementsForSubgroup = nElems / n_total_sub_groups + !!(nElems % n_total_sub_groups);
        const int local_size           = get_sub_group_size();

        const int id           = get_local_id(0);
        const int local_id     = get_sub_group_local_id();
        const int sub_group_id = get_sub_group_id();
        const int group_id     = get_group_id(0) * n_sub_groups + sub_group_id;

        int iStart = group_id * nElementsForSubgroup;
        int iEnd   = (group_id + 1) * nElementsForSubgroup;

        if (iEnd > nElems)
        {
            iEnd = nElems;
        }

        int groupOffset = partialPrefixSums[group_id];
        int totalOffset = nElems - partialPrefixSums[n_total_sub_groups];

        int sum = 0;

        for (int i = iStart + local_id; i < iEnd; i += local_size)
        {
            const int part     = mask[i];
            const int boundary = groupOffset + sum + sub_group_scan_exclusive_add(part);
            if (part) outData[boundary] = data[i];
            sum += sub_group_reduce_add(part);
        }
    }

    __kernel void reorderIndex(const __global int * mask, const __global int * indices, __global int * outData,
                               const __global int * partialPrefixSums, int nElems) {
        const int n_groups             = get_num_groups(0);
        const int n_sub_groups         = get_num_sub_groups();
        const int n_total_sub_groups   = n_sub_groups * n_groups;
        const int nElementsForSubgroup = nElems / n_total_sub_groups + !!(nElems % n_total_sub_groups);
        const int local_size           = get_sub_group_size();

        const int local_id     = get_sub_group_local_id();
        const int sub_group_id = get_sub_group_id();
        const int group_id     = get_group_id(0) * n_sub_groups + sub_group_id;

        int iStart = group_id * nElementsForSubgroup;
        int iEnd   = (group_id + 1) * nElementsForSubgroup;

        if (iEnd > nElems)
        {
            iEnd = nElems;
        }

        int groupOffset = partialPrefixSums[group_id];
        int totalOffset = nElems - partialPrefixSums[n_total_sub_groups];

        int sum = 0;

        for (int i = iStart + local_id; i < iEnd; i += local_size)
        {
            const int indexi   = indices[i];
            const int part     = mask[indexi];
            const int boundary = groupOffset + sum + sub_group_scan_exclusive_add(part);
            if (part) outData[boundary] = indexi;
            sum += sub_group_reduce_add(part);
        }
    }

);

#endif
