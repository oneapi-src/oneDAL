/* file: df_kernels.cl */
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
//  Implementation of decision forest OpenCL kernels.
//--
*/

#ifndef __DF_KERNELS_CL__
#define __DF_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE(name, src) static const char * name = #src;

DECLARE_SOURCE(
    df_common_kernels,

    __kernel void extractColumn(const __global algorithmFPType * data, __global algorithmFPType * values, __global int * indices, int featureId,
                                int nFeatures, int nRows) {
        const int id = get_global_id(0);
        values[id]   = data[id * nFeatures + featureId];
        indices[id]  = id;
    }

    __kernel void collectBinBorders(const __global algorithmFPType * values, const __global int * binOffsets, __global algorithmFPType * binBorders) {
        const int id   = get_global_id(0);
        binBorders[id] = values[binOffsets[id]];
    }

    __kernel void computeBins(const __global algorithmFPType * values, const __global int * indices, const __global algorithmFPType * binBorders,
                              __global int * bins, int nRows, int nBins) {
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

        iEnd = (iEnd > nRows) ? nRows : iEnd;

        int curBin = 0;

        for (int i = iStart + local_id; i < iEnd; i += local_size)
        {
            algorithmFPType value = values[i];
            while (binBorders[curBin] < value) curBin++;
            bins[indices[i]] = curBin;
        }
    }

    __kernel void storeColumn(const __global int * data, __global int * fullData, int featureId, int nFeatures, int nRows) {
        const int id                         = get_global_id(0);
        fullData[id * nFeatures + featureId] = data[id];
    }

);

#endif
