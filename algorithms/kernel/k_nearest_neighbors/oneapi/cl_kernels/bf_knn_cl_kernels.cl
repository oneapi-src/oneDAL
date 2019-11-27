/* file: bf_knn_cl_kernels.cl */
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
//  Implementation of BF KNN OpenCL kernels.
//--
*/

#ifndef __KNN_CL_KERNELS_CL__
#define __KNN_CL_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE(name, src) static const char* name = #src;

DECLARE_SOURCE(bf_knn_cl_kernels,

__kernel void init_distances(__global const algorithmFPType *dataSq,
                             __global       algorithmFPType *distances,
                             int N) {

    const int global_id_0  = get_global_id(0);
    const int global_id_1  = get_global_id(1);

    distances[global_id_0  + global_id_1 * N] = dataSq[global_id_0];
}

__kernel void init_categories(  __global const int *labels,
                                __global       int *categories,
                                int N,
                                int offset) {

    const int global_id_0  = get_global_id(0);
    const int global_id_1  = get_global_id(1);

    categories[global_id_0  + global_id_1 * N] = labels[offset + global_id_0];
}

__kernel void gather_partial_selection( __global const algorithmFPType *distances,
                                        __global const int *categories,
                                        __global       algorithmFPType *partialDistances,
                                        __global       int *partialCategories,
                                        int K,
                                        int Part,
                                        int TotalParts) {
    const int global_id_0  = get_global_id(0);
    const int global_id_1  = get_global_id(1);

    partialDistances[global_id_0 * K * TotalParts + Part * K + global_id_1] =
        distances[global_id_0 * K + global_id_1];
    partialCategories[global_id_0 * K * TotalParts + Part * K + global_id_1] =
        categories[global_id_0 * K + global_id_1];
}

__kernel void find_max_occurance(   __global const sortedType *data,
                                    __global       sortedType *result,
                             int K) {

    const int global_id_0  = get_global_id(0);
     __global const sortedType *array = &data[global_id_0 * K];

    sortedType maxVal = -1;
    sortedType curVal = -1;
    int maxCount = 0;
    int curCount = 0;

    for(int i = 0; i < K; i++) {
        sortedType val = array[i];
        if(val == curVal)
            curCount++;
        else {
            if(curCount > maxCount) {
                maxCount = curCount;
                maxVal = curVal;
            }
            curVal = val;
            curCount = 1;
        }
    }
    if(curCount > maxCount)
        maxVal = curVal;
    result[global_id_0] = maxVal;
}

);

#endif
