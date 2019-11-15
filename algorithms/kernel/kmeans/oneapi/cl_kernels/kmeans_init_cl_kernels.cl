/* file: kmeans_init_cl_kernels.cl */
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
//  Implementation of K-means Init OpenCL kernels.
//--
*/

#ifndef __KMEANS_INIT_CL_KERNELS_CL__
#define __KMEANS_INIT_CL_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE(name, src) static const char* name = #src;

DECLARE_SOURCE(kmeans_init_cl_kernels,

__kernel void gather_random(__global const algorithmFPType *data,
                            __global       algorithmFPType *centroids,
                            __global const int             *indices,
                            int N,
                            int K,
                            int P) {

    const int global_id_0 = get_global_id(0);

    const int local_id_1 = get_local_id(1);
    const int local_size_1 = get_local_size(1);

    int ind = indices[global_id_0];

    if (ind >= 0 && ind < N)
    {
        for (int i = local_id_1; i < P; i += local_size_1)
        {
            centroids[global_id_0 * P + i] = data[ind * P + i];
        }
    }
}

);

#endif
