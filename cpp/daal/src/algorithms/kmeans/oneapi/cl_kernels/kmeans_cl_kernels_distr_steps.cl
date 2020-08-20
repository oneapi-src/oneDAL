/* file: kmeans_cl_kernels_distr_steps.cl */
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
//  Implementation of K-means OpenCL kernels.
//--
*/

#ifndef __KMEANS_CL_KERNELS_DISTR_STEPS_CL__
#define __KMEANS_CL_KERNELS_DISTR_STEPS_CL__

#include <string.h>

#define DECLARE_SOURCE(name, src) static const char * name = #src;

DECLARE_SOURCE(
    kmeans_cl_kernels_distr_steps,

    __kernel void init_clusters(__global int * partialCentroidsCounters, __global algorithmFPType * partialCentroids, __global int * cCounters,
                                __global algorithmFPType * centroids, int P) {
        const int global_id                 = get_global_id(0);
        const int local_id                  = get_local_id(1);
        centroids[global_id * P + local_id] = partialCentroids[global_id * P + local_id];
        if (local_id == 0) cCounters[global_id] = partialCentroidsCounters[global_id];
    }

    __kernel void update_clusters(__global int * partialCentroidsCounters, __global algorithmFPType * partialCentroids, __global int * cCounters,
                                  __global algorithmFPType * centroids, int P) {
        const int global_id                 = get_global_id(0);
        const int local_id                  = get_local_id(1);
        const int oldN                      = partialCentroidsCounters[global_id];
        const int newN                      = cCounters[global_id];
        const algorithmFPType oldContrib    = oldN * partialCentroids[global_id * P + local_id];
        const algorithmFPType newContrib    = newN * centroids[global_id * P + local_id];
        centroids[global_id * P + local_id] = (oldContrib + newContrib) / (oldN + newN);
        if (local_id == 0) cCounters[global_id] = oldN + newN;
    }

    __kernel void init_candidates(__global int * partialCandidates, __global algorithmFPType * partialCValues, __global int * candidates,
                                  __global algorithmFPType * cValues, int K) {
        const int local_id   = get_local_id(1);
        const int local_size = get_local_size(1);
        for (int i = local_id; i < K; i += local_size)
        {
            candidates[i] = partialCandidates[i];
            cValues[i]    = partialCValues[i];
        }
    }

    __kernel void update_candidates(__global int * partialCandidates, __global algorithmFPType * partialCValues, __global int * candidates,
                                    __global algorithmFPType * cValues, int K) {
        for (int i = K - 1; i >= 0; i--)
        {
            int j;
            algorithmFPType last = cValues[K - 1];
            for (j = K - 2; j >= 0 && cValues[j] < partialCValues[i]; j--)
            {
                cValues[j + 1]    = cValues[j];
                candidates[j + 1] = candidates[j];
            }

            if (j != K - 2 || last < partialCValues[i])
            {
                cValues[j + 1]    = partialCValues[i];
                candidates[j + 1] = partialCandidates[i];
            }
        }
    }

);

#endif
