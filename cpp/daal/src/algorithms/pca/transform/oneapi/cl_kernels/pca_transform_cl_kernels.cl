/* file: pca_transform_cl_kernels.cl */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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
//  Implementation of PCA transform OpenCL kernels.
//--
*/

#ifndef __PCA_TRANSFORM_CL_KERNELS_CL__
#define __PCA_TRANSFORM_CL_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE(name, src) static const char * name = #src;

DECLARE_SOURCE(
    pca_transform_cl_kernels,

    __kernel void computeInvSigmas(__global const algorithmFPType * rawVariances, __global algorithmFPType * invSigmas) {
        const unsigned int tid        = get_global_id(0);
        const algorithmFPType epsilon = 1e-10;

        /*Case when rawVariances[tid] < 0 is handled inside compute method*/
        if (rawVariances[tid] > epsilon)
        {
            invSigmas[tid] = (algorithmFPType)1 / (algorithmFPType)sqrt(rawVariances[tid]);
        }
        else
        {
            invSigmas[tid] = (algorithmFPType)0;
        }
    }

    __kernel void normalize(__global algorithmFPType * copyBlock, __global const algorithmFPType * rawMeans,
                            __global const algorithmFPType * invSigmas, const char hasMeans, const char hasInvSigmas, const uint maxWorkItemsPerGroup,
                            const uint numFeatures) {
        const unsigned int glid                 = get_global_id(0);
        const unsigned int numWorkItemsPerGroup = get_local_size(0);
        const unsigned int numVec               = get_num_groups(0);

        uint numOfDataItemsProcessedByWI = numFeatures / maxWorkItemsPerGroup;

        for (uint i = 0; i < numOfDataItemsProcessedByWI + 1; i++)
        {
            const int dataId  = glid + numVec * numWorkItemsPerGroup * i;
            const int meansId = dataId % numFeatures;
            if (dataId < numFeatures * numVec)
            {
                if (hasMeans)
                {
                    copyBlock[dataId] = copyBlock[dataId] - rawMeans[meansId];
                }
                if (hasInvSigmas)
                {
                    copyBlock[dataId] = copyBlock[dataId] * invSigmas[meansId];
                }
            }
        }
    }

    __kernel void whitening(__global algorithmFPType * transformedBlock, __global const algorithmFPType * invEigenValues,
                            const uint maxWorkItemsPerGroup, const uint numComponents) {
        const int glid                 = get_global_id(0);
        const int numWorkItemsPerGroup = get_local_size(0);
        const int numVec               = get_num_groups(0);

        uint numOfDataItemsProcessedByWI = numComponents / maxWorkItemsPerGroup;
        for (uint i = 0; i < numOfDataItemsProcessedByWI + 1; i++)
        {
            const int dataId   = glid + numVec * numWorkItemsPerGroup * i;
            const int eigValId = dataId % numComponents;
            if (dataId < numComponents * numVec)
            {
                transformedBlock[dataId] = transformedBlock[dataId] * invEigenValues[eigValId];
            }
        }
    }

);

#endif
