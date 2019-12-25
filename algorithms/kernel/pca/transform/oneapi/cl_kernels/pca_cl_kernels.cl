/* file: pca_transform_cl_kernels.cl */
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
//  Implementation of PCA transform OpenCL kernels.
//--
*/

#ifndef __PCA_TRANSFORM_CL_KERNELS_CL__
#define __PCA_TRANSFORM_CL_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE(name, src) static const char* name = #src;

DECLARE_SOURCE(pca_transform_cl_kernels,

__kernel void computeInvSigmas(__global algorithmFPType* rawVariances,
                               __global algorithmFPType* invSigmas)
{
    const int tid = get_global_id(0);

    if (rawVariances[tid] != (algorithmFPType)0)
    {
        invSigmas[tid] = (algorithmFPType)1 / (algorithmFPType)sqrt(rawVariances[tid]);
    }
    else
    {
        invSigmas[tid] = (algorithmFPType)0;
    }
}

__kernel void normalize(__global algorithmFPType* pCopyBlock,
						__global algorithmFPType* pRawMeans,
						__global algorithmFPType* pInvSigmas,
						const uint numMeans,
                        const uint numInvSigmas,
                        const uint maxWorkItemsPerGroup,
                        const uint numFeatures)
{
    const int tid = get_local_id(0);
    const int gid = get_group_id(0);
    const int glid = get_global_id(0);
    const int numWorkItemsPerGroup = get_local_size(0);
    const int gnum = get_num_groups(0);

    if (numFeatures > maxWorkItemsPerGroup)
    {
        uint amount = (numFeatures *  gnum) / maxWorkItemsPerGroup;

        for(uint i = 0; i < amount + 1; i++)
        {
            if (numMeans != (algorithmFPType)0)
            {
                if (glid + gnum * numWorkItemsPerGroup * i < numFeatures *  gnum)
                {
                    pCopyBlock[glid + gnum * numWorkItemsPerGroup * i] = pCopyBlock[glid + gnum * numWorkItemsPerGroup * i] - pRawMeans[(glid + gnum * numWorkItemsPerGroup * i) % numFeatures];
                }
            }
            if (numInvSigmas != (algorithmFPType)0)
            {
                if (glid + gnum * numWorkItemsPerGroup * i < numFeatures *  gnum)
                {
                    pCopyBlock[glid + gnum * numWorkItemsPerGroup * i] = pCopyBlock[glid + gnum * numWorkItemsPerGroup * i] * pInvSigmas[(glid + gnum * numWorkItemsPerGroup * i) % numFeatures];
                }
            }
        }
    }
    else
    {
        if (numMeans != (algorithmFPType)0)
        {
            pCopyBlock[gid * numWorkItemsPerGroup + tid] = pCopyBlock[gid * numWorkItemsPerGroup + tid] - pRawMeans[tid];
        }
        if (numInvSigmas != (algorithmFPType)0)
        {
            pCopyBlock[gid * numWorkItemsPerGroup + tid] = pCopyBlock[gid * numWorkItemsPerGroup + tid] * pInvSigmas[tid];
        }
    }
}

__kernel void whitening(__global algorithmFPType* pTransformedBlock, 
                        __global algorithmFPType* pInvEigenValues) 
{
    const int tid = get_local_id(0);
    const int gid = get_group_id(0);
    const int numFeatures = get_local_size(0);
    
    pTransformedBlock[gid * numFeatures + tid] = pTransformedBlock[gid * numFeatures + tid] * pInvEigenValues[tid];
    
}

);

#endif
