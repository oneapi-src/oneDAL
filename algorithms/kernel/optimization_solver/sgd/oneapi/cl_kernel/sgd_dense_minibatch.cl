/* file: sgd_dense_minibatch.cl */
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
//  Implementation of SGD dense minibatch OpenCL kernels.
//--
*/

#ifndef __SGD_DENSE_MINIBATCH_KERNELS_CL__
#define __SGD_DENSE_MINIBATCH_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE_DAAL(name, src) static const char* (name) = #src;

DECLARE_SOURCE_DAAL(clKernelSGDMiniBatch,

inline void __sum(__global algorithmFPType* partialSums, __local algorithmFPType *localSum)
{
    const uint global_group_id = get_group_id(0);
    const uint group_size = get_local_size(0);
    const uint local_id = get_local_id(0);

    for (uint stride = group_size / 2; stride > 0; stride /=2)
    {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_id < stride)
        {
            localSum[local_id] += localSum[local_id + stride];
        }
    }

    if (local_id == 0)
    {
        partialSums[global_group_id] = localSum[0];

    }
}

__kernel void makeStep(
    const __global algorithmFPType* const gradient,
    const __global algorithmFPType* const prevWorkValue,
    __global algorithmFPType* workValue,
    const algorithmFPType learningRate,
    const algorithmFPType consCoeff)
{
    const uint j = get_global_id(0);

    workValue[j] = workValue[j] - learningRate * (gradient[j] + consCoeff*(workValue[j] - prevWorkValue[j]));
}

__kernel void sumSq(
    const __global algorithmFPType *const x,
    const uint n,
    __global algorithmFPType *partialSums)
{
    __local algorithmFPType localSum[LOCAL_SUM_SIZE];
    const uint global_id = get_global_id(0);
    const uint local_id = get_local_id(0);

    if (global_id >= n)
    {
        localSum[local_id] = (algorithmFPType)0;
    }
    else
    {
        localSum[local_id] = x[global_id]*x[global_id];
    }

    __sum(partialSums, localSum);
}

);

#undef DECLARE_SOURCE_DAAL

#endif
