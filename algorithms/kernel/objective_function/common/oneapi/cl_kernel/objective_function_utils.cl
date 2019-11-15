/* file: objective_function_utils.cl */
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
//  Implementation of Log Loss OpenCL kernels.
//--
*/

#ifndef __OBJECTIVE_FUCTION_KERNELS_CL__
#define __OBJECTIVE_FUCTION_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE_DAAL(name, src) static const char* (name) = #src;

DECLARE_SOURCE_DAAL(clKernelObjectiveFunction,


inline void __sum(__global algorithmFPType* partialSums,
    __local algorithmFPType* localSum)
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

__kernel void regularization(const __global algorithmFPType* const beta, const uint nBeta, const uint n,
    __global algorithmFPType* partialSums, const algorithmFPType l1, const algorithmFPType l2)
{
  __local algorithmFPType localSum[LOCAL_SUM_SIZE];

    const uint global_id = get_global_id(0);
    const uint local_id = get_local_id(0);

    if (global_id % nBeta == 0 || global_id >= n)
    {
        localSum[local_id] = (algorithmFPType)0;
    }
    else
    {
        localSum[local_id] = l1 * fabs(beta[global_id]) + l2 * beta[global_id] * beta[global_id];
    }

    __sum(partialSums, localSum);
}

__kernel void transpose(const __global float* x,
    __global float* xt, const int n, const int m)
{
    const uint i = get_global_id(0);
    const uint j = get_global_id(1);
    xt[i + m*j] = x[i*n + j];
}


__kernel void setElem(const uint index, const algorithmFPType elem,
    __global algorithmFPType* buffer)
{
    buffer[index] = elem;
}

__kernel void setColElem(const uint icol, const algorithmFPType elem,
    __global algorithmFPType* buffer, const uint ld)
{
    const uint i = get_global_id(0);
    buffer[i*ld + icol] = elem;
}

__kernel void subVectors(const __global algorithmFPType* const x,
    const __global algorithmFPType* const y, __global algorithmFPType* c)
{
    const uint i = get_global_id(0);
    c[i] = x[i] - y[i];
}

__kernel void addVectorScalar(__global algorithmFPType* x, const algorithmFPType alpha)
{
    const uint i = get_global_id(0);
    x[i] += alpha;
}

__kernel void addVectorScalar2(__global algorithmFPType* x,
    const __global algorithmFPType* const y, const uint id)
{
    const uint i = get_global_id(0);
    x[i] += y[id];
}

// TODO: replace local sum reduction
__kernel void sumReduction(const __global algorithmFPType* const values,
    const uint n, __global algorithmFPType* partialSums)
{
    __local algorithmFPType localSum[LOCAL_SUM_SIZE];

    uint local_id = get_local_id(0);
    uint global_id = get_global_id(0);

    if (global_id >= n)
    {
        localSum[local_id] = (algorithmFPType)0;
    }
    else
    {
        localSum[local_id] = values[global_id];
    }

    __sum(partialSums, localSum);
}

__kernel void getXY(const __global algorithmFPType* const x,
    const __global algorithmFPType* const y,
    const __global int* const ind,
    const uint ldx, const algorithmFPType interceptValue,
    __global algorithmFPType* newX,
    __global algorithmFPType* newY)
{
    const uint index = get_global_id(1);
    const uint jCol = get_global_id(0);

    const int iRow = ind[index];

    const __global algorithmFPType* const xi = &x[iRow*ldx];
    __global algorithmFPType* newXi = &newX[index*(ldx + 1)];

    newXi[jCol + 1] = xi[jCol];

    if (jCol == 0)
    {
        newY[index] = y[iRow];
        newXi[0] = interceptValue;
    }
}

);

#undef DECLARE_SOURCE_DAAL

#endif
