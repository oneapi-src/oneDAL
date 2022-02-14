/* file: sum_reducer.cl */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of sum reduction kernels.
//--
*/

#ifndef __SUM_REDUCER_CL__
#define __SUM_REDUCER_CL__

#define DECLARE_SOURCE(name, src) static const char * name = #src;

DECLARE_SOURCE(
    sum_reducer,

    __kernel void sum_singlepass(uint vectorsAreRows, __global algorithmFPType * vectors, uint nVectors, uint vectorSize,
                                 __global algorithmFPType * sums, __global algorithmFPType * sq_sums) {
        const uint local_size = get_local_size(0);

        __local algorithmFPType partial_sums[LOCAL_BUFFER_SIZE];
        __local algorithmFPType partial_sq_sums[LOCAL_BUFFER_SIZE];

        uint globalDim = 1;
        uint localDim  = nVectors;

        if (vectorsAreRows != 0)
        {
            globalDim = vectorSize;
            localDim  = 1;
        }

        uint itemId  = get_local_id(0);
        uint groupId = get_global_id(1);

        algorithmFPType el      = vectors[groupId * globalDim + itemId * localDim];
        partial_sums[itemId]    = 0;
        partial_sq_sums[itemId] = 0;

        for (uint i = itemId; i < vectorSize; i += local_size)
        {
            el = vectors[groupId * globalDim + i * localDim];
            partial_sums[itemId] += el;
            partial_sq_sums[itemId] += el * el;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint stride = local_size / 2; stride > 1; stride /= 2)
        {
            if (stride > itemId)
            {
                partial_sums[itemId] += partial_sums[itemId + stride];
                partial_sq_sums[itemId] += partial_sq_sums[itemId + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (itemId == 0)
        {
            sums[groupId]    = partial_sums[itemId] + partial_sums[itemId + 1];
            sq_sums[groupId] = partial_sq_sums[itemId] + partial_sq_sums[itemId + 1];
        }
    }

    __kernel void sum_singlesubgroup(__global algorithmFPType * vectors, uint nVectors, uint vectorSize, __global algorithmFPType * sums,
                                     __global algorithmFPType * sq_sums) {
        const uint localId   = get_local_id(1);
        const uint localSize = get_local_size(1);
        const uint groupId   = get_global_id(0);
        const uint offset    = groupId * vectorSize;
        if (get_sub_group_id() > 0) return;

        algorithmFPType partial_sums    = 0;
        algorithmFPType partial_sq_sums = 0;

        for (uint i = localId; i < vectorSize; i += localSize)
        {
            algorithmFPType el = vectors[offset + i];
            partial_sums += el;
            partial_sq_sums += el * el;
        }

        partial_sums    = sub_group_reduce_add(partial_sums);
        partial_sq_sums = sub_group_reduce_add(partial_sq_sums);

        if (localId == 0)
        {
            sums[groupId]    = partial_sums;
            sq_sums[groupId] = partial_sq_sums;
        }
    }

    void __sum_reduce_colmajor(__global const algorithmFPType * vectors, const uint nVectors, const uint vectorSize,
                               __global algorithmFPType * mergedSums, __global algorithmFPType * mergedSqSums, const uint rowPartIndex,
                               const uint rowParts, const uint colPartIndex, const uint colParts, const uint tid, const uint tnum) {
        const uint colOffset = colPartIndex * tnum;
        const uint x         = tid + colOffset;

        if (x < nVectors)
        {
            uint rowPartSize     = (vectorSize + rowParts - 1) / rowParts;
            const uint rowOffset = rowPartSize * rowPartIndex;

            if (rowPartSize + rowOffset > vectorSize)
            {
                rowPartSize = vectorSize - rowOffset;
            }

            algorithmFPType partialSums   = 0.0;
            algorithmFPType partialSqSums = 0.0;

            for (int row = 0; row < rowPartSize; row++)
            {
                const uint y             = (row + rowOffset) * nVectors;
                const algorithmFPType el = vectors[y + x];

                partialSums += el;
                partialSqSums += el * el;
            }

            mergedSums[x * rowParts + rowPartIndex]   = partialSums;
            mergedSqSums[x * rowParts + rowPartIndex] = partialSqSums;
        }
    }

    __kernel void sum_step_colmajor(__global const algorithmFPType * vectors, const uint nVectors, const uint vectorSize,
                                    __global algorithmFPType * mergedSums, __global algorithmFPType * mergedSqSums) {
        const uint tid  = get_local_id(0);
        const uint tnum = get_local_size(0);
        const uint gid  = get_group_id(0);
        const uint gnum = get_num_groups(0);

        const uint colParts = (nVectors + tnum - 1) / tnum;
        const uint rowParts = gnum / colParts;

        const uint rowPartIndex = gid / colParts;
        const uint colPartIndex = gid - rowPartIndex * colParts;

        __sum_reduce_colmajor(vectors, nVectors, vectorSize, mergedSums, mergedSqSums, rowPartIndex, rowParts, colPartIndex, colParts, tid, tnum);
    }

    __kernel void sum_final_step_rowmajor(__global const algorithmFPType * mergedSums, __global const algorithmFPType * mergedSqSums, uint nVectors,
                                          uint vectorSize, __global algorithmFPType * sums, __global algorithmFPType * sqSums) {
        const uint local_size = get_local_size(0);

        __local algorithmFPType partial_sums[LOCAL_BUFFER_SIZE];
        __local algorithmFPType partial_sq_sums[LOCAL_BUFFER_SIZE];

        uint globalDim = vectorSize;
        uint localDim  = 1;
        uint itemId    = get_local_id(0);
        uint groupId   = get_group_id(0);

        partial_sums[itemId]    = 0;
        partial_sq_sums[itemId] = 0;
        for (uint i = itemId; i < vectorSize; i += local_size)
        {
            partial_sums[itemId] += mergedSums[groupId * globalDim + i * localDim];
            partial_sq_sums[itemId] += mergedSqSums[groupId * globalDim + i * localDim];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint stride = local_size / 2; stride > 1; stride /= 2)
        {
            if (stride > itemId)
            {
                partial_sums[itemId] += partial_sums[itemId + stride];
                partial_sq_sums[itemId] += partial_sq_sums[itemId + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (itemId == 0)
        {
            sums[groupId]   = partial_sums[itemId] + partial_sums[itemId + 1];
            sqSums[groupId] = partial_sq_sums[itemId] + partial_sq_sums[itemId + 1];
        }
    }

);

#endif
