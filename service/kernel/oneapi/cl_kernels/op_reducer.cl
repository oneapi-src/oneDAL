/* file: op_reduce.cl */
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
//  Implementation of reduction kernels.
//--
*/

#ifndef __OP_REDUCER_CL__
#define __OP_REDUCER_CL__

#define DECLARE_SOURCE(name, src) static const char * name = #src;

DECLARE_SOURCE(
    op_reduce,

    inline algorithmFPType pow2(const algorithmFPType x) { return x * x; }

    inline algorithmFPType none(const algorithmFPType x) { return x; }

    inline algorithmFPType sum(const algorithmFPType x, const algorithmFPType y) { return x + y; }

    __kernel void reduceSinglepass(uint vectorsAreRows, __global algorithmFPType * vectors, uint nVectors, uint vectorSize,
                                   __global algorithmFPType * reduces) {
        const uint local_size = get_local_size(0);

        __local algorithmFPType partialReduces[LOCAL_BUFFER_SIZE];

        uint globalDim = 1;
        uint localDim  = nVectors;

        if (vectorsAreRows != 0)
        {
            globalDim = vectorSize;
            localDim  = 1;
        }

        uint itemId  = get_local_id(0);
        uint groupId = get_global_id(1);

        algorithmFPType el     = vectors[groupId * globalDim + itemId * localDim];
        partialReduces[itemId] = INIT_VALUE;

        for (uint i = itemId; i < vectorSize; i += local_size)
        {
            el                     = vectors[groupId * globalDim + i * localDim];
            partialReduces[itemId] = BINARY_OP(partialReduces[itemId], UNARY_OP(el));
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint stride = local_size / 2; stride > 1; stride /= 2)
        {
            if (stride > itemId)
            {
                partialReduces[itemId] = BINARY_OP(partialReduces[itemId], partialReduces[itemId + stride]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (itemId == 0)
        {
            reduces[groupId] = BINARY_OP(partialReduces[itemId], partialReduces[itemId + 1]);
        }
    }

    void reduceReduceColmajor(__global const algorithmFPType * vectors, const uint nVectors, const uint vectorSize,
                              __global algorithmFPType * mergedReduce, const uint rowPartIndex, const uint rowParts, const uint colPartIndex,
                              const uint colParts, const uint tid, const uint tnum) {
        const int colOffset = colPartIndex * tnum;
        const int x         = tid + colOffset;

        if (x < nVectors)
        {
            int rowPartSize     = (vectorSize + rowParts - 1) / rowParts;
            const int rowOffset = rowPartSize * rowPartIndex;

            if (rowPartSize + rowOffset > vectorSize)
            {
                rowPartSize = vectorSize - rowOffset;
            }

            algorithmFPType partialRes = INIT_VALUE;

            for (int row = 0; row < rowPartSize; row++)
            {
                const int y              = (row + rowOffset) * nVectors;
                const algorithmFPType el = vectors[y + x];

                partialRes = BINARY_OP(partialRes, UNARY_OP(el));
            }

            mergedReduce[x * rowParts + rowPartIndex] = partialRes;
        }
    }

    __kernel void reduceStepColmajor(__global const algorithmFPType * vectors, const uint nVectors, const uint vectorSize,
                                     __global algorithmFPType * mergedReduce) {
        const int tid  = get_local_id(0);
        const int tnum = get_local_size(0);
        const int gid  = get_group_id(0);
        const int gnum = get_num_groups(0);

        const int colParts = (nVectors + tnum - 1) / tnum;
        const int rowParts = gnum / colParts;

        const int rowPartIndex = gid / colParts;
        const int colPartIndex = gid - rowPartIndex * colParts;

        reduceReduceColmajor(vectors, nVectors, vectorSize, mergedReduce, rowPartIndex, rowParts, colPartIndex, colParts, tid, tnum);
    }

    __kernel void reduceFinalStepRowmajor(__global const algorithmFPType * mergedReduce, uint nVectors, uint vectorSize,
                                          __global algorithmFPType * reduces) {
        const uint local_size = get_local_size(0);

        __local algorithmFPType partialReduces[LOCAL_BUFFER_SIZE];

        uint globalDim = vectorSize;
        uint localDim  = 1;
        uint itemId    = get_local_id(0);
        uint groupId   = get_group_id(0);

        partialReduces[itemId] = INIT_VALUE;
        for (uint i = itemId; i < vectorSize; i += local_size)
        {
            partialReduces[itemId] = BINARY_OP(partialReduces[itemId], mergedReduce[groupId * globalDim + i * localDim]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint stride = local_size / 2; stride > 1; stride >>= 2)
        {
            if (stride > itemId)
            {
                partialReduces[itemId] = BINARY_OP(partialReduces[itemId], partialReduces[itemId + stride]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (itemId == 0)
        {
            reduces[groupId] = BINARY_OP(partialReduces[itemId], partialReduces[itemId + 1]);
        }
    }

);

#endif
