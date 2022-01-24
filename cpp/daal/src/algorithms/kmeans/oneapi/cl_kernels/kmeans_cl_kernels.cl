/* file: kmeans_cl_kernels.cl */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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

#ifndef __KMEANS_CL_KERNELS_CL__
#define __KMEANS_CL_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE(name, src) static const char * name = #src;

DECLARE_SOURCE(
    kmeans_cl_kernels,

    void __sum_reduce(__local algorithmFPType * local_sum, uint local_id, uint local_size) {
        for (uint stride = local_size / 2; stride > 0; stride /= 2)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            if (local_id < stride)
            {
                local_sum[local_id] += local_sum[local_id + stride];
            }
        }
    }

    __kernel void reduce_assignments(__global const algorithmFPType * centroidsSq, __global const algorithmFPType * distances, int N, int K,
                                     algorithmFPType huge, __global int * assignments, __global algorithmFPType * mindistances) {
        const int global_id = get_global_id(0);
        const int size      = get_local_size(1);
        const int local_id  = get_local_id(1);

        algorithmFPType minVal = huge;
        int minIdx             = -1;
        for (int i = local_id; i < K; i += size)
        {
            algorithmFPType dist   = distances[global_id + N * i];
            algorithmFPType sq     = centroidsSq[i];
            algorithmFPType curVal = dist + 0.5 * sq;
            minIdx                 = curVal < minVal ? i : minIdx;
            minVal                 = curVal < minVal ? curVal : minVal;
        }
        algorithmFPType groupMin = sub_group_reduce_min(minVal);
        minIdx                   = sub_group_reduce_min(minVal == groupMin ? minIdx : K);
        if (local_id == 0)
        {
            assignments[global_id]  = minIdx;
            mindistances[global_id] = groupMin;
        }
    }

    int insert_subgroup_shift_right(__local int * data, int index, int offset, int newVal, int sub_group_id, int sub_group_size, int rem) {
        if (index < 0) index = 0;
        int ret     = data[offset + sub_group_size - 1];
        int curVal  = data[offset + sub_group_id];
        int prevVal = curVal;
        int delta   = index < sub_group_id ? 1 : 0;
        int res     = intel_sub_group_shuffle_up(prevVal, curVal, delta);
        if (rem == 0 || sub_group_id < rem)
        {
            int v                       = index == sub_group_id ? newVal : res;
            data[offset + sub_group_id] = v;
        }
        return ret;
    }

    algorithmFPType insert_subgroup_shift_right_fp(__local algorithmFPType * data, int index, int offset, algorithmFPType newVal, int sub_group_id,
                                                   int sub_group_size, int rem) {
        if (index < 0) index = 0;
        algorithmFPType ret     = data[offset + sub_group_size - 1];
        algorithmFPType curVal  = data[offset + sub_group_id];
        algorithmFPType prevVal = curVal;
        int delta               = index < sub_group_id ? 1 : 0;
        algorithmFPType res     = intel_sub_group_shuffle_up(prevVal, curVal, delta);
        if (rem == 0 || sub_group_id < rem)
        {
            data[offset + sub_group_id] = index == sub_group_id ? newVal : res;
        }
        return ret;
    }

    void insert_shift_right(__local int * data, int index, int offset, int newVal, int sub_group_num, int rem, int sub_group_id, int sub_group_size) {
        for (int i = index / sub_group_size; i < sub_group_num; i++)
        {
            int curRem = (i == sub_group_num - 1 && rem > 0) ? rem : sub_group_size;
            newVal = insert_subgroup_shift_right(data, index - i * sub_group_size, offset + i * sub_group_size, newVal, sub_group_id, sub_group_size,
                                                 curRem);
        }
    }

    void insert_shift_right_fp(__local algorithmFPType * data, int index, int offset, algorithmFPType newVal, int sub_group_num, int rem,
                               int sub_group_id, int sub_group_size) {
        for (int i = index / sub_group_size; i < sub_group_num; i++)
        {
            int curRem = (i == sub_group_num - 1 && rem > 0) ? rem : sub_group_size;
            newVal     = insert_subgroup_shift_right_fp(data, index - i * sub_group_size, offset + i * sub_group_size, newVal, sub_group_id,
                                                    sub_group_size, curRem);
        }
    }

    __kernel void partial_candidates(__global const int * assignments, __global const algorithmFPType * mindistances,
                                     __global const algorithmFPType * distSq, __global const int * candidates,
                                     __global const algorithmFPType * candidateDistances, __global int * candidates_tmp,
                                     __global algorithmFPType * candidateDistances_tmp, int N, int K, int Reset) {
        const int global_id  = get_global_id(0);
        const int gsize      = get_global_size(0);
        const int lsize      = get_sub_group_size();
        const int local_id   = get_sub_group_local_id();
        const int local_id_2 = get_local_id(1);
        const int sg_size    = get_max_sub_group_size();
        if (lsize < sg_size || get_sub_group_id() > 0 || global_id >= NUM_PARTS_CND) return;

        const algorithmFPType HUGE = -1.0e-15;

        int numgrp    = K / lsize;
        const int rem = K % lsize;
        if (rem > 0 || numgrp == 0) numgrp++;

        __local algorithmFPType maxDist[CND_PART_SIZE];
        __local int maxItem[CND_PART_SIZE];

        for (int i = 0; i < numgrp; i++)
        {
            if (i < numgrp - 1 || local_id < rem || rem == 0)
            {
                int offset                = local_id + lsize * i;
                algorithmFPType initValue = (global_id == 0 && !Reset) ? candidateDistances[offset] : HUGE;
                int initIndex             = (global_id == 0 && !Reset) ? candidates[offset] : -1;
                maxDist[offset]           = initValue;
                maxItem[offset]           = initIndex;
            }
        }
        for (int iblock = global_id; iblock < N; iblock += gsize)
        {
            algorithmFPType newVal = 2.0 * mindistances[iblock] + distSq[iblock];
            if (newVal <= maxDist[K - 1]) continue;
            int valCentroid = iblock;
            int maxInd      = -1;
            for (int i = 0; i < numgrp; i++)
            {
                algorithmFPType curVal = HUGE;
                if (i < numgrp - 1 || local_id < rem) curVal = maxDist[local_id + i * lsize] - newVal;
                int valInd = curVal > 0.0 ? 1 : local_id - lsize;
                int locInd = sub_group_reduce_min(valInd);
                if (locInd < 0)
                {
                    maxInd = i * lsize + lsize + locInd;
                    break;
                }
            }
            if (maxInd > -1)
            {
                insert_shift_right_fp(maxDist, maxInd, 0, newVal, numgrp, rem, local_id, lsize);
                insert_shift_right(maxItem, maxInd, 0, valCentroid, numgrp, rem, local_id, lsize);
            }
        }

        if (local_id == 0 && global_id < NUM_PARTS_CND)
        {
            for (int i = 0; i < K; i++)
            {
                candidateDistances_tmp[global_id * K + i] = maxDist[i];
                candidates_tmp[global_id * K + i]         = maxItem[i];
            }
        }
    }

    __kernel void merge_candidates(__global int * candidates, __global algorithmFPType * candidateDistances, __global const int * candidates_tmp,
                                   __global const algorithmFPType * candidateDistances_tmp, int K) {
        const int global_id = get_global_id(0);
        const int local_id  = get_sub_group_local_id();

        __local int curInd[NUM_PARTS_CND];
        if (global_id == 0)
        {
            curInd[local_id] = 0;
            for (int i = 0; i < K; i++)
            {
                algorithmFPType curVal = local_id < NUM_PARTS_CND ? -candidateDistances_tmp[local_id * K + curInd[local_id]] : 1.0;
                algorithmFPType maxVal = -sub_group_reduce_min(curVal);
                if (maxVal < 0)
                {
                    if (local_id == 0) candidates[i] = -1;
                    curInd[local_id]++;
                    continue;
                }
                int counterInd = -sub_group_reduce_min(-curVal < maxVal ? 1 : -local_id);
                if (counterInd > NUM_PARTS_CND - 1)
                {
                    if (local_id == 0) candidates[i] = -1;
                    curInd[local_id]++;
                    continue;
                }
                if (local_id == counterInd)
                {
                    candidates[i]         = candidates_tmp[curInd[counterInd] + counterInd * K];
                    candidateDistances[i] = candidateDistances_tmp[curInd[counterInd] + counterInd * K];
                    curInd[local_id]++;
                }
            }
        }
    }

    __kernel void partial_reduce_centroids(__global const algorithmFPType * data, __global const algorithmFPType * distances,
                                           __global const int * assignments, __global algorithmFPType * partialCentroids,
                                           __global int * partialCentroidsCounters, int N, int K, int P, int doReset) {
        const int local_id = get_global_id(0) % P;

        const int global_id   = get_global_id(0) / P;
        const int global_size = get_global_size(0) / P;

        if (doReset)
        {
            for (int i = 0; i < K; i++)
            {
                partialCentroids[global_id * K * P + i * P + local_id] = 0.0;
            }

            if (local_id == 0)
            {
                for (int i = 0; i < K; i++)
                {
                    partialCentroidsCounters[global_id * K + i] = 0;
                }
            }
        }

        for (int i = global_id; i < N; i += global_size)
        {
            int cl = assignments[i];
            if (local_id == 0)
            {
                partialCentroidsCounters[global_id * K + cl]++;
            }
            partialCentroids[global_id * K * P + cl * P + local_id] += data[i * P + local_id];
        }
    }

    __kernel void merge_reduce_centroids(__global algorithmFPType * partialCentroids, __global int * partialCentroidsCounters,
                                         __global algorithmFPType * centroids, int K, int P, int parts) {
        const int local_id   = get_local_id(0);
        const int local_size = get_local_size(0);

        const int cl_id = get_group_id(0);

        for (int i = local_id + local_size; i < parts; i += local_size)
        {
            for (int j = 0; j < P; j++)
            {
                partialCentroids[local_id * K * P + cl_id * P + j] += partialCentroids[i * K * P + cl_id * P + j];
            }
            partialCentroidsCounters[local_id * K + cl_id] += partialCentroidsCounters[i * K + cl_id];
        }

        for (int len = local_size / 2; len > 0; len >>= 1)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            if (local_id < len)
            {
                for (int j = 0; j < P; j++)
                {
                    partialCentroids[local_id * K * P + cl_id * P + j] += partialCentroids[(local_id + len) * K * P + cl_id * P + j];
                }
                partialCentroidsCounters[local_id * K + cl_id] += partialCentroidsCounters[(local_id + len) * K + cl_id];
            }
        }

        if (local_id == 0 && partialCentroidsCounters[cl_id] != 0)
        {
            for (int j = 0; j < P; j++)
            {
                centroids[cl_id * P + j] = partialCentroids[cl_id * P + j] / partialCentroidsCounters[cl_id];
            }
        }
    }

    __kernel void count_empty_clusters(__global const int * partialCentroidsCounters, int K, int nPartialCentroids, __global int * numEmptyClusters) {
        const int local_id   = get_local_id(1);
        const int local_size = get_local_size(1);

        int numEmpty = 0;
        for (int i = local_id; i < K; i += local_size)
        {
            int count = 0;
            for (int j = 0; j < nPartialCentroids; j++)
            {
                count += partialCentroidsCounters[j * K + i];
            }
            numEmpty += count > 0 ? 0 : 1;
        }
        numEmpty = sub_group_reduce_add(numEmpty);
        if (local_id == 0) numEmptyClusters[0] = numEmpty;
    }

    __kernel void update_objective_function(__global const algorithmFPType * dataSq, __global const algorithmFPType * distances, int N, int K,
                                            __global algorithmFPType * objFunction) {
        const int local_id   = get_local_id(0);
        const int local_size = get_local_size(0);

        __local algorithmFPType local_sum[LOCAL_SUM_SIZE];

        local_sum[local_id] = 0.0f;

        for (int i = local_id; i < N; i += local_size)
        {
            local_sum[local_id] += dataSq[i] + 2.0 * distances[i];
        }

        __sum_reduce(local_sum, local_id, local_size);

        if (local_id == 0)
        {
            objFunction[0] += local_sum[0];
        }
    }

);

#endif
