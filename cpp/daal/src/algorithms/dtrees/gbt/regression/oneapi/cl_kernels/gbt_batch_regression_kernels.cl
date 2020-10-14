/* file: gbt_kernels.cl */
/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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
//  Implementation of GBT Batch Regression OpenCL kernels.
//--
*/

#ifndef __GBT_BATCH_REGRESSION_KERNELS_CL__
#define __GBT_BATCH_REGRESSION_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE(name, src) static const char * name = #src;

DECLARE_SOURCE(
    gbt_batch_regression_kernels,

    __kernel void scan(const __global algorithmFPType * values, __global algorithmFPType * partialSums, uint32_t nRows) {
        const uint32_t n_groups             = get_num_groups(0);
        const uint32_t n_sub_groups         = get_num_sub_groups();
        const uint32_t n_total_sub_groups   = n_sub_groups * n_groups;
        const uint32_t nElementsForSubgroup = nRows / n_total_sub_groups + !!(nRows % n_total_sub_groups);
        const uint32_t local_size           = get_sub_group_size();

        const uint32_t id           = get_local_id(0);
        const uint32_t local_id     = get_sub_group_local_id();
        const uint32_t sub_group_id = get_sub_group_id();
        const uint32_t group_id     = get_group_id(0) * n_sub_groups + sub_group_id;

        uint32_t iStart = group_id * nElementsForSubgroup;
        uint32_t iEnd   = (group_id + 1) * nElementsForSubgroup;

        if (iEnd > nRows)
        {
            iEnd = nRows;
        }

        algorithmFPType sum = 0.0;

        for (uint32_t i = iStart + local_id; i < iEnd; i += local_size)
        {
            algorithmFPType partial_sum = sub_group_reduce_add(values[i]);
            sum += partial_sum;
        }

        if (local_id == 0)
        {
            partialSums[group_id] = sum;
        }
    }

    __kernel void reduce(const __global algorithmFPType * partialSums, __global algorithmFPType * totalSum, uint32_t nSubgroupSums) {
        if (get_sub_group_id() > 0) return;

        const uint32_t local_size = get_sub_group_size();
        const uint32_t local_id   = get_sub_group_local_id();

        algorithmFPType sum = 0.0;

        for (uint32_t i = local_id; i < nSubgroupSums; i += local_size)
        {
            algorithmFPType partial_sum = sub_group_reduce_add(partialSums[i]);
            sum += partial_sum;
        }

        if (local_id == 0)
        {
            totalSum[0] = sum;
        }
    }

    __kernel void computeOptCoeffs(const __global algorithmFPType * labels, const __global algorithmFPType * response,
                                   __global algorithmFPType * optCoeffs) {
        const uint32_t id     = get_global_id(0);
        optCoeffs[2 * id + 0] = response[id] - labels[id];
        optCoeffs[2 * id + 1] = 1;
    }

    __kernel void initializeTreeOrder(__global int * treeOrder) {
        const uint32_t id = get_global_id(0);
        treeOrder[id]     = id;
    }

    __kernel void computePartialHistograms(const __global int * data, const __global int * treeOrder, const __global algorithmFPType * optCoeffs,
                                           __global algorithmFPType * partialHistograms, uint32_t offset, uint32_t nRows,
                                           const __global int * binOffsets, uint32_t nTotalBins, uint32_t nFeatures) {
        const uint32_t feat_id           = get_local_id(1);
        const uint32_t global_id         = get_global_id(0);
        const uint32_t global_size       = get_global_size(0);
        const uint32_t nElementsForGroup = nRows / global_size + !!(nRows % global_size);

        uint32_t iStart = global_id * nElementsForGroup;
        uint32_t iEnd   = (global_id + 1) * nElementsForGroup;

        if (iEnd > nRows)
        {
            iEnd = nRows;
        }

        __global algorithmFPType * histogram = partialHistograms + nTotalBins * global_id * 2 + binOffsets[feat_id] * 2;

        uint32_t nBins = binOffsets[feat_id + 1] - binOffsets[feat_id];

        for (uint32_t i = 0; i < 2 * nBins; i++)
        {
            histogram[i] = 0.0;
        }

        for (uint32_t i = iStart; i < iEnd; i++)
        {
            uint32_t id  = treeOrder[offset + i];
            uint32_t bin = data[id * nFeatures + feat_id];
            histogram[bin * 2 + 0] += optCoeffs[id * 2 + 0];
            histogram[bin * 2 + 1] += optCoeffs[id * 2 + 1];
        }
    }

    __kernel void reducePartialHistograms(const __global algorithmFPType * partialHistograms, __global algorithmFPType * histogram,
                                          uint32_t nHistograms, uint32_t nTotalBins) {
        __local algorithmFPType buf[256 * 2];

        const uint32_t bin_id     = get_global_id(0);
        const uint32_t local_id   = get_local_id(1);
        const uint32_t local_size = get_local_size(1);

        buf[local_id * 2 + 0] = 0;
        buf[local_id * 2 + 1] = 0;

        for (uint32_t i = local_id; i < nHistograms; i += local_size)
        {
            buf[local_id * 2 + 0] += partialHistograms[i * nTotalBins * 2 + bin_id * 2 + 0];
            buf[local_id * 2 + 1] += partialHistograms[i * nTotalBins * 2 + bin_id * 2 + 1];
        }

        for (uint32_t offset = local_size / 2; offset > 0; offset >>= 1)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            if (local_id < offset)
            {
                buf[local_id * 2 + 0] += buf[(local_id + offset) * 2 + 0];
                buf[local_id * 2 + 1] += buf[(local_id + offset) * 2 + 1];
            }
        }

        if (local_id == 0)
        {
            histogram[bin_id * 2 + 0] = buf[0];
            histogram[bin_id * 2 + 1] = buf[1];
        }
    }

    __kernel void computeHistogramDiff(const __global algorithmFPType * histogramSrc, const __global algorithmFPType * histogramTotal,
                                       __global algorithmFPType * histogramDst) {
        const uint32_t id        = get_global_id(0);
        histogramDst[id * 2 + 0] = histogramTotal[id * 2 + 0] - histogramSrc[id * 2 + 0];
        histogramDst[id * 2 + 1] = histogramTotal[id * 2 + 1] - histogramSrc[id * 2 + 1];
    }

    __kernel void computeTotalOptCoeffs(const __global algorithmFPType * histogram, __global algorithmFPType * totalOptCoeffs,
                                        const __global int * binOffsets, uint32_t nTotalBins) {
        if (get_sub_group_id() > 0) return;

        const uint32_t feat_id    = get_global_id(1);
        const uint32_t local_id   = get_sub_group_local_id();
        const uint32_t local_size = get_sub_group_size();

        algorithmFPType g = 0.0;
        algorithmFPType h = 0.0;

        uint32_t nBins = binOffsets[feat_id + 1] - binOffsets[feat_id];

        const __global algorithmFPType * histogramForFeature = histogram + binOffsets[feat_id] * 2;

        for (uint32_t i = local_id; i < nBins; i += local_size)
        {
            g += sub_group_reduce_add(histogramForFeature[i * 2 + 0]);
            h += sub_group_reduce_add(histogramForFeature[i * 2 + 1]);
        }

        if (feat_id == 0 && local_id == 0)
        {
            totalOptCoeffs[0] = g;
            totalOptCoeffs[1] = h;
        }
    }

    algorithmFPType impurityValue(algorithmFPType g, algorithmFPType h, algorithmFPType lambda) { return (g / (h + lambda)) * g; }

    __kernel void computeBestSplitForFeatures(const __global algorithmFPType * histogram, const __global algorithmFPType * totalOptCoeffs,
                                              __global algorithmFPType * splitInfo, __global int * splitValue, const __global int * binOffsets,
                                              uint32_t nTotalBins, algorithmFPType lambda) {
        if (get_sub_group_id() > 0) return;

        const uint32_t feat_id = get_global_id(1);

        const uint32_t local_id   = get_sub_group_local_id();
        const uint32_t local_size = get_sub_group_size();

        int curFeatureValue       = -1;
        algorithmFPType curImpDec = -1e30;
        algorithmFPType curGLeft  = 0.0;
        algorithmFPType curHLeft  = 0.0;

        algorithmFPType g = 0.0;
        algorithmFPType h = 0.0;

        const __global algorithmFPType * histogramForFeature = histogram + binOffsets[feat_id] * 2;
        __global algorithmFPType * splitInfoForFeature       = splitInfo + feat_id * 5;
        __global int * splitValueForFeature                  = splitValue + feat_id;
        uint32_t nBins                                       = binOffsets[feat_id + 1] - binOffsets[feat_id];

        for (uint32_t i = local_id; i < nBins; i += local_size)
        {
            algorithmFPType gLeft  = g + sub_group_scan_inclusive_add(histogramForFeature[i * 2 + 0]);
            algorithmFPType hLeft  = h + sub_group_scan_inclusive_add(histogramForFeature[i * 2 + 1]);
            algorithmFPType gRight = totalOptCoeffs[0] - gLeft;
            algorithmFPType hRight = totalOptCoeffs[1] - hLeft;

            algorithmFPType impDec = impurityValue(gLeft, hLeft, lambda) + impurityValue(gRight, hRight, lambda);

            if (curFeatureValue == -1 || impDec > curImpDec)
            {
                curFeatureValue = i;
                curImpDec       = impDec;
                curGLeft        = gLeft;
                curHLeft        = hLeft;
            }

            g += sub_group_reduce_add(histogramForFeature[i * 2 + 0]);
            h += sub_group_reduce_add(histogramForFeature[i * 2 + 1]);
        }

        algorithmFPType bestImpDec = sub_group_reduce_max(curImpDec);
        int bestFeatureValue       = sub_group_reduce_min(bestImpDec == curImpDec ? curFeatureValue : nBins);

        if (curFeatureValue == bestFeatureValue)
        {
            splitValueForFeature[0] = curFeatureValue == nBins ? -1 : curFeatureValue;
            splitInfoForFeature[0]  = curImpDec;
            splitInfoForFeature[1]  = curGLeft;
            splitInfoForFeature[2]  = curHLeft;
            splitInfoForFeature[3]  = totalOptCoeffs[0] - curGLeft;
            splitInfoForFeature[4]  = totalOptCoeffs[1] - curHLeft;
        }
    }

    __kernel void partitionScan(const __global int * data, const __global int * treeOrder, __global int * partialSums, int splitValue,
                                uint32_t offset, uint32_t nRows) {
        const uint32_t n_groups             = get_num_groups(0);
        const uint32_t n_sub_groups         = get_num_sub_groups();
        const uint32_t n_total_sub_groups   = n_sub_groups * n_groups;
        const uint32_t nElementsForSubgroup = nRows / n_total_sub_groups + !!(nRows % n_total_sub_groups);
        const uint32_t local_size           = get_sub_group_size();

        const uint32_t id           = get_local_id(0);
        const uint32_t local_id     = get_sub_group_local_id();
        const uint32_t sub_group_id = get_sub_group_id();
        const uint32_t group_id     = get_group_id(0) * n_sub_groups + sub_group_id;

        uint32_t iStart = group_id * nElementsForSubgroup;
        uint32_t iEnd   = (group_id + 1) * nElementsForSubgroup;

        if (iEnd > nRows)
        {
            iEnd = nRows;
        }

        uint32_t sum = 0;

        for (uint32_t i = iStart + local_id; i < iEnd; i += local_size)
        {
            uint32_t value = (uint32_t)(data[treeOrder[offset + i]] > splitValue);
            sum += sub_group_reduce_add(value);
        }

        if (local_id == 0)
        {
            partialSums[group_id] = sum;
        }
    }

    __kernel void partitionSumScan(const __global int * partialSums, __global int * partialPrefixSums, __global int * totalSum,
                                   uint32_t nSubgroupSums) {
        if (get_sub_group_id() > 0) return;

        const uint32_t local_size = get_sub_group_size();
        const uint32_t local_id   = get_sub_group_local_id();

        uint32_t sum = 0;

        for (uint32_t i = local_id; i < nSubgroupSums; i += local_size)
        {
            uint32_t value       = partialSums[i];
            uint32_t boundary    = sub_group_scan_exclusive_add(value);
            partialPrefixSums[i] = sum + boundary;
            sum += sub_group_reduce_add(value);
        }

        if (local_id == 0)
        {
            totalSum[0]                      = sum;
            partialPrefixSums[nSubgroupSums] = sum;
        }
    }

    __kernel void partitionReorder(const __global int * data, const __global int * treeOrder, __global int * treeOrderBuf,
                                   const __global int * partialPrefixSums, int splitValue, uint32_t offset, uint32_t nRows) {
        const uint32_t n_groups             = get_num_groups(0);
        const uint32_t n_sub_groups         = get_num_sub_groups();
        const uint32_t n_total_sub_groups   = n_sub_groups * n_groups;
        const uint32_t nElementsForSubgroup = nRows / n_total_sub_groups + !!(nRows % n_total_sub_groups);
        const uint32_t local_size           = get_sub_group_size();

        const uint32_t id           = get_local_id(0);
        const uint32_t local_id     = get_sub_group_local_id();
        const uint32_t sub_group_id = get_sub_group_id();
        const uint32_t group_id     = get_group_id(0) * n_sub_groups + sub_group_id;

        uint32_t iStart = group_id * nElementsForSubgroup;
        uint32_t iEnd   = (group_id + 1) * nElementsForSubgroup;

        if (iEnd > nRows)
        {
            iEnd = nRows;
        }

        uint32_t groupOffset = partialPrefixSums[group_id];
        uint32_t totalOffset = nRows - partialPrefixSums[n_total_sub_groups];
        uint32_t sum         = 0;

        for (uint32_t i = iStart + local_id; i < iEnd; i += local_size)
        {
            uint32_t id                    = treeOrder[offset + i];
            uint32_t part                  = (uint32_t)(data[id] > splitValue);
            uint32_t boundary              = groupOffset + sum + sub_group_scan_exclusive_add(part);
            uint32_t pos_new               = (part ? totalOffset + boundary : i - boundary);
            treeOrderBuf[offset + pos_new] = id;
            sum += sub_group_reduce_add(part);
        }
    }

    __kernel void partitionCopy(const __global uint32_t * treeOrderBuf, __global int * treeOrder, uint32_t offset) {
        const uint32_t id      = get_global_id(0);
        treeOrder[offset + id] = treeOrderBuf[offset + id];
    }

    __kernel void updateResponse(const __global int * treeOrder, __global algorithmFPType * response, uint32_t iStart, uint32_t nRows,
                                 algorithmFPType inc) {
        const uint32_t id = get_global_id(0);
        response[treeOrder[id + iStart]] += inc;
    }

);

#endif
