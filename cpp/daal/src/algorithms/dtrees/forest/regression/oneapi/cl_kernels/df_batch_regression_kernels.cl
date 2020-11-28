/* file: df_batch_regression_kernels.cl */
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
//  Implementation of decision forest Batch Regression OpenCL kernels.
//--
*/

#ifndef __DF_BATCH_REGRESSION_KERNELS_CL__
#define __DF_BATCH_REGRESSION_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE(name, src) static const char * name = #src;

DECLARE_SOURCE(
    df_batch_regression_kernels_part1,

    inline int fpEq(algorithmFPType a, algorithmFPType b) { return (int)(fabs(a - b) <= algorithmFPTypeAccuracy); }

    inline int fpGt(algorithmFPType a, algorithmFPType b) { return (int)((a - b) > algorithmFPTypeAccuracy); }

    // merge input arrs with stat values
    void mergeStatArr(algorithmFPType * n, algorithmFPType * mean, algorithmFPType * sum2Cent, algorithmFPType * mrgN, algorithmFPType * mrgMean,
                      algorithmFPType * mrgSum2Cent) {
        algorithmFPType sumN1N2    = n[0] + n[1];
        algorithmFPType mulN1N2    = n[0] * n[1];
        algorithmFPType deltaScale = mulN1N2 / sumN1N2;
        algorithmFPType meanScale  = (algorithmFPType)1 / sumN1N2;
        algorithmFPType delta      = mean[1] - mean[0];

        *mrgSum2Cent = sum2Cent[0] + sum2Cent[1] + delta * delta * deltaScale;
        *mrgMean     = (mean[0] * n[0] + mean[1] * n[1]) * meanScale;
        *mrgN        = sumN1N2;
    }

    // merge single value to stat
    void mergeValToStat(algorithmFPType val, algorithmFPType * mrgN, algorithmFPType * mrgMean, algorithmFPType * mrgSum2Cent) {
        *mrgN += (algorithmFPType)1;
        algorithmFPType invN  = ((algorithmFPType)1) / *mrgN;
        algorithmFPType delta = val - *mrgMean;
        *mrgMean += delta * invN;
        *mrgSum2Cent += delta * (val - *mrgMean);
    }

    __kernel void computeBestSplitSinglePass(const __global int * data, const __global int * treeOrder, const __global int * selectedFeatures,
                                             int nSelectedFeatures, const __global algorithmFPType * response, const __global int * binOffsets,
                                             __global int * nodeList, const __global int * nodeIndices, int nodeIndicesOffset,
                                             __global algorithmFPType * splitInfo, __global algorithmFPType * nodeImpDecreaseList,
                                             int updateImpDecreaseRequired, int nFeatures, int minObservationsInLeafNode,
                                             algorithmFPType impurityThreshold) {
        // this kernel is targeted for processing nodes with small number of rows
        // nodeList will be updated with split attributes
        // spliInfo will contain node impurity and mean
        const int nNodeProp = NODE_PROPS; // num of node properties in nodeList
        const int nImpProp  = IMPURITY_PROPS;
        const int leafMark  = -1;

        const int local_id           = get_local_id(0);
        const int sub_group_local_id = get_sub_group_local_id();
        const int sub_group_size     = get_sub_group_size();
        const int local_size         = get_local_size(0);
        const int n_sub_groups       = local_size / sub_group_size; // num of subgroups for current node processing
        const int sub_group_id       = local_id / sub_group_size;
        const int max_sub_groups_num = 16; //replace with define

        const int bufIdx  = get_global_id(1) % (max_sub_groups_num / n_sub_groups); // local buffer is shared between 16 sub groups
        const int nodeIdx = get_global_id(1);
        const int nodeId  = nodeIndices[nodeIndicesOffset + nodeIdx];

        const int rowsOffset = nodeList[nodeId * nNodeProp + 0];
        const int nRows      = nodeList[nodeId * nNodeProp + 1];

        // each sub group will process 16 bins and produce 1 best split for it
        // expect maximum 16 subgroups only for each node
        const int maxBinsBlocks = 16;
        __local algorithmFPType bufI[maxBinsBlocks]; // storage for impurity decrease
        __local int bufS[maxBinsBlocks * nNodeProp]; // storage for split info

        const algorithmFPType minImpDec = (algorithmFPType)-1e30;
        algorithmFPType curImpDec       = minImpDec;
        int valNotFound                 = 1 << 30;
        int curFeatureValue             = leafMark;
        int curFeatureId                = leafMark;

        nodeList[nodeId * nNodeProp + 2] = curFeatureId;
        nodeList[nodeId * nNodeProp + 3] = curFeatureValue;
        nodeList[nodeId * nNodeProp + 4] = nRows;

        algorithmFPType mrgN        = (algorithmFPType)0;
        algorithmFPType mrgMean     = (algorithmFPType)0;
        algorithmFPType mrgSum2Cent = (algorithmFPType)0;

        algorithmFPType imp = (algorithmFPType)0; // node impurity

        algorithmFPType bestLN = (algorithmFPType)0;

        int totalBins = 0;

        // totalBins is calculated by each subgroup
        for (int featIdx = sub_group_local_id; featIdx < nSelectedFeatures; featIdx += sub_group_size)
        {
            int featId = selectedFeatures[nodeId * nSelectedFeatures + featIdx];
            int nBins  = binOffsets[featId + 1] - binOffsets[featId];
            totalBins += sub_group_reduce_add(nBins);
        }
        totalBins = sub_group_broadcast(totalBins, 0);

        int currFtrIdx  = 0;
        int featId      = selectedFeatures[nodeId * nSelectedFeatures + currFtrIdx];
        int binId       = 0;
        int currFtrBins = binOffsets[featId + 1] - binOffsets[featId];
        int passedBins  = 0;

        for (int i = local_id; i < totalBins; i += local_size)
        {
            while (i >= passedBins + currFtrBins)
            {
                passedBins += currFtrBins;
                currFtrIdx++;
                featId      = selectedFeatures[nodeId * nSelectedFeatures + currFtrIdx];
                currFtrBins = binOffsets[featId + 1] - binOffsets[featId];
            }
            binId = i - passedBins;

            mrgN        = (algorithmFPType)0;
            mrgMean     = (algorithmFPType)0;
            mrgSum2Cent = (algorithmFPType)0;

            algorithmFPType mrgLRN[2]        = { (algorithmFPType)0 };
            algorithmFPType mrgLRMean[2]     = { (algorithmFPType)0 };
            algorithmFPType mrgLRSum2Cent[2] = { (algorithmFPType)0 };

            for (int row = 0; row < nRows; row++)
            {
                int id  = treeOrder[rowsOffset + row];
                int bin = data[id * nFeatures + featId];

                mergeValToStat(response[id], &mrgLRN[(int)(bin > binId)], &mrgLRMean[(int)(bin > binId)], &mrgLRSum2Cent[(int)(bin > binId)]);
            }

            mergeStatArr(mrgLRN, mrgLRMean, mrgLRSum2Cent, &mrgN, &mrgMean, &mrgSum2Cent);

            algorithmFPType impDec = mrgSum2Cent - (mrgLRSum2Cent[0] + mrgLRSum2Cent[1]);
            imp                    = mrgSum2Cent / mrgN; // mrgN isn't 0 due to it is num of rows in node

            if ((algorithmFPType)0 < impDec && (!fpEq(imp, (algorithmFPType)0)) && imp >= impurityThreshold
                && (curFeatureValue == leafMark || fpGt(impDec, curImpDec) || (fpEq(impDec, curImpDec) && featId < curFeatureId))
                && mrgLRN[0] >= minObservationsInLeafNode && mrgLRN[1] >= minObservationsInLeafNode)
            {
                curFeatureId    = featId;
                curFeatureValue = binId;
                curImpDec       = impDec;

                bestLN = mrgLRN[0];
            }
        } // for i

        algorithmFPType bestImpDec = sub_group_reduce_max(curImpDec);

        int impDecIsBest     = fpEq(bestImpDec, curImpDec);
        int bestFeatureId    = sub_group_reduce_min(impDecIsBest ? curFeatureId : valNotFound);
        int bestFeatureValue = sub_group_reduce_min((bestFeatureId == curFeatureId && impDecIsBest) ? curFeatureValue : valNotFound);

        bool noneSplitFoundBySubGroup = ((leafMark == bestFeatureId) && (0 == sub_group_local_id));
        bool mySplitIsBest            = (leafMark != bestFeatureId && curFeatureId == bestFeatureId && curFeatureValue == bestFeatureValue);
        if (noneSplitFoundBySubGroup || mySplitIsBest)
        {
            if (1 == n_sub_groups)
            {
                __global algorithmFPType * splitNodeInfo = splitInfo + nodeId * nImpProp;
                nodeList[nodeId * nNodeProp + 2]         = curFeatureId == valNotFound ? leafMark : curFeatureId;
                nodeList[nodeId * nNodeProp + 3]         = curFeatureValue == valNotFound ? leafMark : curFeatureValue;
                nodeList[nodeId * nNodeProp + 4]         = (int)bestLN;

                splitNodeInfo[0] = imp;
                splitNodeInfo[1] = mrgMean;

                if (updateImpDecreaseRequired) nodeImpDecreaseList[nodeId] = curImpDec / mrgN;
            }
            else
            {
                bufS[(bufIdx + sub_group_id) * nNodeProp + 0] = curFeatureId;
                bufS[(bufIdx + sub_group_id) * nNodeProp + 1] = curFeatureValue;
                bufS[(bufIdx + sub_group_id) * nNodeProp + 2] = (int)bestLN;

                bufI[bufIdx + sub_group_id] = curImpDec;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        if (1 < n_sub_groups && 0 == sub_group_id)
        {
            // first sub group for current node reduces over local buffer if required
            algorithmFPType curImpDec = (sub_group_local_id < n_sub_groups) ? bufI[bufIdx + sub_group_local_id] : minImpDec;

            int curFeatureId    = sub_group_local_id < n_sub_groups ? bufS[(bufIdx + sub_group_local_id) * nNodeProp + 0] : valNotFound;
            int curFeatureValue = sub_group_local_id < n_sub_groups ? bufS[(bufIdx + sub_group_local_id) * nNodeProp + 1] : valNotFound;
            int LN              = sub_group_local_id < n_sub_groups ? bufS[(bufIdx + sub_group_local_id) * nNodeProp + 2] : 0;

            algorithmFPType bestImpDec = sub_group_reduce_max(curImpDec);

            int impDecIsBest     = fpEq(bestImpDec, curImpDec);
            int bestFeatureId    = sub_group_reduce_min(impDecIsBest ? curFeatureId : valNotFound);
            int bestFeatureValue = sub_group_reduce_min((bestFeatureId == curFeatureId && impDecIsBest) ? curFeatureValue : valNotFound);

            bool noneSplitFoundBySubGroup = ((leafMark == bestFeatureId) && (0 == sub_group_local_id));
            bool mySplitIsBest            = (leafMark != bestFeatureId && curFeatureId == bestFeatureId && curFeatureValue == bestFeatureValue);
            if (noneSplitFoundBySubGroup || mySplitIsBest)
            {
                __global algorithmFPType * splitNodeInfo = splitInfo + nodeId * nImpProp;
                nodeList[nodeId * nNodeProp + 2]         = curFeatureId == valNotFound ? leafMark : curFeatureId;
                nodeList[nodeId * nNodeProp + 3]         = curFeatureValue == valNotFound ? leafMark : curFeatureValue;
                nodeList[nodeId * nNodeProp + 4]         = (int)LN;

                splitNodeInfo[0] = imp;
                splitNodeInfo[1] = mrgMean;

                if (updateImpDecreaseRequired) nodeImpDecreaseList[nodeId] = curImpDec / mrgN;
            }
        }
    });

DECLARE_SOURCE(
    df_batch_regression_kernels_part2,

    inline int fpEq(algorithmFPType a, algorithmFPType b) { return (int)(fabs(a - b) <= algorithmFPTypeAccuracy); }

    inline int fpGt(algorithmFPType a, algorithmFPType b) { return (int)((a - b) > algorithmFPTypeAccuracy); }

    // merge input arrs with stat values
    void mergeStatArr(algorithmFPType * n, algorithmFPType * mean, algorithmFPType * sum2Cent, algorithmFPType * mrgN, algorithmFPType * mrgMean,
                      algorithmFPType * mrgSum2Cent) {
        algorithmFPType sumN1N2    = n[0] + n[1];
        algorithmFPType mulN1N2    = n[0] * n[1];
        algorithmFPType deltaScale = mulN1N2 / sumN1N2;
        algorithmFPType meanScale  = (algorithmFPType)1 / sumN1N2;
        algorithmFPType delta      = mean[1] - mean[0];

        *mrgSum2Cent = sum2Cent[0] + sum2Cent[1] + delta * delta * deltaScale;
        *mrgMean     = (mean[0] * n[0] + mean[1] * n[1]) * meanScale;
        *mrgN        = sumN1N2;
    }

    // merge to stats in one
    void mergeStat(algorithmFPType n, algorithmFPType mean, algorithmFPType sum2Cent, algorithmFPType * mrgLN, algorithmFPType * mrgLMean,
                   algorithmFPType * mrgLSum2Cent) {
        algorithmFPType sumN1N2    = *mrgLN + n;
        algorithmFPType mulN1N2    = *mrgLN * n;
        algorithmFPType deltaScale = mulN1N2 / sumN1N2;
        algorithmFPType meanScale  = (algorithmFPType)1 / sumN1N2;
        algorithmFPType delta      = mean - *mrgLMean;

        *mrgLSum2Cent = *mrgLSum2Cent + sum2Cent + delta * delta * deltaScale;
        *mrgLMean     = (*mrgLMean * *mrgLN + mean * n) * meanScale;
        *mrgLN        = sumN1N2;
    }

    __kernel void computeBestSplitByHistogram(const __global algorithmFPType * histograms, const __global int * selectedFeatures,
                                              int nSelectedFeatures, const __global int * binOffsets, __global int * nodeList,
                                              const __global int * nodeIndices, int nodeIndicesOffset, __global algorithmFPType * splitInfo,
                                              __global algorithmFPType * nodeImpDecreaseList, int updateImpDecreaseRequired, int nMaxBinsAmongFtrs,
                                              int minObservationsInLeafNode, algorithmFPType impurityThreshold) {
        // this kernel has almost the same code as computeBestSplitSinglePass
        // the difference is that here for each potential split point we pass through bins hist instead of rows
        // nodeList will be updated with split attributes in this kernel
        // spliInfo will contain node impurity and mean
        const int nProp              = HIST_PROPS; // num of characteristics in histogram
        const int nNodeProp          = NODE_PROPS; // num of node properties in nodeList
        const int nImpProp           = IMPURITY_PROPS;
        const int local_id           = get_local_id(0);
        const int sub_group_local_id = get_sub_group_local_id();
        const int sub_group_size     = get_sub_group_size();
        const int nodeIdx            = get_global_id(1);
        const int nodeId             = nodeIndices[nodeIndicesOffset + nodeIdx];
        const int leafMark           = -1;

        const int local_size         = get_local_size(0);
        const int n_sub_groups       = local_size / sub_group_size; // num of subgroups for current node processing
        const int sub_group_id       = local_id / sub_group_size;
        const int max_sub_groups_num = 16;                                                     //replace with define
        const int bufIdx             = get_global_id(1) % (max_sub_groups_num / n_sub_groups); // local buffer is shared between 16 sub groups

        const int rowsOffset = nodeList[nodeId * nNodeProp + 0];
        const int nRows      = nodeList[nodeId * nNodeProp + 1];

        // each sub group will process 16 bins and produce 1 best split for it
        // expect maximum 16 subgroups only for each node
        const int maxBinsBlocks = 16;
        __local algorithmFPType bufI[maxBinsBlocks]; // storage for impurity decrease
        __local int bufS[maxBinsBlocks * nNodeProp]; // storage for split info

        const algorithmFPType minImpDec = (algorithmFPType)-1e30;
        int valNotFound                 = 1 << 30;
        int curFeatureValue             = leafMark;
        int curFeatureId                = leafMark;
        algorithmFPType curImpDec       = minImpDec;

        nodeList[nodeId * nNodeProp + 2] = curFeatureId;
        nodeList[nodeId * nNodeProp + 3] = curFeatureValue;
        nodeList[nodeId * nNodeProp + 4] = nRows;

        algorithmFPType mrgN        = (algorithmFPType)0;
        algorithmFPType mrgMean     = (algorithmFPType)0;
        algorithmFPType mrgSum2Cent = (algorithmFPType)0;

        algorithmFPType imp = (algorithmFPType)0; // node impurity

        algorithmFPType bestLN = (algorithmFPType)0;

        int totalBins = 0;

        // totalBins is calculated by each subgroup
        for (int featIdx = sub_group_local_id; featIdx < nSelectedFeatures; featIdx += sub_group_size)
        {
            int featId = selectedFeatures[nodeId * nSelectedFeatures + featIdx];
            int nBins  = binOffsets[featId + 1] - binOffsets[featId];
            totalBins += sub_group_reduce_add(nBins);
        }

        totalBins = sub_group_broadcast(totalBins, 0);

        int currFtrIdx  = 0;
        int featId      = selectedFeatures[nodeId * nSelectedFeatures + currFtrIdx];
        int binId       = 0;
        int currFtrBins = binOffsets[featId + 1] - binOffsets[featId];
        int passedBins  = 0;

        for (int i = local_id; i < totalBins; i += local_size)
        {
            while (i >= passedBins + currFtrBins)
            {
                passedBins += currFtrBins;
                currFtrIdx++;
                featId      = selectedFeatures[nodeId * nSelectedFeatures + currFtrIdx];
                currFtrBins = binOffsets[featId + 1] - binOffsets[featId];
            }
            binId = i - passedBins;

            const __global algorithmFPType * nodeHistogram       = histograms + nodeIdx * nSelectedFeatures * nMaxBinsAmongFtrs * nProp;
            const __global algorithmFPType * histogramForFeature = nodeHistogram + currFtrIdx * nMaxBinsAmongFtrs * nProp;

            // calculate merged statistics
            mrgN        = (algorithmFPType)0;
            mrgMean     = (algorithmFPType)0;
            mrgSum2Cent = (algorithmFPType)0;

            algorithmFPType mrgLRN[2]        = { (algorithmFPType)0 };
            algorithmFPType mrgLRMean[2]     = { (algorithmFPType)0 };
            algorithmFPType mrgLRSum2Cent[2] = { (algorithmFPType)0 };

            for (int tbin = 0; tbin < currFtrBins; tbin++)
            {
                int binOffset     = tbin * nProp;
                algorithmFPType n = histogramForFeature[binOffset + 0];

                algorithmFPType mean     = histogramForFeature[binOffset + 1];
                algorithmFPType sum2Cent = histogramForFeature[binOffset + 2];
                if ((algorithmFPType)0 == n) continue;

                mergeStat(n, mean, sum2Cent, &mrgLRN[(int)(tbin > binId)], &mrgLRMean[(int)(tbin > binId)], &mrgLRSum2Cent[(int)(tbin > binId)]);
            }

            mergeStatArr(mrgLRN, mrgLRMean, mrgLRSum2Cent, &mrgN, &mrgMean, &mrgSum2Cent);

            algorithmFPType impDec = mrgSum2Cent - (mrgLRSum2Cent[0] + mrgLRSum2Cent[1]);
            imp                    = mrgSum2Cent / mrgN; // mrgN isn't 0 due to it is num of rows in node

            if ((algorithmFPType)0 < impDec && (!fpEq(imp, (algorithmFPType)0)) && imp >= impurityThreshold
                && (curFeatureValue == leafMark || fpGt(impDec, curImpDec) || (fpEq(impDec, curImpDec) && featId < curFeatureId))
                && mrgLRN[0] >= minObservationsInLeafNode && mrgLRN[1] >= minObservationsInLeafNode)
            {
                curFeatureId    = featId;
                curFeatureValue = binId;
                curImpDec       = impDec;

                bestLN = mrgLRN[0];
            }
        } // for i

        algorithmFPType bestImpDec = sub_group_reduce_max(curImpDec);

        int impDecIsBest     = fpEq(bestImpDec, curImpDec);
        int bestFeatureId    = sub_group_reduce_min(impDecIsBest ? curFeatureId : valNotFound);
        int bestFeatureValue = sub_group_reduce_min((bestFeatureId == curFeatureId && impDecIsBest) ? curFeatureValue : valNotFound);

        bool noneSplitFoundBySubGroup = ((leafMark == bestFeatureId) && (0 == sub_group_local_id));
        bool mySplitIsBest            = (leafMark != bestFeatureId && curFeatureId == bestFeatureId && curFeatureValue == bestFeatureValue);
        if (noneSplitFoundBySubGroup || mySplitIsBest)
        {
            if (1 == n_sub_groups)
            {
                __global algorithmFPType * splitNodeInfo = splitInfo + nodeId * nImpProp;
                nodeList[nodeId * nNodeProp + 2]         = curFeatureId == valNotFound ? leafMark : curFeatureId;
                nodeList[nodeId * nNodeProp + 3]         = curFeatureValue == valNotFound ? leafMark : curFeatureValue;
                nodeList[nodeId * nNodeProp + 4]         = (int)bestLN;

                splitNodeInfo[0] = imp;
                splitNodeInfo[1] = mrgMean;

                if (updateImpDecreaseRequired) nodeImpDecreaseList[nodeId] = curImpDec / mrgN;
            }
            else
            {
                bufS[(bufIdx + sub_group_id) * nNodeProp + 0] = curFeatureId;
                bufS[(bufIdx + sub_group_id) * nNodeProp + 1] = curFeatureValue;
                bufS[(bufIdx + sub_group_id) * nNodeProp + 2] = (int)bestLN;

                bufI[bufIdx + sub_group_id] = curImpDec;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        if (1 < n_sub_groups && 0 == sub_group_id)
        {
            // first sub group for current node reduces over local buffer if required
            algorithmFPType curImpDec = (sub_group_local_id < n_sub_groups) ? bufI[bufIdx + sub_group_local_id] : minImpDec;

            int curFeatureId    = sub_group_local_id < n_sub_groups ? bufS[(bufIdx + sub_group_local_id) * nNodeProp + 0] : valNotFound;
            int curFeatureValue = sub_group_local_id < n_sub_groups ? bufS[(bufIdx + sub_group_local_id) * nNodeProp + 1] : valNotFound;
            int LN              = sub_group_local_id < n_sub_groups ? bufS[(bufIdx + sub_group_local_id) * nNodeProp + 2] : 0;

            algorithmFPType bestImpDec = sub_group_reduce_max(curImpDec);

            int impDecIsBest     = fpEq(bestImpDec, curImpDec);
            int bestFeatureId    = sub_group_reduce_min(impDecIsBest ? curFeatureId : valNotFound);
            int bestFeatureValue = sub_group_reduce_min((bestFeatureId == curFeatureId && impDecIsBest) ? curFeatureValue : valNotFound);

            bool noneSplitFoundBySubGroup = ((leafMark == bestFeatureId) && (0 == sub_group_local_id));
            bool mySplitIsBest            = (leafMark != bestFeatureId && curFeatureId == bestFeatureId && curFeatureValue == bestFeatureValue);
            if (noneSplitFoundBySubGroup || mySplitIsBest)
            {
                __global algorithmFPType * splitNodeInfo = splitInfo + nodeId * nImpProp;
                nodeList[nodeId * nNodeProp + 2]         = curFeatureId == valNotFound ? leafMark : curFeatureId;
                nodeList[nodeId * nNodeProp + 3]         = curFeatureValue == valNotFound ? leafMark : curFeatureValue;
                nodeList[nodeId * nNodeProp + 4]         = (int)LN;

                splitNodeInfo[0] = imp;
                splitNodeInfo[1] = mrgMean;

                if (updateImpDecreaseRequired) nodeImpDecreaseList[nodeId] = curImpDec / mrgN;
            }
        }
    }

    __kernel void computePartialHistograms(const __global int * data, const __global int * treeOrder, const __global int * nodeList,
                                           const __global int * nodeIndices, int nodeIndicesOffset, const __global int * selectedFeatures,
                                           const __global algorithmFPType * response, const __global int * binOffsets, int nMaxBinsAmongFtrs,
                                           int nFeatures, __global algorithmFPType * partialHistograms) {
        const int nProp     = HIST_PROPS; // num of characteristics in histogram
        const int nNodeProp = NODE_PROPS; // num of node properties in nodeOffsets

        const int nodeIdx           = get_global_id(2);
        const int nodeId            = nodeIndices[nodeIndicesOffset + nodeIdx];
        const int featIdx           = get_local_id(1);
        const int nSelectedFeatures = get_local_size(1);
        const int histIdx           = get_global_id(0);
        const int nPartHist         = get_global_size(0);

        const int featId     = selectedFeatures[nodeId * nSelectedFeatures + featIdx];
        const int rowsOffset = nodeList[nodeId * nNodeProp + 0];
        const int nRows      = nodeList[nodeId * nNodeProp + 1];

        const int nElementsForGroup = nRows / nPartHist + !!(nRows % nPartHist);

        int iStart = histIdx * nElementsForGroup;
        int iEnd   = (histIdx + 1) * nElementsForGroup;

        iEnd = (iEnd > nRows) ? nRows : iEnd;

        __global algorithmFPType * histogram =
            partialHistograms + ((nodeIdx * nPartHist + histIdx) * nSelectedFeatures + featIdx) * nMaxBinsAmongFtrs * nProp;

        int nBins = binOffsets[featId + 1] - binOffsets[featId];

        for (int i = 0; i < nProp * nBins; i++)
        {
            histogram[i] = (algorithmFPType)0;
        }

        for (int i = iStart; i < iEnd; i++)
        {
            int id  = treeOrder[rowsOffset + i];
            int bin = data[id * nFeatures + featId];

            histogram[bin * nProp + 0] += 1.0; // N + 1
            algorithmFPType invN  = ((algorithmFPType)1) / histogram[bin * nProp + 0];
            algorithmFPType delta = response[id] - histogram[bin * nProp + 1];                 // y[i] - mean
            histogram[bin * nProp + 1] += delta * invN;                                        // updated mean
            histogram[bin * nProp + 2] += delta * (response[id] - histogram[bin * nProp + 1]); // updated sum2Cent
        }
    }

    __kernel void reducePartialHistograms(const __global algorithmFPType * partialHistograms, __global algorithmFPType * histograms,
                                          int nPartialHistograms, int nSelectedFeatures, int nMaxBinsAmongFtrs) {
        const int nProp = HIST_PROPS; // num of characteristics in histogram
        __local algorithmFPType buf[LOCAL_BUFFER_SIZE * nProp];

        const int nodeIdx    = get_global_id(2);
        const int binId      = get_global_id(0);
        const int local_id   = get_local_id(1);
        const int local_size = get_local_size(1);

        buf[local_id * nProp + 0] = (algorithmFPType)0;
        buf[local_id * nProp + 1] = (algorithmFPType)0;
        buf[local_id * nProp + 2] = (algorithmFPType)0;

        algorithmFPType mrgN        = (algorithmFPType)0;
        algorithmFPType mrgMean     = (algorithmFPType)0;
        algorithmFPType mrgSum2Cent = (algorithmFPType)0;

        const __global algorithmFPType * nodePartialHistograms =
            partialHistograms + nodeIdx * nPartialHistograms * nSelectedFeatures * nMaxBinsAmongFtrs * nProp;
        __global algorithmFPType * nodeHistogram = histograms + nodeIdx * nSelectedFeatures * nMaxBinsAmongFtrs * nProp;

        for (int i = local_id; i < nPartialHistograms; i += local_size)
        {
            int offset        = i * nSelectedFeatures * nMaxBinsAmongFtrs * nProp + binId * nProp;
            algorithmFPType n = nodePartialHistograms[offset + 0];

            if ((algorithmFPType)0 == n) continue;

            algorithmFPType mean     = nodePartialHistograms[offset + 1];
            algorithmFPType sum2Cent = nodePartialHistograms[offset + 2];

            mergeStat(n, mean, sum2Cent, &mrgN, &mrgMean, &mrgSum2Cent);

            buf[local_id * nProp + 0] += n;
            buf[local_id * nProp + 1] = mrgMean;
            buf[local_id * nProp + 2] = mrgSum2Cent;
        }

        for (int offset = local_size / 2; offset > 0; offset >>= 1)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            if (local_id < offset)
            {
                algorithmFPType n = buf[(local_id + offset) * nProp + 0];
                if ((algorithmFPType)0 == n) continue;
                algorithmFPType mean     = buf[(local_id + offset) * nProp + 1];
                algorithmFPType sum2Cent = buf[(local_id + offset) * nProp + 2];

                mergeStat(n, mean, sum2Cent, &mrgN, &mrgMean, &mrgSum2Cent);

                buf[local_id * nProp + 0] += n;
                buf[local_id * nProp + 1] = mrgMean;
                buf[local_id * nProp + 2] = mrgSum2Cent;
            }
        }

        if (local_id == 0)
        {
            // item 0 collects all results in private vars
            nodeHistogram[binId * nProp + 0] = mrgN;
            nodeHistogram[binId * nProp + 1] = mrgMean;
            nodeHistogram[binId * nProp + 2] = mrgSum2Cent;
        }
    });

#endif
