/* file: df_kernels.cl */
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
    df_batch_regression_kernels,

    int fpIsEqual(algorithmFPType a, algorithmFPType b) { return (int)(fabs(a - b) <= algorithmFPTypeAccuracy); }

    __kernel void initializeTreeOrder(__global int * treeOrder) {
        const int id  = get_global_id(0);
        treeOrder[id] = id;
    }

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
        //const int nNodeProp = 5; // num of node properties in nodeList
        //const int nImpProp  = 2;
        const int nNodeProp = NODE_PROPS; // num of node properties in nodeList
        const int nImpProp  = IMPURITY_PROPS;

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

        algorithmFPType curImpDec = (algorithmFPType)-1e30;
        int valNotFound           = 1 << 30;
        int curFeatureValue       = -1;
        int curFeatureId          = -1;

        nodeList[nodeId * nNodeProp + 2] = curFeatureId;
        nodeList[nodeId * nNodeProp + 3] = curFeatureValue;
        nodeList[nodeId * nNodeProp + 4] = nRows;

        algorithmFPType mrgN        = (algorithmFPType)0;
        algorithmFPType mrgMean     = (algorithmFPType)0;
        algorithmFPType mrgSum2Cent = (algorithmFPType)0;

        algorithmFPType bestRN        = (algorithmFPType)0;
        algorithmFPType bestRMean     = (algorithmFPType)0;
        algorithmFPType bestRSum2Cent = (algorithmFPType)0;

        algorithmFPType bestLN        = (algorithmFPType)0;
        algorithmFPType bestLMean     = (algorithmFPType)0;
        algorithmFPType bestLSum2Cent = (algorithmFPType)0;

        algorithmFPType mrgRN        = (algorithmFPType)0;
        algorithmFPType mrgRMean     = (algorithmFPType)0;
        algorithmFPType mrgRSum2Cent = (algorithmFPType)0;

        algorithmFPType mrgLN        = (algorithmFPType)0;
        algorithmFPType mrgLMean     = (algorithmFPType)0;
        algorithmFPType mrgLSum2Cent = (algorithmFPType)0;

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

            mrgRN        = (algorithmFPType)0;
            mrgRMean     = (algorithmFPType)0;
            mrgRSum2Cent = (algorithmFPType)0;

            mrgLN        = (algorithmFPType)0;
            mrgLMean     = (algorithmFPType)0;
            mrgLSum2Cent = (algorithmFPType)0;

            for (int row = 0; row < nRows; row++)
            {
                int id  = treeOrder[rowsOffset + row];
                int bin = data[id * nFeatures + featId];

                if (bin <= binId)
                {
                    mergeValToStat(response[id], &mrgLN, &mrgLMean, &mrgLSum2Cent);
                }
                else
                {
                    mergeValToStat(response[id], &mrgRN, &mrgRMean, &mrgRSum2Cent);
                }
            }

            mrgN        = mrgLN;
            mrgMean     = mrgLMean;
            mrgSum2Cent = mrgLSum2Cent;
            mergeStat(mrgRN, mrgRMean, mrgRSum2Cent, &mrgN, &mrgMean, &mrgSum2Cent);

            algorithmFPType impDec = mrgSum2Cent - (mrgLSum2Cent + mrgRSum2Cent);

            if ((algorithmFPType)0 < impDec && (mrgSum2Cent / mrgN) >= impurityThreshold
                && (curFeatureValue == -1 || impDec > curImpDec || (fpIsEqual(impDec, curImpDec) && featId < curFeatureId))
                && mrgRN >= minObservationsInLeafNode && mrgLN >= minObservationsInLeafNode)
            {
                curFeatureId    = featId;
                curFeatureValue = binId;
                curImpDec       = impDec;

                bestRN        = mrgRN;
                bestRMean     = mrgRMean;
                bestRSum2Cent = mrgRSum2Cent;

                bestLN        = mrgLN;
                bestLMean     = mrgLMean;
                bestLSum2Cent = mrgLSum2Cent;
            }
        } // for i

        algorithmFPType bestImpDec = sub_group_reduce_max(curImpDec);

        int impDecIsBest     = fpIsEqual(bestImpDec, curImpDec);
        int bestFeatureId    = sub_group_reduce_min(impDecIsBest ? curFeatureId : valNotFound);
        int bestFeatureValue = sub_group_reduce_min((bestFeatureId == curFeatureId && impDecIsBest) ? curFeatureValue : valNotFound);

        if (curFeatureId == bestFeatureId && curFeatureValue == bestFeatureValue)
        {
            if (1 == n_sub_groups)
            {
                __global algorithmFPType * splitNodeInfo = splitInfo + nodeId * nImpProp;
                nodeList[nodeId * nNodeProp + 2]         = curFeatureId == valNotFound ? -1 : curFeatureId;
                nodeList[nodeId * nNodeProp + 3]         = curFeatureValue == valNotFound ? -1 : curFeatureValue;
                nodeList[nodeId * nNodeProp + 4]         = (int)bestLN;

                splitNodeInfo[0] = ((algorithmFPType)0 != mrgN) ? mrgSum2Cent / mrgN : (algorithmFPType)0;
                splitNodeInfo[1] = mrgMean;

                if (updateImpDecreaseRequired) nodeImpDecreaseList[nodeId] = ((algorithmFPType)0 != mrgN) ? curImpDec / mrgN : (algorithmFPType)0;

                return;
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
            algorithmFPType curImpDec = (sub_group_local_id < n_sub_groups) ? bufI[bufIdx + sub_group_local_id] : (algorithmFPType)0;

            int curFeatureId    = sub_group_local_id < n_sub_groups ? bufS[(bufIdx + sub_group_local_id) * nNodeProp + 0] : valNotFound;
            int curFeatureValue = sub_group_local_id < n_sub_groups ? bufS[(bufIdx + sub_group_local_id) * nNodeProp + 1] : valNotFound;
            int LN              = sub_group_local_id < n_sub_groups ? bufS[(bufIdx + sub_group_local_id) * nNodeProp + 2] : 0;

            algorithmFPType bestImpDec = sub_group_reduce_max(curImpDec);

            int bestFeatureId    = sub_group_reduce_min(bestImpDec == curImpDec ? curFeatureId : valNotFound);
            int bestFeatureValue = sub_group_reduce_min((bestFeatureId == curFeatureId && bestImpDec == curImpDec) ? curFeatureValue : valNotFound);

            if (curFeatureId == bestFeatureId && curFeatureValue == bestFeatureValue)
            {
                __global algorithmFPType * splitNodeInfo = splitInfo + nodeId * nImpProp;
                nodeList[nodeId * nNodeProp + 2]         = curFeatureId == valNotFound ? -1 : curFeatureId;
                nodeList[nodeId * nNodeProp + 3]         = curFeatureValue == valNotFound ? -1 : curFeatureValue;
                nodeList[nodeId * nNodeProp + 4]         = (int)LN;

                splitNodeInfo[0] = ((algorithmFPType)0 != mrgN) ? mrgSum2Cent / mrgN : 0;
                splitNodeInfo[1] = mrgMean;

                if (updateImpDecreaseRequired) nodeImpDecreaseList[nodeId] = ((algorithmFPType)0 != mrgN) ? curImpDec / mrgN : (algorithmFPType)0;
            }
        }
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

        int valNotFound           = 1 << 30;
        int curFeatureValue       = -1;
        int curFeatureId          = -1;
        algorithmFPType curImpDec = (algorithmFPType)-1e30;

        nodeList[nodeId * nNodeProp + 2] = curFeatureId;
        nodeList[nodeId * nNodeProp + 3] = curFeatureValue;
        nodeList[nodeId * nNodeProp + 4] = nRows;

        algorithmFPType mrgN        = (algorithmFPType)0;
        algorithmFPType mrgMean     = (algorithmFPType)0;
        algorithmFPType mrgSum2Cent = (algorithmFPType)0;

        algorithmFPType bestRN        = (algorithmFPType)0;
        algorithmFPType bestRMean     = (algorithmFPType)0;
        algorithmFPType bestRSum2Cent = (algorithmFPType)0;

        algorithmFPType bestLN        = (algorithmFPType)0;
        algorithmFPType bestLMean     = (algorithmFPType)0;
        algorithmFPType bestLSum2Cent = (algorithmFPType)0;

        algorithmFPType mrgRN        = (algorithmFPType)0;
        algorithmFPType mrgRMean     = (algorithmFPType)0;
        algorithmFPType mrgRSum2Cent = (algorithmFPType)0;

        algorithmFPType mrgLN        = (algorithmFPType)0;
        algorithmFPType mrgLMean     = (algorithmFPType)0;
        algorithmFPType mrgLSum2Cent = (algorithmFPType)0;

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

            mrgRN        = (algorithmFPType)0;
            mrgRMean     = (algorithmFPType)0;
            mrgRSum2Cent = (algorithmFPType)0;

            mrgLN        = (algorithmFPType)0;
            mrgLMean     = (algorithmFPType)0;
            mrgLSum2Cent = (algorithmFPType)0;

            for (int tbin = 0; tbin < currFtrBins; tbin++)
            {
                int binOffset     = tbin * nProp;
                algorithmFPType n = histogramForFeature[binOffset + 0];

                algorithmFPType mean     = histogramForFeature[binOffset + 1];
                algorithmFPType sum2Cent = histogramForFeature[binOffset + 2];
                if ((algorithmFPType)0 == n) continue;

                if (tbin <= binId)
                {
                    mergeStat(n, mean, sum2Cent, &mrgLN, &mrgLMean, &mrgLSum2Cent);
                }
                else
                {
                    mergeStat(n, mean, sum2Cent, &mrgRN, &mrgRMean, &mrgRSum2Cent);
                }
            }

            mrgN        = mrgLN;
            mrgMean     = mrgLMean;
            mrgSum2Cent = mrgLSum2Cent;
            mergeStat(mrgRN, mrgRMean, mrgRSum2Cent, &mrgN, &mrgMean, &mrgSum2Cent);

            algorithmFPType impDec = mrgSum2Cent - (mrgLSum2Cent + mrgRSum2Cent);

            if ((algorithmFPType)0 < impDec && (mrgSum2Cent / mrgN) >= impurityThreshold
                && (curFeatureValue == -1 || impDec > curImpDec || (fpIsEqual(impDec, curImpDec) && featId < curFeatureId))
                && mrgRN >= minObservationsInLeafNode && mrgLN >= minObservationsInLeafNode)
            {
                curFeatureId    = featId;
                curFeatureValue = binId;
                curImpDec       = impDec;

                bestRN        = mrgRN;
                bestRMean     = mrgRMean;
                bestRSum2Cent = mrgRSum2Cent;

                bestLN        = mrgLN;
                bestLMean     = mrgLMean;
                bestLSum2Cent = mrgLSum2Cent;
            }
        } // for i

        algorithmFPType bestImpDec = sub_group_reduce_max(curImpDec);

        int impDecIsBest     = fpIsEqual(bestImpDec, curImpDec);
        int bestFeatureId    = sub_group_reduce_min(impDecIsBest ? curFeatureId : valNotFound);
        int bestFeatureValue = sub_group_reduce_min((bestFeatureId == curFeatureId && impDecIsBest) ? curFeatureValue : valNotFound);

        if (curFeatureId == bestFeatureId && curFeatureValue == bestFeatureValue)
        {
            if (1 == n_sub_groups)
            {
                __global algorithmFPType * splitNodeInfo = splitInfo + nodeId * nImpProp;
                nodeList[nodeId * nNodeProp + 2]         = curFeatureId == valNotFound ? -1 : curFeatureId;
                nodeList[nodeId * nNodeProp + 3]         = curFeatureValue == valNotFound ? -1 : curFeatureValue;
                nodeList[nodeId * nNodeProp + 4]         = (int)bestLN;

                splitNodeInfo[0] = ((algorithmFPType)0 != mrgN) ? mrgSum2Cent / mrgN : (algorithmFPType)0;
                splitNodeInfo[1] = mrgMean;

                if (updateImpDecreaseRequired) nodeImpDecreaseList[nodeId] = ((algorithmFPType)0 != mrgN) ? curImpDec / mrgN : (algorithmFPType)0;
                return;
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
            algorithmFPType curImpDec = (sub_group_local_id < n_sub_groups) ? bufI[bufIdx + sub_group_local_id] : (algorithmFPType)0;

            int curFeatureId    = sub_group_local_id < n_sub_groups ? bufS[(bufIdx + sub_group_local_id) * nNodeProp + 0] : valNotFound;
            int curFeatureValue = sub_group_local_id < n_sub_groups ? bufS[(bufIdx + sub_group_local_id) * nNodeProp + 1] : valNotFound;
            int LN              = sub_group_local_id < n_sub_groups ? bufS[(bufIdx + sub_group_local_id) * nNodeProp + 2] : 0;

            algorithmFPType bestImpDec = sub_group_reduce_max(curImpDec);

            int bestFeatureId    = sub_group_reduce_min(bestImpDec == curImpDec ? curFeatureId : valNotFound);
            int bestFeatureValue = sub_group_reduce_min((bestFeatureId == curFeatureId && bestImpDec == curImpDec) ? curFeatureValue : valNotFound);

            if (curFeatureId == bestFeatureId && curFeatureValue == bestFeatureValue)
            {
                __global algorithmFPType * splitNodeInfo = splitInfo + nodeId * nImpProp;
                nodeList[nodeId * nNodeProp + 2]         = curFeatureId == valNotFound ? -1 : curFeatureId;
                nodeList[nodeId * nNodeProp + 3]         = curFeatureValue == valNotFound ? -1 : curFeatureValue;
                nodeList[nodeId * nNodeProp + 4]         = (int)LN;

                splitNodeInfo[0] = ((algorithmFPType)0 != mrgN) ? mrgSum2Cent / mrgN : 0;
                splitNodeInfo[1] = mrgMean;

                if (updateImpDecreaseRequired) nodeImpDecreaseList[nodeId] = ((algorithmFPType)0 != mrgN) ? curImpDec / mrgN : (algorithmFPType)0;
                return;
            }
        }
    }

    __kernel void getNumOfSplitNodes(const __global int * nodeList, int nNodes, __global int * nSplitNodes) {
        const int local_id   = get_sub_group_local_id();
        const int local_size = get_sub_group_size();
        const int nNodeProp  = NODE_PROPS; // num of node properties in node
        const int badVal     = -1;

        int sum = 0;
        for (int i = local_id; i < nNodes; i += local_size)
        {
            sum += (int)(nodeList[i * nNodeProp + 2] != badVal);
        }

        sum = sub_group_reduce_add(sum);

        if (local_id == 0)
        {
            nSplitNodes[0] = sum;
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

            // will replace with mergeStat func call
            algorithmFPType sumN1N2    = mrgN + n;
            algorithmFPType mulN1N2    = mrgN * n;
            algorithmFPType deltaScale = mulN1N2 / sumN1N2;
            algorithmFPType meanScale  = (algorithmFPType)1 / sumN1N2;
            algorithmFPType delta      = mean - mrgMean;

            mrgSum2Cent = mrgSum2Cent + sum2Cent + delta * delta * deltaScale;
            mrgMean     = (mrgMean * mrgN + mean * n) * meanScale;
            mrgN        = sumN1N2;

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

                // will replace with mergeStat func call
                algorithmFPType sumN1N2    = mrgN + n;
                algorithmFPType mulN1N2    = mrgN * n;
                algorithmFPType deltaScale = mulN1N2 / sumN1N2;
                algorithmFPType meanScale  = (algorithmFPType)1 / sumN1N2;
                algorithmFPType delta      = mean - mrgMean;

                mrgSum2Cent = mrgSum2Cent + sum2Cent + delta * delta * deltaScale;
                mrgMean     = (mrgMean * mrgN + mean * n) * meanScale;
                mrgN        = sumN1N2;

                // item 0 collects all results in private vars
                // but all others need to store it
                if (local_id > 0)
                {
                    buf[local_id * nProp + 0] += n;
                    buf[local_id * nProp + 1] = mrgMean;
                    buf[local_id * nProp + 2] = mrgSum2Cent;
                }
            }
        }

        if (local_id == 0)
        {
            nodeHistogram[binId * nProp + 0] = mrgN;
            nodeHistogram[binId * nProp + 1] = mrgMean;
            nodeHistogram[binId * nProp + 2] = mrgSum2Cent;
        }
    }

    __kernel void partitionCopy(const __global int * treeOrderBuf, __global int * treeOrder, int offset) {
        const int id           = get_global_id(0);
        treeOrder[offset + id] = treeOrderBuf[offset + id];
    }

    __kernel void doLevelPartition(const __global int * data, const __global int * nodeList, const __global int * treeOrder,
                                   __global int * treeOrderBuf, int nFeatures) {
        const int nNodeProp  = NODE_PROPS; // num of split attributes for node
        const int local_id   = get_sub_group_local_id();
        const int nodeId     = get_global_id(1);
        const int local_size = get_sub_group_size();

        __global const int * node = nodeList + nodeId * nNodeProp;
        const int offset          = node[0];
        const int nRows           = node[1];
        const int featId          = node[2];
        const int splitVal        = node[3];
        const int nRowsLeft       = node[4]; // num of items in Left part

        if (featId < 0) // leaf node
        {
            return;
        }

        int sum = 0;

        for (int i = local_id; i < nRows; i += local_size)
        {
            int id                        = treeOrder[offset + i];
            int toRight                   = (int)(data[id * nFeatures + featId] > splitVal);
            int boundary                  = sum + sub_group_scan_exclusive_add(toRight);
            int posNew                    = (toRight ? nRowsLeft + boundary : i - boundary);
            treeOrderBuf[offset + posNew] = id;
            sum += sub_group_reduce_add(toRight);
        }
    }

    __kernel void convertSplitToLeaf(__global int * nodeList) {
        const int nNodeProp = NODE_PROPS; // num of split attributes for node
        const int badVal    = -1;
        const int id        = get_global_id(0);

        nodeList[id * nNodeProp + 2] = badVal;
        nodeList[id * nNodeProp + 3] = badVal;
    }

    __kernel void doNodesSplit(const __global int * nodeList, int nNodes, __global int * nodeListNew) {
        const int nNodeProp  = NODE_PROPS; // num of split attributes for node
        const int badVal     = -1;
        const int local_id   = get_sub_group_local_id();
        const int local_size = get_sub_group_size();

        int nCreatedNodes = 0;
        for (int i = local_id; i < nNodes; i += local_size)
        {
            int splitNode      = (int)(nodeList[i * nNodeProp + 2] != badVal); // featId != -1
            int newLeftNodePos = nCreatedNodes + sub_group_scan_exclusive_add(splitNode) * 2;
            if (splitNode)
            {
                // split parent node on left and right nodes
                __global const int * nodeP = nodeList + i * nNodeProp;
                __global int * nodeL       = nodeListNew + newLeftNodePos * nNodeProp;
                __global int * nodeR       = nodeListNew + (newLeftNodePos + 1) * nNodeProp;

                nodeL[0] = nodeP[0]; // rows offset
                nodeL[1] = nodeP[4]; // nRows
                nodeL[2] = badVal;   // featureId
                nodeL[3] = badVal;   // featureVal
                nodeL[4] = nodeP[4]; // num of items in Left part = nRows in new node

                nodeR[0] = nodeL[0] + nodeL[1];
                nodeR[1] = nodeP[1] - nodeL[1];
                nodeR[2] = badVal;
                nodeR[3] = badVal;
                nodeR[4] = nodeR[1]; // num of items in Left part = nRows in new node
            }
            nCreatedNodes += sub_group_reduce_add(splitNode) * 2;
        }
    }

    __kernel void splitNodeListOnGroupsBySize(const __global int * nodeList, int nNodes, __global int * bigNodesGroups, __global int * nodeIndeces,
                                              int minRowsBlock) {
        /*for now only 3 groups are produced, may be more required*/
        const int bigNodeLowBorderBlocksNum = BIG_NODE_LOW_BORDER_BLOCKS_NUM; // fine, need to experiment and find better one
        const int blockSize                 = minRowsBlock;
        const int nNodeProp                 = NODE_PROPS; // num of split attributes for node

        const int local_id   = get_sub_group_local_id();
        const int nodeId     = get_global_id(1);
        const int local_size = get_sub_group_size();

        int nBigNodes       = 0;
        int maxBigBlocksNum = 1;
        int nMidNodes       = 0;
        int maxMidBlocksNum = 1;

        /*calculate num of big and mid nodes*/
        for (int i = local_id; i < nNodes; i += local_size)
        {
            int nRows   = nodeList[i * nNodeProp + 1];
            int nBlocks = nRows / blockSize + !!(nRows % blockSize);
            int bigNode = 0;
            int midNode = 0;
            if (nBlocks > bigNodeLowBorderBlocksNum)
                bigNode = 1;
            else if (nBlocks > 1)
                midNode = 1;

            nBigNodes += sub_group_reduce_add(bigNode);
            nMidNodes += sub_group_reduce_add(midNode);
            maxBigBlocksNum = max(maxBigBlocksNum, bigNode ? nBlocks : 0);
            maxBigBlocksNum = sub_group_reduce_max(maxBigBlocksNum);
            maxMidBlocksNum = max(maxMidBlocksNum, midNode ? nBlocks : 0);
            maxMidBlocksNum = sub_group_reduce_max(maxMidBlocksNum);
        }

        nBigNodes = sub_group_broadcast(nBigNodes, 0);
        nMidNodes = sub_group_broadcast(nMidNodes, 0);

        if (0 == local_id)
        {
            bigNodesGroups[0] = nBigNodes;
            bigNodesGroups[1] = maxBigBlocksNum;
            bigNodesGroups[2] = nMidNodes;
            bigNodesGroups[3] = maxMidBlocksNum;
            bigNodesGroups[4] = nNodes - nBigNodes - nMidNodes;
            bigNodesGroups[5] = 1;
        }

        int sumBig = 0;
        int sumMid = 0;

        /*split nodes on groups*/
        for (int i = local_id; i < nNodes; i += local_size)
        {
            int nRows   = nodeList[i * nNodeProp + 1];
            int nBlocks = nRows / blockSize + !!(nRows % blockSize);
            int bigNode = 0;
            int midNode = 0;
            if (nBlocks > bigNodeLowBorderBlocksNum)
                bigNode = 1;
            else if (nBlocks > 1)
                midNode = 1;

            int boundaryBig     = sumBig + sub_group_scan_exclusive_add(bigNode);
            int boundaryMid     = sumMid + sub_group_scan_exclusive_add(midNode);
            int posNew          = (bigNode ? boundaryBig : midNode ? nBigNodes + boundaryMid : nBigNodes + nMidNodes + i - boundaryBig - boundaryMid);
            nodeIndeces[posNew] = i;
            sumBig += sub_group_reduce_add(bigNode);
            sumMid += sub_group_reduce_add(midNode);
        }
    }

    __kernel void updateMDIVarImportance(const __global int * nodeList, const __global algorithmFPType * nodeImpDecreaseList, int nNodes,
                                         __global algorithmFPType * featureImportanceList) {
        const int nNodeProp = NODE_PROPS; // num of node properties in nodeList

        const int local_id           = get_local_id(0);
        const int sub_group_local_id = get_sub_group_local_id();
        const int sub_group_size     = get_sub_group_size();
        const int local_size         = get_local_size(0);
        const int n_sub_groups       = local_size / sub_group_size; // num of subgroups for current node processing
        const int sub_group_id       = local_id / sub_group_size;
        const int max_sub_groups_num = 16; //replace with define

        const int bufIdx = get_global_id(1) % (max_sub_groups_num / n_sub_groups); // local buffer is shared between 16 sub groups
        const int ftrId  = get_global_id(1);

        const int leafMark             = -1;
        const int nElementsForSubgroup = nNodes / n_sub_groups + !!(nNodes % n_sub_groups);

        __local algorithmFPType bufI[max_sub_groups_num]; // storage for impurity decrease

        int iStart = sub_group_id * nElementsForSubgroup;
        int iEnd   = (sub_group_id + 1) * nElementsForSubgroup;

        iEnd = (iEnd > nNodes) ? nNodes : iEnd;

        algorithmFPType ftrImp = (algorithmFPType)0;

        for (int nodeIdx = iStart + local_id; nodeIdx < iEnd; nodeIdx += local_size)
        {
            int splitFtrId = nodeList[nodeIdx * nNodeProp + 2];
            ftrImp += sub_group_reduce_add((splitFtrId != leafMark && ftrId == splitFtrId) ? nodeImpDecreaseList[nodeIdx] : (algorithmFPType)0);
        }

        if (0 == sub_group_local_id)
        {
            if (1 == n_sub_groups)
            {
                featureImportanceList[ftrId] += ftrImp;
                return;
            }
            else
            {
                bufI[bufIdx + sub_group_id] = ftrImp;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        if (1 < n_sub_groups && 0 == sub_group_id)
        {
            // first sub group for current node reduces over local buffer if required
            algorithmFPType ftrImp      = (sub_group_local_id < n_sub_groups) ? bufI[bufIdx + sub_group_local_id] : (algorithmFPType)0;
            algorithmFPType totalFtrImp = sub_group_reduce_add(ftrImp);

            if (0 == local_id)
            {
                featureImportanceList[ftrId] += totalFtrImp;
            }
        }
    }

    __kernel void markPresentRows(const __global int * rowsList, __global int * rowsBuffer, int nRows) {
        const int n_groups             = get_num_groups(0);
        const int n_sub_groups         = get_num_sub_groups();
        const int n_total_sub_groups   = n_sub_groups * n_groups;
        const int nElementsForSubgroup = nRows / n_total_sub_groups + !!(nRows % n_total_sub_groups);
        const int local_size           = get_sub_group_size();

        const int id           = get_local_id(0);
        const int local_id     = get_sub_group_local_id();
        const int sub_group_id = get_sub_group_id();
        const int group_id     = get_group_id(0) * n_sub_groups + sub_group_id;

        const int itemPresentMark = 1;

        int iStart = group_id * nElementsForSubgroup;
        int iEnd   = (group_id + 1) * nElementsForSubgroup;

        iEnd = (iEnd > nRows) ? nRows : iEnd;

        for (int i = iStart + local_id; i < iEnd; i += local_size)
        {
            rowsBuffer[rowsList[i]] = itemPresentMark;
        }
    }

    __kernel void countAbsentRowsForBlocks(const __global int * rowsBuffer, __global int * partialSums, int nRows) {
        const int n_groups             = get_num_groups(0);
        const int n_sub_groups         = get_num_sub_groups();
        const int n_total_sub_groups   = n_sub_groups * n_groups;
        const int nElementsForSubgroup = nRows / n_total_sub_groups + !!(nRows % n_total_sub_groups);
        const int local_size           = get_sub_group_size();

        const int id           = get_local_id(0);
        const int local_id     = get_sub_group_local_id();
        const int sub_group_id = get_sub_group_id();
        const int group_id     = get_group_id(0) * n_sub_groups + sub_group_id;

        const int itemAbsentMark = -1;

        int iStart = group_id * nElementsForSubgroup;
        int iEnd   = (group_id + 1) * nElementsForSubgroup;

        iEnd = (iEnd > nRows) ? nRows : iEnd;

        int sum = 0;

        for (int i = iStart + local_id; i < iEnd; i += local_size)
        {
            int value = (int)(itemAbsentMark == rowsBuffer[i]);
            sum += sub_group_reduce_add(value);
        }

        if (local_id == 0)
        {
            partialSums[group_id] = sum;
        }
    }

    __kernel void countAbsentRowsTotal(const __global int * partialSums, __global int * partialPrefixSums, __global int * totalSum,
                                       int nSubgroupSums) {
        if (get_sub_group_id() > 0) return;

        const int local_size = get_sub_group_size();
        const int local_id   = get_sub_group_local_id();

        int sum = 0;

        for (int i = local_id; i < nSubgroupSums; i += local_size)
        {
            int value            = partialSums[i];
            int boundary         = sub_group_scan_exclusive_add(value);
            partialPrefixSums[i] = sum + boundary;
            sum += sub_group_reduce_add(value);
        }

        if (local_id == 0)
        {
            totalSum[0] = sum;
        }
    }

    __kernel void fillOOBRowsListByBlocks(const __global int * rowsBuffer, const __global int * partialPrefixSums, __global int * oobRowsList,
                                          int nRows) {
        const int n_groups           = get_num_groups(0);
        const int n_sub_groups       = get_num_sub_groups();
        const int n_total_sub_groups = n_sub_groups * n_groups;
        const int local_size         = get_sub_group_size();

        const int id           = get_local_id(0);
        const int local_id     = get_sub_group_local_id();
        const int sub_group_id = get_sub_group_id();
        const int group_id     = get_group_id(0) * n_sub_groups + sub_group_id;

        const int nElementsForSubgroup = nRows / n_total_sub_groups + !!(nRows % n_total_sub_groups);

        const int itemAbsentMark = -1;

        int iStart = group_id * nElementsForSubgroup;
        int iEnd   = (group_id + 1) * nElementsForSubgroup;

        iEnd = (iEnd > nRows) ? nRows : iEnd;

        int groupOffset = partialPrefixSums[group_id];
        int sum         = 0;

        for (int i = iStart + local_id; i < iEnd; i += local_size)
        {
            int oobRow = (int)(itemAbsentMark == rowsBuffer[i]);
            int pos    = groupOffset + sum + sub_group_scan_exclusive_add(oobRow);
            if (oobRow)
            {
                oobRowsList[pos] = i;
            }
            sum += sub_group_reduce_add(oobRow);
        }
    }

);

#endif
