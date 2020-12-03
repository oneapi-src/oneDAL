/* file: df_tree_level_build_helper_kernels.cl */
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
//  Implementation of tree level build helper OpenCL kernels.
//--
*/

#ifndef __DF_TREE_LEVEL_BUILD_HELPER_KERNELS_CL__
#define __DF_TREE_LEVEL_BUILD_HELPER_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE(name, src) static const char * name = #src;

DECLARE_SOURCE(
    df_tree_level_build_helper_kernels,

    __kernel void initializeTreeOrder(__global int * treeOrder) {
        const int id  = get_global_id(0);
        treeOrder[id] = id;
    }

    __kernel void partitionCopy(const __global int * treeOrderBuf, __global int * treeOrder, int offset) {
        const int id           = get_global_id(0);
        treeOrder[offset + id] = treeOrderBuf[offset + id];
    }

    __kernel void doLevelPartitionByGroups(const __global int * data, const __global int * nodeList, __global int * nodeAuxList,
                                           const __global int * treeOrder, __global int * treeOrderBuf, int nNodes, int nFeatures) {
        const int nNodeProp    = NODE_PROPS;     // num of split attributes for node
        const int nAuxNodeProp = AUX_NODE_PROPS; // num of auxilliary attributes for node
        const int maxBlocksNum = PARTITION_MAX_BLOCKS_NUM;
        const int minBlockSize = PARTITION_MIN_BLOCK_SIZE;
        const int leafMark     = -1;

        const int sub_group_size               = get_sub_group_size();
        const int work_group_size              = get_local_size(0);
        const int sub_groups_in_work_group_num = work_group_size / sub_group_size; // num of subgroups for current node processing

        const int sub_group_local_id  = get_sub_group_local_id();
        const int work_group_local_id = get_local_id(0);

        const int sub_group_id   = get_group_id(0) * sub_groups_in_work_group_num + work_group_local_id / sub_group_size;
        const int sub_groups_num = get_num_groups(0) * sub_groups_in_work_group_num; // num of subgroups for current node processing

        const int totalBlocksNum = nNodes * maxBlocksNum;

        for (int blockIndGlob = sub_group_id; blockIndGlob < totalBlocksNum; blockIndGlob += sub_groups_num)
        {
            const int nodeId   = blockIndGlob / maxBlocksNum;
            const int blockInd = blockIndGlob % maxBlocksNum;

            __global const int * node = nodeList + nodeId * nNodeProp;
            const int offset          = node[0];
            const int nRows           = node[1];
            const int featId          = node[2];
            const int splitVal        = node[3];
            const int nRowsLeft       = node[4]; // num of items in Left part of node

            int nodeBlocks = nRows / minBlockSize ? min(nRows / minBlockSize, maxBlocksNum) : 1;

            // if node has blocks less than maxBlocksNum then subgroup will just go to the next node
            if (featId != leafMark && blockInd < nodeBlocks) // split node
            {
                __global const int * nodeAux = nodeAuxList + nodeId * nAuxNodeProp;

                const int blockSize = nodeBlocks > 1 ? nRows / nodeBlocks + !!(nRows % nodeBlocks) : nRows;

                const int iEnd         = min((blockInd + 1) * blockSize, nRows);
                const int iStart       = min(blockInd * blockSize, iEnd);
                const int rowsForGroup = iEnd - iStart;

                int groupLeftBoundary  = 0;
                int groupRightBoundary = 0;

                if (nodeBlocks > 1 && rowsForGroup > 0)
                {
                    int groupRowsToRight = 0;
                    for (int i = iStart + sub_group_local_id; i < iEnd; i += sub_group_size)
                    {
                        int id      = treeOrder[offset + i];
                        int toRight = (int)(data[id * nFeatures + featId] > splitVal);
                        groupRowsToRight += sub_group_reduce_add(toRight);
                    }

                    if (0 == sub_group_local_id)
                    {
                        groupLeftBoundary  = atomic_add(nodeAux + 0, rowsForGroup - groupRowsToRight);
                        groupRightBoundary = atomic_add(nodeAux + 1, groupRowsToRight);
                    }
                    groupLeftBoundary  = sub_group_broadcast(groupLeftBoundary, 0);
                    groupRightBoundary = sub_group_broadcast(groupRightBoundary, 0);
                }

                int groupRowsToRight = 0;
                for (int i = iStart + sub_group_local_id; i < iEnd; i += sub_group_size)
                {
                    const int id                        = treeOrder[offset + i];
                    const int toRight                   = (int)(data[id * nFeatures + featId] > splitVal);
                    const int boundary                  = groupRowsToRight + sub_group_scan_exclusive_add(toRight);
                    const int posNew                    = (toRight ? nRowsLeft + groupRightBoundary + boundary : groupLeftBoundary + i - iStart - boundary);
                    treeOrderBuf[offset + posNew] = id;
                    groupRowsToRight += sub_group_reduce_add(toRight);
                }
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

    __kernel void convertSplitToLeaf(__global int * nodeList) {
        const int nNodeProp = NODE_PROPS; // num of split attributes for node
        const int leafMark  = -1;
        const int id        = get_global_id(0);

        nodeList[id * nNodeProp + 2] = leafMark;
        nodeList[id * nNodeProp + 3] = leafMark;
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

    __kernel void splitNodeListOnGroupsBySize(const __global int * nodeList, int nNodes, __global int * bigNodesGroups, __global int * nodeIndices,
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

            int bigNode = (int)(nBlocks > bigNodeLowBorderBlocksNum);
            int midNode = (int)(nBlocks <= bigNodeLowBorderBlocksNum && nBlocks > 1);

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
            int bigNode = (int)(nBlocks > bigNodeLowBorderBlocksNum);
            int midNode = (int)(nBlocks <= bigNodeLowBorderBlocksNum && nBlocks > 1);

            int boundaryBig = sumBig + sub_group_scan_exclusive_add(bigNode);
            int boundaryMid = sumMid + sub_group_scan_exclusive_add(midNode);
            int posNew      = (bigNode ? boundaryBig : (midNode ? nBigNodes + boundaryMid : nBigNodes + nMidNodes + i - boundaryBig - boundaryMid));
            nodeIndices[posNew] = i;
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

        for (int nodeIdx = iStart + sub_group_local_id; nodeIdx < iEnd; nodeIdx += sub_group_size)
        {
            int splitFtrId = nodeList[nodeIdx * nNodeProp + 2];
            ftrImp += sub_group_reduce_add((splitFtrId != leafMark && ftrId == splitFtrId) ? nodeImpDecreaseList[nodeIdx] : (algorithmFPType)0);
        }

        if (0 == sub_group_local_id)
        {
            if (1 == n_sub_groups)
            {
                featureImportanceList[ftrId] += ftrImp;
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

        int subSum = 0;

        for (int i = iStart + local_id; i < iEnd; i += local_size)
        {
            subSum += (int)(itemAbsentMark == rowsBuffer[i]);
        }

        int sum = sub_group_reduce_add(subSum);

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
