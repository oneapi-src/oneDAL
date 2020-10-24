/* file: df_batch_classification_kernels.cl */
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
//  Implementation of decision forest Batch classification OpenCL kernels.
//--
*/

#ifndef __DF_BATCH_PREDICT_CLASSIFICATION_KERNELS_CL__
#define __DF_BATCH_PREDICT_CLASSIFICATION_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE(name, src) static const char * name = #src;

DECLARE_SOURCE(
    df_batch_predict_classification_kernels,
    __kernel void predictByTreesWeighted(const __global algorithmFPType * data, const __global int * ftrIdx,
                                         const __global int * classLabelsOrNextNodeIdx, const __global algorithmFPType * ftrValue,
                                         const __global double * classProba, __global algorithmFPType * obsClassHist, algorithmFPType scale,
                                         int nRows, int nCols, int nTrees, int maxTreeSize, int treeOffset) {
        const int nClasses      = NUM_OF_CLASSES;
        const int local_id      = get_local_id(0);
        const int local_size    = get_local_size(0);
        const int n_groups      = get_num_groups(0);
        const int group_id      = get_group_id(0);
        const int n_tree_groups = get_num_groups(1);
        const int tree_group_id = get_group_id(1);
        const int tree_id       = treeOffset + tree_group_id;
        const int leafMark      = -1;

        const int nElementsForGroup = nRows / n_groups + !!(nRows % n_groups);

        const int iStart = group_id * nElementsForGroup;
        int iEnd         = (group_id + 1) * nElementsForGroup;

        iEnd = (iEnd > nRows) ? nRows : iEnd;

        if (tree_id < nTrees)
        {
            const __global int * ftrIdxForTree                   = ftrIdx + tree_id * maxTreeSize;
            const __global int * classLabelsOrNextNodeIdxForTree = classLabelsOrNextNodeIdx + tree_id * maxTreeSize;
            const __global algorithmFPType * ftrValueForTree     = ftrValue + tree_id * maxTreeSize;
            const __global double * classProbaForTree            = classProba + tree_id * maxTreeSize * nClasses;

            uint treeRootIsSplit = (uint)(leafMark != ftrIdxForTree[0]);

            for (int i = iStart + local_id; i < iEnd; i += local_size)
            {
                uint obsCurrNodeForTree  = 0;
                uint obsSplitMarkForTree = treeRootIsSplit;
                for (; obsSplitMarkForTree > 0;)
                {
                    uint idx = obsSplitMarkForTree * ftrIdxForTree[obsCurrNodeForTree];
                    uint sn  = (uint)(data[i * nCols + idx] > ftrValueForTree[obsCurrNodeForTree]);
                    obsCurrNodeForTree -= obsSplitMarkForTree * (obsCurrNodeForTree - (uint)classLabelsOrNextNodeIdxForTree[obsCurrNodeForTree] - sn);
                    obsSplitMarkForTree = (uint)(ftrIdxForTree[obsCurrNodeForTree] != leafMark);
                }
                for (int clIdx = 0; clIdx < nClasses; clIdx++)
                {
                    obsClassHist[i * n_tree_groups * nClasses + clIdx * n_tree_groups + tree_group_id] +=
                        scale * (algorithmFPType)classProbaForTree[obsCurrNodeForTree * nClasses + clIdx];
                }
            }
        }
    }

    __kernel void predictByTreesUnweighted(const __global algorithmFPType * data, const __global int * ftrIdx,
                                           const __global int * classLabelsOrNextNodeIdx, const __global algorithmFPType * ftrValue,
                                           __global algorithmFPType * obsClassHist, algorithmFPType scale, int nRows, int nCols, int nTrees,
                                           int maxTreeSize, int treeOffset) {
        const int nClasses      = NUM_OF_CLASSES;
        const int local_id      = get_local_id(0);
        const int local_size    = get_local_size(0);
        const int n_groups      = get_num_groups(0);
        const int group_id      = get_group_id(0);
        const int n_tree_groups = get_num_groups(1);
        const int tree_group_id = get_group_id(1);
        const int tree_id       = treeOffset + tree_group_id;
        const int leafMark      = -1;

        const int nElementsForGroup = nRows / n_groups + !!(nRows % n_groups);

        const int iStart = group_id * nElementsForGroup;
        int iEnd         = (group_id + 1) * nElementsForGroup;

        iEnd = (iEnd > nRows) ? nRows : iEnd;

        if (tree_id < nTrees)
        {
            const __global int * ftrIdxForTree                   = ftrIdx + tree_id * maxTreeSize;
            const __global int * classLabelsOrNextNodeIdxForTree = classLabelsOrNextNodeIdx + tree_id * maxTreeSize;
            const __global algorithmFPType * ftrValueForTree     = ftrValue + tree_id * maxTreeSize;

            uint treeRootIsSplit = (uint)(leafMark != ftrIdxForTree[0]);

            for (int i = iStart + local_id; i < iEnd; i += local_size)
            {
                uint obsCurrNodeForTree  = 0;
                uint obsSplitMarkForTree = treeRootIsSplit;
                for (; obsSplitMarkForTree > 0;)
                {
                    uint idx = obsSplitMarkForTree * ftrIdxForTree[obsCurrNodeForTree];
                    uint sn  = (uint)(data[i * nCols + idx] > ftrValueForTree[obsCurrNodeForTree]);
                    obsCurrNodeForTree -= obsSplitMarkForTree * (obsCurrNodeForTree - (uint)classLabelsOrNextNodeIdxForTree[obsCurrNodeForTree] - sn);
                    obsSplitMarkForTree = (uint)(ftrIdxForTree[obsCurrNodeForTree] != leafMark);
                }
                int clIdx = classLabelsOrNextNodeIdxForTree[obsCurrNodeForTree];
                obsClassHist[i * n_tree_groups * nClasses + clIdx * n_tree_groups + tree_group_id] += scale;
            }
        }
    }

    __kernel void reduceClassHist(__global algorithmFPType * obsClassHist, __global algorithmFPType * resObsClassHist, int nRows, int nTrees) {
        const int nClasses           = NUM_OF_CLASSES;
        const int group_id           = get_group_id(0);
        const int n_groups           = get_num_groups(0);
        const int sub_group_local_id = get_sub_group_local_id();
        const int sub_group_size     = get_sub_group_size();

        const int nElementsForGroup = nRows / n_groups + !!(nRows % n_groups);

        const int iStart = group_id * nElementsForGroup;
        int iEnd         = (group_id + 1) * nElementsForGroup;

        iEnd = (iEnd > nRows) ? nRows : iEnd;

        // obsClassHist each row contains certain class values from each tree for this observation
        // obsClassHist[0] = obs0_cls0_val_from_tree0, obs0_cls0_val_from_tree1 ...
        // obsClassHist[1] = obs0_cls1_val_from_tree0, obs0_cls1_val_from_tree1 ...

        for (int rowIdx = iStart; rowIdx < iEnd; rowIdx++)
        {
            for (int clIdx = 0; clIdx < nClasses; clIdx++)
            {
                int class_offset = rowIdx * nTrees * nClasses + clIdx * nTrees;

                algorithmFPType class_val = (algorithmFPType)0;
                for (int i = sub_group_local_id; i < nTrees; i += sub_group_size)
                {
                    class_val += obsClassHist[class_offset + i];
                }

                class_val = sub_group_reduce_add(class_val);

                if (0 == sub_group_local_id)
                {
                    resObsClassHist[rowIdx * nClasses + clIdx] = class_val;
                }
            }
        }
    }

    __kernel void determineWinners(const __global algorithmFPType * classHist, __global algorithmFPType * res, int nRows) {
        const int nClasses   = NUM_OF_CLASSES;
        const int local_id   = get_local_id(0);
        const int local_size = get_local_size(0);
        const int n_groups   = get_num_groups(0);
        const int group_id   = get_group_id(0);

        const int nElementsForGroup = nRows / n_groups + !!(nRows % n_groups);

        const int iStart = group_id * nElementsForGroup;
        int iEnd         = (group_id + 1) * nElementsForGroup;

        iEnd = (iEnd > nRows) ? nRows : iEnd;

        for (int i = iStart + local_id; i < iEnd; i += local_size)
        {
            algorithmFPType clsCount  = (algorithmFPType)0;
            algorithmFPType clsWinner = (algorithmFPType)0;
            for (int clIdx = 0; clIdx < nClasses; clIdx++)
            {
                if (clsCount < classHist[i * nClasses + clIdx])
                {
                    clsCount  = classHist[i * nClasses + clIdx];
                    clsWinner = (algorithmFPType)clIdx;
                }
            }
            res[i] = clsWinner;
        }
    });

#endif
