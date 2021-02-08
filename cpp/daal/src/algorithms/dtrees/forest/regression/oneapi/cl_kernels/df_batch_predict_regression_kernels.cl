/* file: df_batch_regression_kernels.cl */
/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
//  Implementation of decision forest Batch regression OpenCL kernels.
//--
*/

#ifndef __DF_BATCH_PREDICT_REGRESSION_KERNELS_CL__
#define __DF_BATCH_PREDICT_REGRESSION_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE(name, src) static const char * name = #src;

DECLARE_SOURCE(
    df_batch_predict_regression_kernels,
    __kernel void predictByTreesGroup(const __global algorithmFPType * data, const __global int * ftrIdx,
                                      const __global int * classLabelsOrNextNodeIdx, const __global algorithmFPType * ftrValueOrResponse,
                                      __global algorithmFPType * obsResponses, int nRows, int nCols, int nTrees, int maxTreeSize, int treeOffset) {
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
            const __global int * ftrIdxForTree                         = ftrIdx + tree_id * maxTreeSize;
            const __global int * classLabelsOrNextNodeIdxForTree       = classLabelsOrNextNodeIdx + tree_id * maxTreeSize;
            const __global algorithmFPType * ftrValueOrResponseForTree = ftrValueOrResponse + tree_id * maxTreeSize;

            uint treeRootIsSplit = (uint)(leafMark != ftrIdxForTree[0]);

            for (int i = iStart + local_id; i < iEnd; i += local_size)
            {
                uint obsCurrNodeForTree  = 0;
                uint obsSplitMarkForTree = treeRootIsSplit;
                for (; obsSplitMarkForTree > 0;)
                {
                    uint idx = obsSplitMarkForTree * ftrIdxForTree[obsCurrNodeForTree];
                    uint sn  = (uint)(data[i * nCols + idx] > ftrValueOrResponseForTree[obsCurrNodeForTree]);
                    obsCurrNodeForTree -= obsSplitMarkForTree * (obsCurrNodeForTree - (uint)classLabelsOrNextNodeIdxForTree[obsCurrNodeForTree] - sn);
                    obsSplitMarkForTree = (uint)(ftrIdxForTree[obsCurrNodeForTree] != leafMark);
                }
                obsResponses[i * n_tree_groups + tree_group_id] += ftrValueOrResponseForTree[obsCurrNodeForTree];
            }
        }
    }

    __kernel void reduceResponse(__global algorithmFPType * obsResponses, __global algorithmFPType * resObsResponse, int nRows, int nTrees,
                                 algorithmFPType scale) {
        const int group_id           = get_group_id(0);
        const int n_groups           = get_num_groups(0);
        const int sub_group_local_id = get_sub_group_local_id();
        const int sub_group_size     = get_sub_group_size();

        const int nElementsForGroup = nRows / n_groups + !!(nRows % n_groups);

        const int iStart = group_id * nElementsForGroup;
        int iEnd         = (group_id + 1) * nElementsForGroup;

        iEnd = (iEnd > nRows) ? nRows : iEnd;

        // obsResponses each row contains responses from each tree for this observation
        // obsResponses[0] = obs0_resp0_from_tree0, obs0_resp1_from_tree1 ...
        // obsResponses[1] = obs1_resp0_from_tree0, obs1_resp1_from_tree1 ...

        for (int rowIdx = iStart; rowIdx < iEnd; rowIdx++)
        {
            int resp_offset = rowIdx * nTrees;

            algorithmFPType resp_val = (algorithmFPType)0;
            for (int i = sub_group_local_id; i < nTrees; i += sub_group_size)
            {
                resp_val += obsResponses[resp_offset + i];
            }

            resp_val = sub_group_reduce_add(resp_val);

            if (0 == sub_group_local_id)
            {
                resObsResponse[rowIdx] = resp_val * scale;
            }
        }
    });

#endif
