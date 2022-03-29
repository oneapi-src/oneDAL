/** file tsne_gradient_descent_fpt.cpp */
/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef __INTERNAL_TSNE_GRADIENT_DESCENT_FPT_CPP__
#define __INTERNAL_TSNE_GRADIENT_DESCENT_FPT_CPP__

#include "algorithms/tsne/tsne_gradient_descent.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/csr_numeric_table.h"
#include "services/daal_defines.h"
#include "services/env_detect.h"
#include "src/externals/service_math.h"
#include "src/externals/service_dispatch.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_utils.h"
#include "src/services/service_defines.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"

#include <algorithm>
#include <iostream>
//#include <execution>

using namespace daal::data_management;
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace internal
{
template <typename DataType, CpuType cpu>
class TlsMax : public daal::TlsMem<DataType, cpu, services::internal::ScalableCalloc<DataType, cpu> >
{
public:
    typedef daal::TlsMem<DataType, cpu, services::internal::ScalableCalloc<DataType, cpu> > super;
    TlsMax(size_t n) : super(n) {}
    void reduceTo(DataType * res, size_t n)
    {
        bool bFirst = true;
        this->reduce([=, &bFirst](DataType * ptr) -> void {
            if (!ptr) return;
            if (bFirst)
            {
                for (size_t i = 0; i < n; ++i) res[i] = ptr[i];
                bFirst = false;
            }
            else
            {
                for (size_t i = 0; i < n; ++i) res[i] = services::internal::max<cpu, DataType>(res[i], ptr[i]);
            }
        });
    }
};

template <typename IdxType, daal::CpuType cpu>
services::Status maxRowElementsImpl(const size_t * row, const IdxType N, IdxType & nElements, const IdxType & blockOfRows)
{
    TlsMax<IdxType, cpu> maxTlsData(1);
    const IdxType nThreads    = threader_get_threads_number();
    const IdxType sizeOfBlock = services::internal::min<cpu, IdxType>(blockOfRows, N / nThreads + 1);
    const IdxType nBlocks     = N / sizeOfBlock + !!(N % sizeOfBlock);

    daal::threader_for(nBlocks, nBlocks, [&](IdxType iBlock) {
        const IdxType iStart = iBlock * sizeOfBlock;
        const IdxType iEnd   = services::internal::min<cpu, IdxType>(N, iStart + sizeOfBlock);
        IdxType * localMax   = maxTlsData.local();
        for (IdxType i = iStart; i < iEnd; ++i)
        {
            localMax[0] = services::internal::max<cpu, IdxType>(localMax[0], IdxType((row[i + 1] - row[i])));
        }
    });
    maxTlsData.reduceTo(&nElements, 1);

    return services::Status();
}

template <typename IdxType, typename DataType, daal::CpuType cpu>
services::Status boundingBoxKernelImpl(DataType * posX, DataType * posY, const IdxType N, const IdxType nNodes, DataType & radius,
                                       const IdxType & blockOfRows)
{
    DataType box[4]       = { posX[0], posX[0], posY[0], posY[0] };
    const DataType boxEps = 1e-3;

    daal::static_tls<DataType *> tlsBox([=]() {
        DataType * localBox = services::internal::service_malloc<DataType, cpu>(4);
        localBox[0]         = daal::services::internal::MaxVal<DataType>::get();
        localBox[1]         = -daal::services::internal::MaxVal<DataType>::get();
        localBox[2]         = daal::services::internal::MaxVal<DataType>::get();
        localBox[3]         = -daal::services::internal::MaxVal<DataType>::get();
        return localBox;
    });
    const IdxType nThreads    = tlsBox.nthreads();
    const IdxType sizeOfBlock = services::internal::min<cpu, IdxType>(blockOfRows, N / nThreads + 1);
    const IdxType nBlocks     = N / sizeOfBlock + !!(N % sizeOfBlock);

    daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        const IdxType iStart = iBlock * sizeOfBlock;
        const IdxType iEnd   = services::internal::min<cpu, IdxType>(N, iStart + sizeOfBlock);
        DataType * localBox  = tlsBox.local(tid);

        for (IdxType i = iStart; i < iEnd; ++i)
        {
            localBox[0] = services::internal::min<cpu, DataType>(localBox[0], posX[i]);
            localBox[1] = services::internal::max<cpu, DataType>(localBox[1], posX[i]);
            localBox[2] = services::internal::min<cpu, DataType>(localBox[2], posY[i]);
            localBox[3] = services::internal::max<cpu, DataType>(localBox[3], posY[i]);
        }
    });

    tlsBox.reduce([&](DataType * ptr) -> void {
        if (!ptr) return;
        box[0] = services::internal::min<cpu, DataType>(box[0], ptr[0]);
        box[1] = services::internal::max<cpu, DataType>(box[1], ptr[1]);
        box[2] = services::internal::min<cpu, DataType>(box[2], ptr[2]);
        box[3] = services::internal::max<cpu, DataType>(box[3], ptr[3]);
        services::internal::service_free<DataType, cpu>(ptr);
    });

    //scale the maximum to get all points strictly in the bounding box
    if (box[1] >= 0.)
        box[1] = (box[1] * (DataType(1.) + boxEps));
    else
        box[1] = (box[1] * (DataType(1.) - boxEps));
    if (box[3] >= 0.)
        box[3] = (box[3] * (DataType(1.) + boxEps));
    else
        box[3] = (box[3] * (DataType(1.) - boxEps));

    //save results
    radius       = services::internal::max<cpu, DataType>(box[1] - box[0], box[3] - box[2]) * DataType(0.5);
    posX[nNodes] = (box[0] + box[1]) * DataType(0.5);
    posY[nNodes] = (box[2] + box[3]) * DataType(0.5);

    return services::Status();
}

// template <typename IdxType, typename DataType, daal::CpuType cpu>
// services::Status boundingBoxKernelImpl(DataType * posx, DataType * posy, const IdxType N, const IdxType nNodes, DataType & radius, const IdxType & blockOfRows)
// {
//     DataType box[4] = { posx[0], posx[0], posy[0], posy[0] };

//     daal::static_tls<DataType *> tlsBox([=]() {
//         auto localBox = services::internal::service_malloc<DataType, cpu>(4);
//         localBox[0]   = daal::services::internal::MaxVal<DataType>::get();
//         localBox[1]   = -daal::services::internal::MaxVal<DataType>::get();
//         localBox[2]   = daal::services::internal::MaxVal<DataType>::get();
//         localBox[3]   = -daal::services::internal::MaxVal<DataType>::get();
//         return localBox;
//     });
//     const IdxType nThreads    = tlsBox.nthreads();
//     const IdxType sizeOfBlock = services::internal::min<cpu, IdxType>(256, N / nThreads + 1);
//     const IdxType nBlocks     = N / sizeOfBlock + !!(N % sizeOfBlock);

//     daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
//         const IdxType iStart = iBlock * sizeOfBlock;
//         const IdxType iEnd   = services::internal::min<cpu, IdxType>(N, iStart + sizeOfBlock);
//         DataType * localBox  = tlsBox.local(tid);

//         for (IdxType i = iStart; i < iEnd; ++i)
//         {
//             localBox[0] = services::internal::min<cpu, DataType>(localBox[0], posx[i]);
//             localBox[1] = services::internal::max<cpu, DataType>(localBox[1], posx[i]);
//             localBox[2] = services::internal::min<cpu, DataType>(localBox[2], posy[i]);
//             localBox[3] = services::internal::max<cpu, DataType>(localBox[3], posy[i]);
//         }
//     });

//     tlsBox.reduce([&](DataType * ptr) -> void {
//         if (!ptr) return;
//         box[0] = services::internal::min<cpu, DataType>(box[0], ptr[0]);
//         box[1] = services::internal::max<cpu, DataType>(box[1], ptr[1]);
//         box[2] = services::internal::min<cpu, DataType>(box[2], ptr[2]);
//         box[3] = services::internal::max<cpu, DataType>(box[3], ptr[3]);
//         services::internal::service_free<DataType, cpu>(ptr);
//     });

//     radius       = services::internal::max<cpu, DataType>(box[1] - box[0], box[3] - box[2]) * 0.5;
//     //scale the maximum to get all points strictly in the bounding box
//     if (box[1] >= 0.)
//         box[1] = services::internal::max<cpu, DataType>(box[1] * (1. + 1e-3), box[1] + radius*1e-3);
//     else
//         box[1] = services::internal::max<cpu, DataType>(box[1] * (1. - 1e-3), box[1] + radius*1e-3);
//     if (box[3] >= 0.)
//         box[3] = services::internal::max<cpu, DataType>(box[3] * (1. + 1e-3), box[3] + radius*1e-3);
//     else
//         box[3] = services::internal::max<cpu, DataType>(box[3] * (1. - 1e-3), box[3] + radius*1e-3);

//     //save results
//     radius       = services::internal::max<cpu, DataType>(box[1] - box[0], box[3] - box[2]) * 0.5;
//     posx[nNodes] = (box[0] + box[1]) * 0.5;
//     posy[nNodes] = (box[2] + box[3]) * 0.5;

//     return services::Status();
// }

#define MAX_LEVEL           32
#define SEQ_UPTO_LEVEL      4
#define SEQ_SPLIT_LIST_SIZE 1364

template <typename IdxType>
struct CodeType
{
    uint64_t morton;
    IdxType index;
};

template <typename IdxType>
struct SplitType
{
    IdxType level;
    IdxType m_index;
};

template <typename IdxType>
struct TreeNode
{
    IdxType first, second;
    IdxType pos;
    IdxType parent;
    IdxType child[4];
    IdxType child_internal[4];
};

template <typename IdxType, daal::CpuType cpu>
IdxType build_tree(CodeType<IdxType> * morton_code, SplitType<IdxType> * split_list, const IdxType split_list_size, IdxType & tree_allocation,
                   IdxType * level_size, IdxType root_level, TreeNode<IdxType> *& tree)
{
    IdxType cur_parent_node = 0;
    IdxType cur_node        = 1;
    TreeNode<IdxType> tree_node;
    tree_node.parent = -1;
    tree_node.pos    = -1;
    for (int i = 0; i < 4; i++)
    {
        tree_node.child[i]          = -1;
        tree_node.child_internal[i] = 0;
    }

    // std::cout << "$tree_node.first = " << tree_node.first << std::endl;
    // std::cout << "$tree_node.second = " << tree_node.second << std::endl;
    // std::cout << "$tree_node.parent = " << tree_node.parent << std::endl;
    // std::cout << "$tree_node.pos = " << tree_node.pos << std::endl;
    // for (int itt = 0; itt < 4 ; itt++) std::cout << "$tree_node.child[" << itt << "] = " << tree_node.child[itt] << std::endl;
    // for (int itt = 0; itt < 4 ; itt++) std::cout << "$tree_node.child_internal[" << itt << "] = " << tree_node.child_internal[itt] << std::endl;
    // std::cout << std::endl;

    level_size[root_level] = 1;
    IdxType i              = 0;
    IdxType lev            = root_level;
    IdxType split_level, split_m_index;
    split_level   = split_list[0].level;
    split_m_index = split_list[0].m_index;
    while (i < split_list_size && lev < MAX_LEVEL)
    {
        for (IdxType k = 0; k < level_size[lev]; k++)
        {
            IdxType first  = tree[cur_parent_node].first;
            IdxType second = tree[cur_parent_node].second;
            IdxType prev   = first;

            // Add all the children of the current_parent_node
            while (i < split_list_size && split_level == lev && split_m_index < second)
            {
                if (split_m_index < prev)
                {
                    //fprintf(stderr, "ERROR split_list_size = %d, lev = %d, i = %d, k = %d, prev = %d, split_m_index = %d, split_level = %d, cur_parent_node = %d, first = %d, second = %d\n",
                    //               split_list_size, lev, i, k, prev, split_m_index, split_level, cur_parent_node, first, second);
                    exit(0);
                }

                if (split_m_index == prev) // leaf node
                {
                    IdxType j                               = (morton_code[split_m_index].morton >> (64 - 2 * lev - 2)) & 0x3;
                    tree[cur_parent_node].child[j]          = morton_code[split_m_index].index;
                    tree[cur_parent_node].child_internal[j] = -1;
                    prev++;
                }
                else // internal node
                {
                    if (cur_node >= tree_allocation)
                    {
                        // fprintf(stderr, "allocation = %d, cur_node = %d. Reallocating\n", tree_allocation, cur_node);
                        DAAL_OVERFLOW_CHECK_BY_ADDING(IdxType, tree_allocation, IdxType(tree_allocation * 0.3));
                        IdxType new_allocation       = tree_allocation + IdxType(tree_allocation * 0.3);
                        TreeNode<IdxType> * new_tree = services::internal::service_scalable_malloc<TreeNode<IdxType>, cpu>(new_allocation);
                        //TreeNode<IdxType> * new_tree = services::internal::service_scalable_calloc<TreeNode<IdxType>, cpu>(new_allocation);
                        DAAL_CHECK_MALLOC(new_tree);
                        services::internal::tmemcpy<TreeNode<IdxType>, cpu>(new_tree, tree, tree_allocation);
                        services::internal::service_scalable_free<TreeNode<IdxType>, cpu>(tree);
                        tree = new_tree;
                        DAAL_CHECK_MALLOC(tree);
                        tree_allocation = new_allocation;
                        //exit(0);
                    }
                    IdxType j                               = (morton_code[split_m_index].morton >> (64 - 2 * lev - 2)) & 0x3;
                    tree[cur_parent_node].child[j]          = cur_node;
                    tree[cur_parent_node].child_internal[j] = 1;
                    tree_node.parent                        = cur_parent_node;
                    tree_node.first                         = prev;
                    tree_node.second                        = split_list[i].m_index;
                    prev                                    = split_list[i].m_index + 1;
                    tree[cur_node]                          = tree_node;
                    level_size[lev + 1]++;
                    cur_node++;
                }
                i++;
                if (i == split_list_size) break;
                split_level   = split_list[i].level;
                split_m_index = split_list[i].m_index;
            }
            // Add the last child
            if (prev == second) // node is a leaf
            {
                IdxType j                               = (morton_code[prev].morton >> (64 - 2 * lev - 2)) & 0x3;
                tree[cur_parent_node].child[j]          = morton_code[second].index;
                tree[cur_parent_node].child_internal[j] = -1;
            }
            else // internal node
            {
                if (cur_node >= tree_allocation)
                {
                    //   fprintf(stderr, "allocation = %d, cur_node = %d. Reallocating\n", tree_allocation, cur_node);
                    DAAL_OVERFLOW_CHECK_BY_ADDING(IdxType, tree_allocation, IdxType(tree_allocation * 0.3));
                    IdxType new_allocation       = tree_allocation + IdxType(tree_allocation * 0.3);
                    TreeNode<IdxType> * new_tree = services::internal::service_scalable_malloc<TreeNode<IdxType>, cpu>(new_allocation);
                    //TreeNode<IdxType> * new_tree = services::internal::service_scalable_calloc<TreeNode<IdxType>, cpu>(new_allocation);
                    DAAL_CHECK_MALLOC(new_tree);
                    services::internal::tmemcpy<TreeNode<IdxType>, cpu>(new_tree, tree, tree_allocation);
                    services::internal::service_scalable_free<TreeNode<IdxType>, cpu>(tree);
                    tree = new_tree;
                    DAAL_CHECK_MALLOC(tree);
                    tree_allocation = new_allocation;
                    //exit(0);
                }
                IdxType j                               = (morton_code[second].morton >> (64 - 2 * lev - 2)) & 0x3;
                tree[cur_parent_node].child[j]          = cur_node;
                tree[cur_parent_node].child_internal[j] = 1;
                tree_node.parent                        = cur_parent_node;
                tree_node.first                         = prev;
                tree_node.second                        = second;
                tree[cur_node]                          = tree_node;
                level_size[lev + 1]++;
                cur_node++;
            }

            cur_parent_node++;
        }
        lev++;
    }
    return cur_node;
}

// template <typename IdxType>
// inline void sort_splits(SplitType<IdxType> *in_start, IdxType cnt, SplitType<IdxType>* out_start)
// {
//     int i, t, a, hist[MAX_LEVEL + 1]={};

//     std::cout << "hist 0 =  ";
//     for (int j = 0; j < MAX_LEVEL; j++) std::cout << hist[j] << "  " ;
//     std::cout << std::endl;
//     std::cout << "sort_splits: debug 0" << std::endl;

//     for (i = 0; i < cnt; i++) hist[in_start[i].level]++;
//     std::cout << "sort_splits: debug 1" << std::endl;
//     for (a = -1, i = 0; i < MAX_LEVEL; i++) { t = hist[i] + a; hist[i] = a; a = t; }
//     std::cout << "sort_splits: debug 2" << std::endl;

//     std::cout << "hist 1 =  ";
//     for (int j = 0; j < MAX_LEVEL; j++) std::cout << hist[j] << "  " ;
//     std::cout << std::endl;

//     std::cout << "in_start[].level =  ";
//     for (int j = 0; j < MAX_LEVEL; j++) std::cout << hist[j] << "  " ;
//     std::cout << std::endl;

//     //for (i = 0; i < cnt; i++) out_start[++hist[in_start[i].level]] = in_start[i];

//     for (i = 0; i < cnt; i++)
//     {
//         std::cout << "i = " << i << std::endl;
//         std::cout << "in_start[i].level = " << in_start[i].level << std::endl;
//         std::cout << "hist[in_start[i].level] = " << hist[in_start[i].level] << std::endl;
//         hist[in_start[i].level] += 1;
//         std::cout << "hist[in_start[i].level] + 1 = " << hist[in_start[i].level] << std::endl;
//         out_start[hist[in_start[i].level]] = in_start[i];
//     }

//     std::cout << "sort_splits: debug 3" << std::endl;
// }

template <typename IdxType>

inline void sort_splits(SplitType<IdxType> * in_start, IdxType cnt, SplitType<IdxType> * out_start)

{
    int i, t, a, hist[MAX_LEVEL + 1] = {};

    for (i = 0; i < cnt; i++) hist[in_start[i].level]++;

    for (a = -1, i = 0; i < MAX_LEVEL + 1; i++)
    {
        t       = hist[i] + a;
        hist[i] = a;
        a       = t;
    }

    for (i = 0; i < cnt; i++) out_start[++hist[in_start[i].level]] = in_start[i];
}

template <typename IdxType>
inline void sort_morton_codes(CodeType<IdxType> * mc, IdxType cnt, CodeType<IdxType> * tb, int * hist)
{
    uint64_t t;
    IdxType i;

    int * b1 = hist + 0 * 1024;
    int * b2 = hist + 2 * 1024;
    int * b3 = hist + 4 * 1024;
    int * b4 = hist + 5 * 1024;
    int * b5 = hist + 7 * 1024;
    int * b6 = hist + 9 * 1024;

    int tmp, a1, a2, a3, a4, a5, a6;

    //daal::services::internal::service_memset<int, cpu>(hist, 0, 10 * 1024);
    //memset(hist, 0, 10 * 1024 * sizeof(int));

    for (i = 0; i < cnt; i++)
    {
        t = mc[i].morton;
        b1[(t >> 00) & 0x7FF]++;
        b2[(t >> 11) & 0x7FF]++;
        b3[(t >> 22) & 0x3FF]++;
        b4[(t >> 32) & 0x7FF]++;
        b5[(t >> 43) & 0x7FF]++;
        b6[(t >> 54)]++;
    }

    a1 = a2 = a3 = a4 = a5 = a6 = -1;
    for (i = 0; i < 1024; i++)
    {
        tmp   = b1[i] + a1;
        b1[i] = a1;
        a1    = tmp;
        tmp   = b2[i] + a2;
        b2[i] = a2;
        a2    = tmp;
        tmp   = b3[i] + a3;
        b3[i] = a3;
        a3    = tmp;
        tmp   = b4[i] + a4;
        b4[i] = a4;
        a4    = tmp;
        tmp   = b5[i] + a5;
        b5[i] = a5;
        a5    = tmp;
        tmp   = b6[i] + a6;
        b6[i] = a6;
        a6    = tmp;
    }
    for (i = 1024; i < 2048; i++)
    {
        tmp   = b1[i] + a1;
        b1[i] = a1;
        a1    = tmp;
        tmp   = b2[i] + a2;
        b2[i] = a2;
        a2    = tmp;
        tmp   = b4[i] + a4;
        b4[i] = a4;
        a4    = tmp;
        tmp   = b5[i] + a5;
        b5[i] = a5;
        a5    = tmp;
    }

    for (i = 0; i < cnt; i++)
    {
        t                           = mc[i].morton;
        tb[++b1[(t >> 00) & 0x7FF]] = mc[i];
    }
    for (i = 0; i < cnt; i++)
    {
        t                           = tb[i].morton;
        mc[++b2[(t >> 11) & 0x7FF]] = tb[i];
    }
    for (i = 0; i < cnt; i++)
    {
        t                           = mc[i].morton;
        tb[++b3[(t >> 22) & 0x3FF]] = mc[i];
    }
    for (i = 0; i < cnt; i++)
    {
        t                           = tb[i].morton;
        mc[++b4[(t >> 32) & 0x7FF]] = tb[i];
    }
    for (i = 0; i < cnt; i++)
    {
        t                           = mc[i].morton;
        tb[++b5[(t >> 43) & 0x7FF]] = mc[i];
    }
    for (i = 0; i < cnt; i++)
    {
        t                   = tb[i].morton;
        mc[++b6[(t >> 54)]] = tb[i];
    }
}

template <typename IdxType, typename DataType, daal::CpuType cpu>
services::Status qTreeBuildingKernelImpl(IdxType * sort, IdxType * child, const DataType * posx, const DataType * posy, IdxType * duplicates,
                                         const IdxType nNodes, const IdxType N, IdxType & maxDepth, IdxType & bottom, const DataType & radius,
                                         const IdxType & blockOfRows)
{
    //initialize array
    services::internal::service_memset<IdxType, cpu>(child, -1, (nNodes + 1) * 4);
    services::internal::service_memset<IdxType, cpu>(duplicates, 1, N);
    bottom = nNodes;

    // cache root data
    // const DataType rootx = posx[nNodes];
    // const DataType rooty = posy[nNodes];

    const DataType rootx = posx[nNodes] - radius;
    const DataType rooty = posy[nNodes] - radius;

    TlsMax<IdxType, cpu> maxTlsDepth(1);
    //std::cout << "debug 1" << std::endl;

    // cast all float point X and Y to morton code (Z order)
    IdxType nThreads    = threader_get_threads_number();
    IdxType sizeOfBlock = services::internal::min<cpu, IdxType>(blockOfRows, N / nThreads + 1);
    IdxType nBlocks     = N / sizeOfBlock + !!(N % sizeOfBlock);

    CodeType<IdxType> * morton_code   = services::internal::service_scalable_calloc<CodeType<IdxType>, cpu>(N);
    CodeType<IdxType> * t_morton_code = services::internal::service_scalable_calloc<CodeType<IdxType>, cpu>(N);
    int * t_hist                      = services::internal::service_scalable_calloc<int, cpu>(5 * 2048);

    daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        const IdxType iStart = iBlock * sizeOfBlock;
        const IdxType iEnd   = services::internal::min<cpu, IdxType>(N, iStart + sizeOfBlock);

        uint64_t x, y;

        double scale = 2147483648.0 / radius;

        // iterate over all bodies assigned to thread
        for (IdxType i = iStart; i < iEnd; i++)
        {
            // x = (uint64_t)services::internal::min<cpu, DataType>(
            //     services::internal::max<cpu, DataType>(((posx[i] - (rootx - radius)) / (2.0d * radius)) * 4294967296.0d, 0.0d), 4294967295.0d);
            // y = (uint64_t)services::internal::min<cpu, DataType>(
            //     services::internal::max<cpu, DataType>(((posy[i] - (rooty - radius)) / (2.0d * radius)) * 4294967296.0d, 0.0d), 4294967295.0d);

            x = (uint64_t)services::internal::min<cpu, DataType>(services::internal::max<cpu, DataType>(((posx[i] - rootx) * scale), 0.0d),
                                                                 4294967295.0d);
            y = (uint64_t)services::internal::min<cpu, DataType>(services::internal::max<cpu, DataType>(((posy[i] - rooty) * scale), 0.0d),
                                                                 4294967295.0d);

            x &=
                0x00000000ffffffff; // x = -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,-  31,30,29,28  27,26,25,24 23,22,21,20  19,18,17,16  15,14,13,12  11,10,9,8  7,6,5,4  3,2,1,0
            x = (x ^ (x << 16))
                & 0x0000ffff0000ffff; // x = -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,-  31,30,29,28  27,26,25,24  23,22,21,20  19,18,17,16  -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,- 15,14,13,12  11,10,9,8  7,6,5,4  3,2,1,0
            x = (x ^ (x << 8))
                & 0x00ff00ff00ff00ff; // x = -,-,-,-  -,-,-,-  31,30,29,28  27,26,25,24  -,-,-,-  -,-,-,-  23,22,21,20  19,18,17,16  -,-,-,-  -,-,-,-  15,14,13,12  11,10,9,8  -,-,-,-  -,-,-,- 7,6,5,4  3,2,1,0
            x = (x ^ (x << 4))
                & 0x0f0f0f0f0f0f0f0f; // x = -,-,-,-  31,30,29,28  -,-,-,-  27,26,25,24  -,-,-,- 23,22,21,20  -,-,-,-  19,18,17,16  -,-,-,-  15,14,13,12  -,-,-,-  11,10,9,8  -,-,-,-  7,6,5,4  -,-,-,-  3,2,1,0
            x = (x ^ (x << 2))
                & 0x3333333333333333; // x = -,-,31,30 -,-,29,28  -,-,27,26 -,-,25,24   -,-,23,22  -,-,21,20  -,-,19,18  -,-,17,16  -,-,15,14  -,-,13,12  -,-,11,10  -,-,9,8  -,-,7,6  -,-,5,4  -,-,3,2  -,-,1,0
            x = (x ^ (x << 1))
                & 0x5555555555555555; // x = -,31,-,30 -,29,-,28  -,27,-,26 -,25,-,24   -,23,-,22  -,21,-,20  -,19,-,18  -,17,-,16  -,15,-,14  -,13,-,12  -,11,-,10  -,9,-,8  -,7,-,6  -,5,-,4  -,3,-,2  -,1,-,0

            y &=
                0x00000000ffffffff; // y = -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,-  31,30,29,28  27,26,25,24 23,22,21,20  19,18,17,16  15,14,13,12  11,10,9,8  7,6,5,4  3,2,1,0
            y = (y ^ (y << 16))
                & 0x0000ffff0000ffff; // y = -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,-  31,30,29,28  27,26,25,24  23,22,21,20  19,18,17,16  -,-,-,-  -,-,-,-  -,-,-,-  -,-,-,- 15,14,13,12  11,10,9,8  7,6,5,4  3,2,1,0
            y = (y ^ (y << 8))
                & 0x00ff00ff00ff00ff; // y = -,-,-,-  -,-,-,-  31,30,29,28  27,26,25,24  -,-,-,-  -,-,-,-  23,22,21,20  19,18,17,16  -,-,-,-  -,-,-,-  15,14,13,12  11,10,9,8  -,-,-,-  -,-,-,- 7,6,5,4  3,2,1,0
            y = (y ^ (y << 4))
                & 0x0f0f0f0f0f0f0f0f; // y = -,-,-,-  31,30,29,28  -,-,-,-  27,26,25,24  -,-,-,- 23,22,21,20  -,-,-,-  19,18,17,16  -,-,-,-  15,14,13,12  -,-,-,-  11,10,9,8  -,-,-,-  7,6,5,4  -,-,-,-  3,2,1,0
            y = (y ^ (y << 2))
                & 0x3333333333333333; // y = -,-,31,30 -,-,29,28  -,-,27,26 -,-,25,24   -,-,23,22  -,-,21,20  -,-,19,18  -,-,17,16  -,-,15,14  -,-,13,12  -,-,11,10  -,-,9,8  -,-,7,6  -,-,5,4  -,-,3,2  -,-,1,0
            y = (y ^ (y << 1))
                & 0x5555555555555555; // y = -,31,-,30 -,29,-,28  -,27,-,26 -,25,-,24   -,23,-,22  -,21,-,20  -,19,-,18  -,17,-,16  -,15,-,14  -,13,-,12  -,11,-,10  -,9,-,8  -,7,-,6  -,5,-,4  -,3,-,2  -,1,-,0

            morton_code[i].morton = (x | (y << 1)); // merged x and y to one uint64
            morton_code[i].index  = i;              // index of x, y point
        }
    });

    /************************* Sorting ***********************/

    // sort all morton codes in binary representation

    // struct
    // {
    //     bool operator()(CodeType<IdxType> c1, CodeType<IdxType> c2) const { return c1.morton < c2.morton; }
    // } customLess;
    // std::sort(morton_code, morton_code + N, customLess);

    //std::cout << "debug 2" << std::endl;
    sort_morton_codes(morton_code, N, t_morton_code, t_hist);

    // for (int i =0 ; i < N; i++)
    // {
    //     std::cout << "i = " << i << std::endl;
    //     std::cout << "morton_code[ " << i << "].morton" << morton_code[i].morton << std::endl;
    //     std::cout << "morton_code[ " << i << "].index" << morton_code[i].index << std::endl;

    // }

    // copy sorted indices to sort array. Indices in right to left order in each part of bounding box
    //std::cout << "debug 3" << std::endl;
    daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        const IdxType iStart = iBlock * sizeOfBlock;
        const IdxType iEnd   = services::internal::min<cpu, IdxType>(N, iStart + sizeOfBlock);

        // iterate over all bodies assigned to thread
        for (IdxType i = iStart; i < iEnd; i++)
        {
            sort[i] = morton_code[i].index;
        }
    });

    // filling split list structure. see on two bits for each level and search place, when result of XOR will be 01 or 10.
    // It means, that next point placed in the next level of bounding box
    SplitType<IdxType> * split_list = services::internal::service_scalable_malloc<SplitType<IdxType>, cpu>(2 * (N - 1));

    //std::cout << "debug 4" << std::endl;
    nThreads = threader_get_threads_number();

    sizeOfBlock = services::internal::min<cpu, IdxType>(256, (N - 1) / nThreads + 1);

    nBlocks = (N - 1) / sizeOfBlock + !!((N - 1) % sizeOfBlock);
    //std::cout << "debug 41" << std::endl;
    // std::cout << "N = " << N << std::endl;
    // std::cout << "sizeOfBlock = " << sizeOfBlock << std::endl;
    // std::cout << "nBlocks = " << nBlocks << std::endl;
    // std::cout << "nThreads = " << nThreads << std::endl;
    daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        const IdxType iStart = iBlock * sizeOfBlock;
        const IdxType iEnd   = services::internal::min<cpu, IdxType>((N - 1), iStart + sizeOfBlock);
        IdxType lev;
        IdxType * localmaxDepth = maxTlsDepth.local();
        for (IdxType i = iStart; i < iEnd; i++)
        {
            //std::cout << tid << std::endl;
            lev = 0;
            while (lev < 32)
            {
                if (((morton_code[i].morton ^ morton_code[i + 1].morton) & (0xC000000000000000 >> (lev * 2))) != 0)
                    break;
                else
                    lev++;
            }
            split_list[i].level   = lev;
            split_list[i].m_index = i;

            if (split_list[i].level > localmaxDepth[0])
            {
                localmaxDepth[0] = split_list[i].level;
            }
        }
    });
    //std::cout << "debug 42" << std::endl;
    SplitType<IdxType> * split_list_seq = services::internal::service_scalable_calloc<SplitType<IdxType>, cpu>(SEQ_SPLIT_LIST_SIZE);

    IdxType NSEQ = 0;

    IdxType block_size = (N / 1000) ? (N / 1000) : N;

    IdxType num_blocks = (N + block_size - 1) / block_size;

    IdxType * nseq = services::internal::service_scalable_calloc<IdxType, cpu>(num_blocks); // number of points with lev <= 4 in each block

    sizeOfBlock = 1;
    nBlocks     = num_blocks;

    //std::cout << "debug 5" << std::endl;
    daal::threader_for(nBlocks, nBlocks, [&](IdxType iBlock) {
        //daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        const IdxType b = iBlock * sizeOfBlock;
        IdxType lo      = b * block_size;                                                     // lower border of block in indices
        IdxType hi      = services::internal::min<cpu, IdxType>((b + 1) * block_size, N - 1); // upper border of block
        for (IdxType j = lo; j < hi; j++)
        {
            if (split_list[j].level <= SEQ_UPTO_LEVEL) nseq[b]++;
        }
    });
    IdxType nseq_start = 0;
    for (IdxType b = 0; b < num_blocks; b++)
    {
        IdxType sum = nseq_start + nseq[b];
        nseq[b]     = nseq_start;
        nseq_start  = sum;
    }

    //std::cout << "debug 6" << std::endl;
    daal::threader_for(nBlocks, nBlocks, [&](IdxType iBlock) {
        //daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        const IdxType b = iBlock * sizeOfBlock;
        IdxType lo      = b * block_size;
        IdxType hi      = services::internal::min<cpu, IdxType>((b + 1) * block_size, N - 1);
        for (IdxType j = lo; j < hi; j++)
        {
            if (split_list[j].level <= SEQ_UPTO_LEVEL)
            {
                split_list_seq[nseq[b]] = split_list[j];
                nseq[b]++;
            }
        }
    });
    NSEQ = nseq[num_blocks - 1];
    //std::cout << "debug 7" << std::endl;

    // for (int i =0 ; i < NSEQ; i++)
    // {
    //     std::cout << "i = " << i << std::endl;
    //     std::cout << "split_list_seq.level = " << split_list_seq[i].level << std::endl;
    //     std::cout << "split_list_seq.m_index = " << split_list_seq[i].m_index << std::endl;

    // }

    // for (int i =0 ; i < N-1; i++)
    // {
    //     std::cout << "i = " << i << std::endl;
    //     std::cout << "split_list.level = " << split_list[i].level << std::endl;
    //     std::cout << "split_list.m_index = " << split_list[i].m_index << std::endl;
    // }

    // Sort the new split list and create tree using it.

    // struct
    // {
    //     bool operator()(SplitType<IdxType> s1, SplitType<IdxType> s2) const { return s1.level < s2.level; }
    // } customLess2;
    // std::stable_sort(split_list_seq, split_list_seq + NSEQ, customLess2);

    //std::cout << "debug 7" << std::endl;
    sort_splits(split_list_seq, NSEQ, split_list + N - 1);
    //std::cout << "debug 8" << std::endl;

    IdxType level_size[MAX_LEVEL + 1];
    for (IdxType i = 0; i <= MAX_LEVEL; i++) level_size[i] = 0; // memset
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(IdxType, 2, NSEQ);
    IdxType tree_allocation      = services::internal::max<cpu, IdxType>(64, 2 * NSEQ);
    TreeNode<IdxType> * tree_seq = services::internal::service_scalable_malloc<TreeNode<IdxType>, cpu>(tree_allocation);
    //TreeNode<IdxType> *tree_seq = services::internal::service_scalable_calloc<TreeNode<IdxType>, cpu>(tree_allocation);
    TreeNode<IdxType> t_node;
    t_node.parent = -1; // root node
    t_node.first  = 0;
    t_node.second = N - 1;
    tree_seq[0]   = t_node;
    for (int i = 0; i < 4; i++)
    {
        tree_seq[0].child[i]          = -1;
        tree_seq[0].child_internal[i] = 0;
    }
    //std::cout << "debug 9" << std::endl;
    //IdxType tree_seq_size = build_tree<IdxType, cpu>(morton_code, split_list_seq, NSEQ, tree_allocation, level_size, 0, tree_seq);
    IdxType tree_seq_size = build_tree<IdxType, cpu>(morton_code, split_list + N - 1, NSEQ, tree_allocation, level_size, 0, tree_seq);

    //std::cout << "debug 10" << std::endl;
    IdxType i = 0;
    // for (IdxType lev = 0; lev <= SEQ_UPTO_LEVEL; lev++)
    // {
    //     for (IdxType k = 0; k < level_size[lev]; k++)
    //     {
    //         TreeNode<IdxType> tree_node = tree_seq[i];
    //         tree_seq[i].pos             = bottom;
    //         for (IdxType j = 0; j < 4; j++)
    //         {
    //             if (tree_node.child_internal[j] > 0)
    //             {
    //                 child[bottom * 4 + j] = nNodes - tree_node.child[j];
    //             }
    //             else
    //             {
    //                 child[bottom * 4 + j] = tree_node.child[j];
    //             }
    //         }
    //         bottom--;
    //         i++;
    //     }
    // }

    IdxType lev;
    for (lev = 0; lev <= SEQ_UPTO_LEVEL; lev++)
    {
        for (IdxType k = 0; k < level_size[lev]; k++)
        {
            TreeNode<IdxType> tree_node = tree_seq[i];

            bool isTerminalST = (tree_seq[i].second - tree_seq[i].first) > 0;
            tree_seq[i].pos   = bottom;

            // std::cout << "i = " << i << std::endl;
            // std::cout << "tree_node.first = " << tree_node.first << std::endl;
            // std::cout << "tree_node.second = " << tree_node.second << std::endl;
            // std::cout << "tree_node.parent = " << tree_node.parent << std::endl;
            // std::cout << "tree_node.pos = " << tree_node.pos << std::endl;
            // for (int itt = 0; itt < 4 ; itt++) std::cout << "tree_node.child[" << itt << "] = " << tree_node.child[itt] << std::endl;
            // for (int itt = 0; itt < 4 ; itt++) std::cout << "tree_node.child_internal[" << itt << "] = " << tree_node.child_internal[itt] << std::endl;
            // std::cout << std::endl;

            for (IdxType j = 0; j < 4; j++)
            {
                if (tree_node.child_internal[j] > 0)
                {
                    child[bottom * 4 + j] = nNodes - tree_node.child[j];
                    isTerminalST          = false;
                }
                else
                {
                    child[bottom * 4 + j] = tree_node.child[j];
                    if (child[bottom * 4 + j] > 0) isTerminalST = false;
                }
            }
            if (isTerminalST) break;
            bottom--;
            i++;
        }
        if (level_size[lev + 1] == 0) break;
    }

    IdxType par_tree_start = i;
    //std::cout << "par_tree_start = " << par_tree_start << std::endl;
    IdxType num_subtrees = tree_seq_size - par_tree_start;
    //std::cout << "tree_seq_size = " << tree_seq_size << std::endl;
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(IdxType, num_subtrees, (MAX_LEVEL + 1));
    IdxType * subtree_level_size = services::internal::service_scalable_calloc<IdxType, cpu>(num_subtrees * (MAX_LEVEL + 1));

    TreeNode<IdxType> * tree_par[num_subtrees];
    IdxType tree_par_allocation[num_subtrees];

    sizeOfBlock = 1;
    nBlocks     = num_subtrees;

    // std::cout << "debug 11" << std::endl;
    // std::cout << "nBlocks = " << nBlocks << std::endl;
    // std::cout << "tree_seq size = " <<  tree_allocation << std::endl;

    daal::threader_for(nBlocks, nBlocks, [&](IdxType iBlock) {
        //daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        //std::cout << "debug 11_0" << std::endl;
        const IdxType subtree_id        = iBlock * sizeOfBlock;
        TreeNode<IdxType> t_node        = tree_seq[par_tree_start + subtree_id];
        IdxType first                   = t_node.first;
        IdxType second                  = t_node.second;
        tree_par_allocation[subtree_id] = services::internal::max<cpu, IdxType>(128, (second - first));
        tree_par[subtree_id]            = services::internal::service_scalable_malloc<TreeNode<IdxType>, cpu>(tree_par_allocation[subtree_id]);
        //tree_par[subtree_id] = services::internal::service_scalable_calloc<TreeNode<IdxType>, cpu>(tree_par_allocation[subtree_id]);

        // Sort the subtree split list and create tree using it.
        //std::stable_sort(split_list + first, split_list + second, customLess2);
        // std::cout << "debug 11_1" << std::endl;
        // std::cout << "first = " << first << std::endl;
        // std::cout << "first + N - 1 = " << first + N - 1 << std::endl;
        // std::cout << "second - first = " << second - first << std::endl;
        sort_splits(split_list + first, second - first, split_list + first + N - 1);
        //std::cout << "debug 11_2" << std::endl;
        tree_par[subtree_id][0] = t_node;
        // for (int i = 0; i < 4; i++)
        // {
        //     tree_par[subtree_id][0].child[i] = t_node.child[i];
        //     tree_par[subtree_id][0].child_internal[i] = t_node.child_internal[i];
        // }
        // IdxType subtree_size    = build_tree<IdxType, cpu>(morton_code, split_list + first, second - first, tree_par_allocation[subtree_id],
        //                                                 subtree_level_size + subtree_id * (MAX_LEVEL + 1), SEQ_UPTO_LEVEL + 1, tree_par[subtree_id]);
        //std::cout << "debug 11_3" << std::endl;
        IdxType subtree_size = build_tree<IdxType, cpu>(morton_code, split_list + first + N - 1, second - first, tree_par_allocation[subtree_id],
                                                        subtree_level_size + subtree_id * (MAX_LEVEL + 1), SEQ_UPTO_LEVEL + 1, tree_par[subtree_id]);
        //std::cout << "debug 11_4" << std::endl;
    });

    //std::cout << "debug 12" << std::endl;
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(IdxType, num_subtrees, (MAX_LEVEL + 1));
    IdxType * subtree_level_start = services::internal::service_scalable_calloc<IdxType, cpu>(num_subtrees * (MAX_LEVEL + 1));

    // Can also be parallelized but has only ~20K inner loop iterations
    IdxType start = 0;
    for (IdxType lev = 0; lev < MAX_LEVEL; lev++)
    {
        for (IdxType subtree_id = 0; subtree_id < num_subtrees; subtree_id++)
        {
            subtree_level_start[subtree_id * (MAX_LEVEL + 1) + lev] = start;
            start                                                   = start + subtree_level_size[subtree_id * (MAX_LEVEL + 1) + lev];
        }
    }
    IdxType * min_pos = services::internal::service_scalable_malloc<IdxType, cpu>(num_subtrees);
    //IdxType *min_pos = services::internal::service_scalable_calloc<IdxType, cpu>(num_subtrees);
    for (int sm = 0; sm < num_subtrees; sm++)
    {
        min_pos[sm] = bottom;
    }

    sizeOfBlock = 1;
    nBlocks     = num_subtrees;

    //std::cout << "debug 13" << std::endl;
    daal::threader_for(nBlocks, nBlocks, [&](IdxType iBlock) {
        //daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        const IdxType subtree_id = iBlock * sizeOfBlock;
        IdxType subtree_i        = 0;
        for (IdxType lev = 0; lev < MAX_LEVEL; lev++)
        {
            IdxType lev_start = subtree_level_start[subtree_id * (MAX_LEVEL + 1) + lev];
            IdxType lev_size  = subtree_level_size[subtree_id * (MAX_LEVEL + 1) + lev];

            for (IdxType k = 0; k < lev_size; k++)
            {
                IdxType pos                         = bottom - lev_start - k;
                tree_par[subtree_id][subtree_i].pos = pos;
                min_pos[subtree_id]                 = pos;
                subtree_i++;
            }
        }
    });

    //std::cout << "debug 14" << std::endl;
    for (int sm = 0; sm < num_subtrees; sm++)
    {
        if (bottom > min_pos[sm]) bottom = min_pos[sm];
    }

    sizeOfBlock = 1;
    nBlocks     = num_subtrees;
    //std::cout << "debug 15" << std::endl;
    daal::threader_for(nBlocks, nBlocks, [&](IdxType iBlock) {
        //daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        const IdxType subtree_id = iBlock * sizeOfBlock;
        IdxType subtree_i        = 0;
        bool flag_tmp            = false;
        for (IdxType lev = 0; lev < MAX_LEVEL; lev++)
        {
            IdxType lev_start = subtree_level_start[subtree_id * (MAX_LEVEL + 1) + lev];
            IdxType lev_size  = subtree_level_size[subtree_id * (MAX_LEVEL + 1) + lev];
            for (IdxType k = 0; k < lev_size; k++)
            {
                TreeNode<IdxType> tree_node = tree_par[subtree_id][subtree_i];
                for (IdxType j = 0; j < 4; j++)
                {
                    if (tree_node.child_internal[j] > 0)
                    {
                        if (tree_node.child[j] <= subtree_i || tree_node.child[j] >= tree_par_allocation[subtree_id])
                        {
                            //fprintf(stderr, "subtree_id = %d, subtree_i = %d, j = %d, child_j = %d, child_internal = %d, allocation = %d\n",
                            //    subtree_id, subtree_i, j, tree_node.child[j], tree_node.child_internal[j], tree_par_allocation[subtree_id]);

                            exit(0);
                        }
                        if (lev == (MAX_LEVEL - 1))
                        {
                            child[tree_node.pos * 4 + j] = tree_par[subtree_id][tree_node.child[j]].first;
                        }
                        else
                        {
                            IdxType first  = tree_par[subtree_id][tree_node.child[j]].first;
                            IdxType second = tree_par[subtree_id][tree_node.child[j]].second;

                            if (second != first && morton_code[first].morton == morton_code[second].morton)
                            {
                                flag_tmp                     = true;
                                child[tree_node.pos * 4 + j] = morton_code[first].index;
                                duplicates[morton_code[first].index] += second - first;
                                break;
                            }
                            else
                            {
                                child[tree_node.pos * 4 + j] = tree_par[subtree_id][tree_node.child[j]].pos;
                            }

                            //child[tree_node.pos * 4 + j] = tree_par[subtree_id][tree_node.child[j]].pos;
                        }
                    }
                    else
                    {
                        child[tree_node.pos * 4 + j] = tree_node.child[j];
                    }
                }
                if (flag_tmp) break;
                subtree_i++;
            }
            if (flag_tmp) break;
        }
    });
    //std::cout << "debug 16" << std::endl;
    services::internal::service_scalable_free<TreeNode<IdxType>, cpu>(tree_seq);
    services::internal::service_scalable_free<IdxType, cpu>(subtree_level_size);
    services::internal::service_scalable_free<IdxType, cpu>(subtree_level_start);
    services::internal::service_scalable_free<SplitType<IdxType>, cpu>(split_list_seq);
    services::internal::service_scalable_free<IdxType, cpu>(min_pos);

    sizeOfBlock = 1;
    nBlocks     = num_subtrees;

    daal::threader_for(nBlocks, nBlocks, [&](IdxType iBlock) {
        //daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        const IdxType subtree_id = iBlock * sizeOfBlock;
        services::internal::service_scalable_free<TreeNode<IdxType>, cpu>(tree_par[subtree_id]);
    });
    // free memory
    services::internal::service_scalable_free<int, cpu>(t_hist);
    services::internal::service_scalable_free<CodeType<IdxType>, cpu>(t_morton_code);

    services::internal::service_scalable_free<CodeType<IdxType>, cpu>(morton_code);
    services::internal::service_scalable_free<SplitType<IdxType>, cpu>(split_list);
    services::internal::service_scalable_free<IdxType, cpu>(nseq);

    maxTlsDepth.reduceTo(&maxDepth, 1);
    return services::Status();
}

// template <typename IdxType, typename DataType, daal::CpuType cpu>
// services::Status summarizationKernelImpl(IdxType * count, IdxType * child, DataType * mass, DataType * posX, DataType * posY, IdxType * duplicates,
//                                          const IdxType nNodes, const IdxType N, const IdxType & bottom)
// {
//     bool flag = false;
//     DataType cm, px, py;
//     IdxType curChild[4];
//     DataType curMass[4];

//     const IdxType inc = 1;
//     auto k            = bottom;
//     //std::cout << "summarizationKernelImpl" << std::endl;
//     //std::cout << "***************************************************************\n******************************************************************" << std::endl;
//     //std::cout << "bottom = " << bottom << std::endl;

//     // for (int i = 4*bottom; i < (nNodes+1)*4; i++)
//     // {
//     //     std::cout << "child[" << i << "] = " << child[i] << std::endl;
//     // }

//     //initialize array
//     services::internal::service_memset<DataType, cpu>(mass, DataType(1), k);
//     services::internal::service_memset<DataType, cpu>(&mass[k], DataType(-1), nNodes - k + 1);

//     const auto restart = k;
//     // iterate over all cells assigned to thread
//     while (k <= nNodes)
//     {
//         if (mass[k] < 0.)
//         {
//             for (IdxType i = 0; i < 4; i++)
//             {
//                 const auto ch = child[k * 4 + i];
//                 curChild[i]   = ch;
//                 if (ch >= 0) curMass[i] = mass[ch];
//             }

//             // all children are ready
//             cm       = 0.;
//             px       = 0.;
//             py       = 0.;
//             auto cnt = 0;

//             for (IdxType i = 0; i < 4; i++)
//             {
//                 const IdxType ch = curChild[i];
//                 if (ch >= 0)
//                 {
//                     DataType m = 0;
//                     if (duplicates[ch] > 1)
//                     {
//                         if (ch >= N)
//                         {
//                             cnt += count[ch];
//                             m = curMass[i];
//                         }
//                         else
//                         {
//                             cnt += duplicates[ch];
//                             m = mass[ch] + DataType(duplicates[ch]) - DataType(1);
//                         }
//                     }
//                     else
//                         m = (ch >= N) ? (cnt += count[ch], curMass[i]) : (cnt++, mass[ch]);
//                     // add child's contribution
//                     cm += m;
//                     px += posX[ch] * m;
//                     py += posY[ch] * m;
//                 }
//             }
//             count[k]         = cnt;
//             const DataType m = cm ? DataType(1) / cm : DataType(1);

//             posX[k] = px * m;
//             posY[k] = py * m;
//             mass[k] = cm;
//         }

//         k += inc; // move on to next cell
//     }
//     return services::Status();
// }

template <typename IdxType, typename DataType, daal::CpuType cpu>
services::Status summarizationKernelImpl(IdxType * count, IdxType * child, DataType * mass, DataType * posX, DataType * posY, IdxType * duplicates,
                                         const IdxType nNodes, const IdxType N, const IdxType & bottom)
{
    const auto inc = 1;
    auto k         = bottom;

    // std::cout << "bottom = " << bottom << std::endl;
    // for (int i = 4*bottom; i < (nNodes+1)*4; i++)
    // {
    //     std::cout << "child[" << i << "] = " << child[i] << std::endl;
    // }

    //initialize array
    services::internal::service_memset<DataType, cpu>(mass, DataType(1), k);
    services::internal::service_memset<DataType, cpu>(&mass[k], DataType(-1), nNodes - k + IdxType(1));

    const IdxType nThreads = threader_get_threads_number();
    const IdxType nBlocks  = nNodes - k + IdxType(1);

    daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        const IdxType iStart = k + iBlock;

        IdxType curChild[4];
        DataType curMass[4];
        DataType cm, px, py;

        if (mass[iStart] < DataType(0))
        {
            IdxType j = 0;
            while (j < 4)
            {
                const IdxType ch = child[iStart * 4 + j];
                curChild[j]      = ch;

                curMass[j] = mass[ch];
                //std::cout << "ch = " << ch << std::endl;
                //std::cout << "mass[ch] = " << mass[ch] << std::endl;

                if (ch >= k && curMass[j] < DataType(0))
                {
                    continue;
                }
                j++;
            }

            // all children are ready
            cm          = 0.;
            px          = 0.;
            py          = 0.;
            IdxType cnt = 0;

            for (IdxType i = 0; i < 4; i++)
            {
                const IdxType ch = curChild[i];

                if (ch >= 0)
                {
                    DataType m = 0;
                    if (duplicates[ch] > 1)
                    {
                        if (ch >= N)
                        {
                            cnt += count[ch];
                            m = curMass[i];
                        }
                        else
                        {
                            cnt += duplicates[ch];
                            m = mass[ch] + DataType(duplicates[ch]) - DataType(1);
                        }
                    }
                    else
                        m = (ch >= N) ? (cnt += count[ch], curMass[i]) : (cnt++, mass[ch]);

                    // add child's contribution
                    cm += m;
                    px += posX[ch] * m;
                    py += posY[ch] * m;
                }
            }
            count[iStart] = cnt;

            const DataType m = cm ? DataType(1) / cm : DataType(1);

            posX[iStart] = px * m;
            posY[iStart] = py * m;

            mass[iStart] = cm;
        }
    });
    return services::Status();
}

template <typename IdxType, daal::CpuType cpu>
services::Status sortKernelImpl(IdxType * sort, const IdxType * count, IdxType * start, IdxType * child, const IdxType nNodes, const IdxType N,
                                const IdxType & bottom)
{
    const IdxType dec = 1;

    IdxType nThreads    = threader_get_threads_number();
    IdxType sizeOfBlock = services::internal::min<cpu, IdxType>(256, (nNodes - bottom + 1) / nThreads + 1);
    IdxType nBlocks     = (nNodes - bottom + 1) / sizeOfBlock + !!((nNodes - bottom + 1) % sizeOfBlock);

    daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        const IdxType iStart = bottom + iBlock * sizeOfBlock;
        const IdxType iEnd   = services::internal::min<cpu, IdxType>((nNodes + 1), iStart + sizeOfBlock);
        for (IdxType k = iStart; k < iEnd; k++)
        {
            IdxType j = 0;
            for (IdxType i = 0; i < 4; i++)
            {
                const auto ch = child[k * 4 + i];
                if (ch >= 0)
                {
                    if (i != j)
                    {
                        // move children to front (needed later for speed)
                        child[k * 4 + i] = -1;
                        child[k * 4 + j] = ch;
                    }
                    j++;
                }
            }
        }
    });

    return services::Status();
}

template <typename IdxType, typename DataType, daal::CpuType cpu>
services::Status repulsionKernelImpl(const DataType theta, const DataType eps, const IdxType * sort, const IdxType * child, const DataType * mass,
                                     const DataType * posX, const DataType * posY, DataType * repX, DataType * repY, DataType & zNorm,
                                     const IdxType nNodes, const IdxType N, const DataType & radius, const IdxType & maxDepth,
                                     const IdxType & blockOfRows)
{
    SafeStatus safeStat;

    //struct for tls
    struct RepulsionTask
    {
    public:
        DAAL_NEW_DELETE();
        DataType * sumData;
        IdxType * posData;
        IdxType * nodeData;

        static RepulsionTask * create(const IdxType maxDepth)
        {
            auto object = new RepulsionTask(maxDepth);
            if (object && object->isValid()) return object;
            delete object;
            return nullptr;
        }

        bool isValid() const { return _sum.get() && _pos.get() && _node.get(); }

    private:
        RepulsionTask(IdxType maxDepth)
        {
            _sum.reset(1);
            sumData = _sum.get();
            services::internal::service_memset_seq<DataType, cpu>(sumData, DataType(0), 1);

            _pos.reset(maxDepth);
            posData = _pos.get();
            services::internal::service_memset_seq<IdxType, cpu>(posData, IdxType(0), maxDepth);

            _node.reset(maxDepth);
            nodeData = _node.get();
            services::internal::service_memset_seq<IdxType, cpu>(nodeData, IdxType(0), maxDepth);
        }

        TArrayScalable<DataType, cpu> _sum;
        TArrayScalable<IdxType, cpu> _pos;
        TArrayScalable<IdxType, cpu> _node;
    };

    //initialize arrays
    services::internal::service_memset<DataType, cpu>(repX, DataType(0), nNodes + 1);
    services::internal::service_memset<DataType, cpu>(repY, DataType(0), nNodes + 1);
    zNorm = DataType(0);

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(IdxType, nNodes, 4);
    const IdxType fourNNodes     = 4 * nNodes;
    const DataType thetaSquared  = theta * theta;
    const DataType radiusSquared = radius * radius;
    const DataType epsInc        = eps + DataType(1);
    TArrayCalloc<DataType, cpu> dqArray(maxDepth);
    DAAL_CHECK_MALLOC(dqArray.get());
    DataType * dq = dqArray.get();

    daal::static_tls<RepulsionTask *> tlsTask([=, &safeStat]() {
        auto tlsData = RepulsionTask::create(maxDepth);
        if (!tlsData)
        {
            safeStat.add(services::ErrorMemoryAllocationFailed);
        }
        return tlsData;
    });

    const IdxType nThreads    = threader_get_threads_number();
    const IdxType sizeOfBlock = services::internal::min<cpu, IdxType>(blockOfRows, N / nThreads + 1);
    const IdxType nBlocks     = N / sizeOfBlock + !!(N % sizeOfBlock);

    dq[0] = radiusSquared / thetaSquared;
    for (auto i = 1; i < maxDepth; i++)
    {
        dq[i] = dq[i - 1] * DataType(0.25);
        dq[i - 1] += eps;
    }
    dq[maxDepth - 1] += eps;

    // Add one so epsInc can be compared
    for (auto i = 0; i < maxDepth; i++) dq[i] += 1.;

    // iterate over all bodies assigned to thread
    DAAL_OVERFLOW_CHECK_BY_ADDING(IdxType, fourNNodes, 4);
    const auto MAX_SIZE = fourNNodes + 4;

    daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        const IdxType iStart      = iBlock * sizeOfBlock;
        const IdxType iEnd        = services::internal::min<cpu, IdxType>(N, iStart + sizeOfBlock);
        const RepulsionTask * tls = tlsTask.local(tid);
        DAAL_CHECK_MALLOC_THR(tls);

        IdxType * pos       = tls->posData;
        IdxType * node      = tls->nodeData;
        DataType * localSum = tls->sumData;
        for (IdxType k = iStart; k < iEnd; ++k)
        {
            const auto i = sort[k];

            const DataType px = posX[i];
            const DataType py = posY[i];

            DataType vx = 0.;
            DataType vy = 0.;

            // initialize iteration stack, i.e., push root node onto stack
            IdxType depth = 0;
            pos[0]        = 0;
            node[0]       = fourNNodes;

            do
            {
                // stack is not empty
                auto pd = pos[depth];
                auto nd = node[depth];

                while (pd < 4)
                {
                    const auto index = nd + pd++;
                    if (index < 0 || index >= MAX_SIZE) break;

                    const auto n = child[index]; // load child pointer

                    // Non child
                    if (n < 0 || n > nNodes) break;

                    const DataType dx   = px - posX[n];
                    const DataType dy   = py - posY[n];
                    const DataType dxy1 = dx * dx + dy * dy + epsInc;

                    if ((n < N) || (dxy1 >= dq[depth]))
                    {
                        const DataType tdist_2 = mass[n] / (dxy1 * dxy1);
                        localSum[0] += tdist_2 * dxy1;
                        vx += dx * tdist_2;
                        vy += dy * tdist_2;
                    }
                    else
                    {
                        pos[depth]  = pd;
                        node[depth] = nd;
                        depth++;
                        pd = 0;
                        nd = n * 4;
                    }
                }
            } while (--depth >= 0); // done with this level

            // update velocity
            repX[i] += vx;
            repY[i] += vy;
        }
    });

    tlsTask.reduce([&](RepulsionTask * tls) {
        DataType * sumLocal = tls->sumData;
        zNorm += sumLocal[0];

        delete tls;
    });

    return safeStat.detach();
}

template <bool DivComp, typename IdxType, typename DataType, daal::CpuType cpu>
services::Status attractiveKernelImpl(const DataType * val, const size_t * col, const size_t * row, const DataType * posX, const DataType * posY,
                                      DataType * attrX, DataType * attrY, DataType & zNorm, DataType & divergence, const IdxType nNodes,
                                      const IdxType N, const IdxType nnz, const IdxType nElements, const DataType exaggeration, const DataType eps,
                                      const IdxType & blockOfRows)
{
    //initialize arrays
    services::internal::service_memset<DataType, cpu>(attrX, DataType(0), N);
    services::internal::service_memset<DataType, cpu>(attrY, DataType(0), N);

    const DataType multiplier = exaggeration * DataType(zNorm);
    divergence                = DataType(0);

    daal::StaticTlsSum<DataType, cpu> divTlsData(1);
    daal::static_tls<DataType *> logTlsData([=]() { return services::internal::service_scalable_calloc<DataType, cpu>(nElements); });

    const IdxType nThreads    = IdxType(logTlsData.nthreads());
    const IdxType sizeOfBlock = services::internal::min<cpu, IdxType>(blockOfRows, N / nThreads + 1);
    const IdxType nBlocks     = IdxType(N) / sizeOfBlock + !!(IdxType(N) % sizeOfBlock);

    daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        const IdxType iStart = iBlock * sizeOfBlock;
        const IdxType iEnd   = services::internal::min<cpu, IdxType>(IdxType(N), iStart + sizeOfBlock);
        DataType * logLocal  = logTlsData.local(tid);
        DataType * divLocal  = divTlsData.local(tid);
        for (IdxType iRow = iStart; iRow < iEnd; ++iRow)
        {
            IdxType iSize = 0;
            for (IdxType index = row[iRow] - 1; index < row[iRow + 1] - 1; ++index)
            {
                const IdxType iCol = col[index] - 1;

                const DataType y1d    = posX[iRow] - posX[iCol];
                const DataType y2d    = posY[iRow] - posY[iCol];
                const DataType sqDist = services::internal::max<cpu, DataType>(DataType(0), y1d * y1d + y2d * y2d);
                const DataType PQ     = val[index] / (sqDist + DataType(1));

                // Apply forces
                attrX[iRow] += PQ * (posX[iRow] - posX[iCol]);
                attrY[iRow] += PQ * (posY[iRow] - posY[iCol]);
                if (DivComp)
                {
                    logLocal[iSize++] = val[index] * multiplier * (1. + sqDist);
                }
            }
            if (DivComp)
            {
                Math<DataType, cpu>::vLog(iSize, logLocal, logLocal);
                IdxType start = row[iRow] - 1;
                for (IdxType index = 0; index < iSize; ++index)
                {
                    divLocal[0] += val[start + index] * logLocal[index];
                }
            }
        }
    });
    divTlsData.reduceTo(&divergence, 1);
    divergence *= exaggeration;
    logTlsData.reduce([&](DataType * buf) { services::internal::service_scalable_free<DataType, cpu>(buf); });

    //Find_Normalization
    zNorm = (zNorm - DataType(N)) ? DataType(1) / (zNorm - DataType(N)) : (DataType(1) / eps);

    return services::Status();
}

template <typename IdxType, typename DataType, daal::CpuType cpu>
services::Status integrationKernelImpl(const DataType eta, const DataType momentum, const DataType exaggeration, DataType * posX, DataType * posY,
                                       const DataType * attrX, const DataType * attrY, const DataType * repX, const DataType * repY, DataType * gainX,
                                       DataType * gainY, DataType * oldForceX, DataType * oldForceY, DataType & gradNorm, const DataType & zNorm,
                                       const IdxType nNodes, const IdxType N, const IdxType & blockOfRows)
{
    const IdxType nThreads    = threader_get_threads_number();
    const IdxType sizeOfBlock = services::internal::min<cpu, IdxType>(blockOfRows, N / nThreads + 1);
    const IdxType nBlocks     = N / sizeOfBlock + !!(N % sizeOfBlock);
    daal::StaticTlsSum<DataType, cpu> sumTlsData(1);
    gradNorm = 0.;

    daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        const IdxType iStart = iBlock * sizeOfBlock;
        const IdxType iEnd   = services::internal::min<cpu, IdxType>(N, iStart + sizeOfBlock);
        DataType ux, uy, gx, gy;
        DataType * localSum = sumTlsData.local(tid);
        for (IdxType i = iStart; i < iEnd; ++i)
        {
            const DataType dx = exaggeration * attrX[i] - zNorm * repX[i];
            const DataType dy = exaggeration * attrY[i] - zNorm * repY[i];
            localSum[0] += dx * dx + dy * dy;

            gx = (dx * (ux = oldForceX[i]) < DataType(0)) ? gainX[i] + DataType(0.2) : gainX[i] * DataType(0.8);
            if (gx < DataType(0.01)) gx = DataType(0.01);

            gy = (dy * (uy = oldForceY[i]) < DataType(0)) ? gainY[i] + DataType(0.2) : gainY[i] * DataType(0.8);
            if (gy < DataType(0.01)) gy = DataType(0.01);

            gainX[i] = gx;
            gainY[i] = gy;

            oldForceX[i] = ux = momentum * ux - DataType(4) * eta * gx * dx;
            oldForceY[i] = uy = momentum * uy - DataType(4) * eta * gy * dy;

            posX[i] += ux;
            posY[i] += uy;
        }
    });
    sumTlsData.reduceTo(&gradNorm, 1);
    gradNorm = Math<DataType, cpu>::sSqrt(gradNorm);

    return services::Status();
}

template <typename IdxType, typename DataType, daal::CpuType cpu>
services::Status tsneGradientDescentImpl(const NumericTablePtr initTable, const CSRNumericTablePtr pTable, const NumericTablePtr sizeIterTable,
                                         const NumericTablePtr paramTable, const NumericTablePtr resultTable)
{
    // sizes and number of iterations
    daal::internal::ReadColumns<IdxType, cpu> sizeIterDataBlock(*sizeIterTable, 0, 0, sizeIterTable->getNumberOfRows());
    const IdxType * sizeIter = sizeIterDataBlock.get();
    DAAL_CHECK_BLOCK_STATUS(sizeIterDataBlock);
    DAAL_CHECK(sizeIterTable->getNumberOfRows() == 4, daal::services::ErrorIncorrectSizeOfInputNumericTable);
    const IdxType N                    = sizeIter[0]; // Number of points
    const IdxType nnz                  = sizeIter[1]; // Number of elements in sparce matrix P
    const IdxType nIterWithoutProgress = sizeIter[2]; // Number of iterations without introducing changes
    const IdxType maxIter              = sizeIter[3]; // Number of iterations
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(IdxType, 2, N);
    const IdxType nNodes          = N <= 50 ? 6 * N : 2 * N; // A small number of points may require more memory to store tree nodes
    const IdxType nIterCheck      = 50;
    const IdxType explorationIter = 250; // Aligned with scikit-learn
    const IdxType blockOfRows     = 256;

    // parameters
    daal::internal::ReadColumns<DataType, cpu> paramDataBlock(*paramTable, 0, 0, paramTable->getNumberOfRows());
    const DataType * params = paramDataBlock.get();
    DAAL_CHECK_BLOCK_STATUS(paramDataBlock);
    DAAL_CHECK(paramTable->getNumberOfRows() == 4, daal::services::ErrorIncorrectSizeOfInputNumericTable);
    const DataType eps = 0.000001; // A tiny jitter to promote numerical stability
    DataType momentum  = 0.5;      // The momentum used during the exaggeration phase. Aligned with scikit-learn
    DataType exaggeration =
        params[0]; // How much pressure to apply to clusters to spread out during the exaggeration phase. Aligned with scikit-learn
    const DataType eta         = params[1]; // Learning rate. Aligned with scikit-learn
    const DataType minGradNorm = params[2]; // The smallest gradient norm TSNE should terminate on
    const DataType theta       = params[3]; // is the angular size of a distant node as measured from a point. Tradeoff for speed (0) vs accuracy (1)

    // results
    daal::internal::WriteColumns<DataType, cpu> resultDataBlock(*resultTable, 0, 0, resultTable->getNumberOfRows());
    DataType * results = resultDataBlock.get();
    DAAL_CHECK_BLOCK_STATUS(resultDataBlock);
    DAAL_CHECK(resultTable->getNumberOfRows() == 3, daal::services::ErrorIncorrectSizeOfInputNumericTable);
    DataType & curIter    = results[0];
    DataType & divergence = results[1];
    DataType & gradNorm   = results[2];

    // internal values
    services::Status status;
    IdxType maxDepth        = 1;
    IdxType bottom          = nNodes;
    IdxType nElements       = 0;
    IdxType bestIter        = 0;
    DataType radius         = 0.;
    DataType zNorm          = 0.;
    DataType bestDivergence = daal::services::internal::MaxVal<DataType>::get();

    // daal checks
    DAAL_CHECK(initTable->getNumberOfRows() == N, daal::services::ErrorInconsistentNumberOfRows);
    DAAL_CHECK(initTable->getNumberOfColumns() == 2, daal::services::ErrorInconsistentNumberOfColumns);

    daal::internal::WriteColumns<DataType, cpu> xInitDataBlock(*initTable, 0, 0, N);
    daal::internal::WriteColumns<DataType, cpu> yInitDataBlock(*initTable, 1, 0, N);
    DataType * xInit = xInitDataBlock.get();
    DataType * yInit = yInitDataBlock.get();
    DAAL_CHECK_MALLOC(xInit);
    DAAL_CHECK_MALLOC(yInit);

    CSRBlockDescriptor<DataType> CSRBlock;
    status = pTable->getSparseBlock(0, N, readOnly, CSRBlock);
    DAAL_CHECK_STATUS_VAR(status);
    DataType * val = CSRBlock.getBlockValuesPtr();
    size_t * col   = CSRBlock.getBlockColumnIndicesPtr();
    size_t * row   = CSRBlock.getBlockRowIndicesPtr();

    // allocate and init memory for auxiliary arrays: posX & posY
    TArrayScalableCalloc<DataType, cpu> posX(nNodes + 1);
    DAAL_CHECK_MALLOC(posX.get());
    services::internal::tmemcpy<DataType, cpu>(posX.get(), xInit, N);
    TArrayScalableCalloc<DataType, cpu> posY(nNodes + 1);
    DAAL_CHECK_MALLOC(posY.get());
    services::internal::tmemcpy<DataType, cpu>(posY.get(), yInit, N);

    // allocate and init memory for auxiliary arrays
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(IdxType, (nNodes + 1), 4);
    TArrayScalableCalloc<IdxType, cpu> child((nNodes + 1) * 4);
    DAAL_CHECK_MALLOC(child.get());
    TArrayScalableCalloc<IdxType, cpu> count(nNodes + 1);
    DAAL_CHECK_MALLOC(count.get());
    TArrayScalableCalloc<DataType, cpu> mass(nNodes + 1);
    DAAL_CHECK_MALLOC(mass.get());
    TArrayScalableCalloc<IdxType, cpu> sort(nNodes + 1);
    DAAL_CHECK_MALLOC(sort.get());
    TArrayScalableCalloc<IdxType, cpu> start(nNodes + 1);
    DAAL_CHECK_MALLOC(start.get());
    TArrayScalableCalloc<DataType, cpu> repX(nNodes + 1);
    DAAL_CHECK_MALLOC(repX.get());
    TArrayScalableCalloc<DataType, cpu> repY(nNodes + 1);
    DAAL_CHECK_MALLOC(repY.get());
    TArrayScalableCalloc<DataType, cpu> attrX(N);
    DAAL_CHECK_MALLOC(attrX.get());
    TArrayScalableCalloc<DataType, cpu> attrY(N);
    DAAL_CHECK_MALLOC(attrY.get());
    TArrayScalableCalloc<DataType, cpu> gainX(N);
    DAAL_CHECK_MALLOC(gainX.get());
    TArrayScalableCalloc<DataType, cpu> gainY(N);
    DAAL_CHECK_MALLOC(gainY.get());
    TArrayScalableCalloc<DataType, cpu> oldForceX(N);
    DAAL_CHECK_MALLOC(oldForceX.get());
    TArrayScalableCalloc<DataType, cpu> oldForceY(N);
    DAAL_CHECK_MALLOC(oldForceY.get());
    TArrayScalableCalloc<IdxType, cpu> duplicates(N);
    DAAL_CHECK_MALLOC(duplicates.get());

    status = maxRowElementsImpl<IdxType, cpu>(row, N, nElements, blockOfRows);
    DAAL_CHECK_STATUS_VAR(status);

    //start iterations
    for (IdxType i = 0; i < explorationIter; ++i)
    {
        status = boundingBoxKernelImpl<IdxType, DataType, cpu>(posX.get(), posY.get(), N, nNodes, radius, blockOfRows);
        DAAL_CHECK_STATUS_VAR(status);
        //std::cout << "BB" << std::endl;

        status = qTreeBuildingKernelImpl<IdxType, DataType, cpu>(sort.get(), child.get(), posX.get(), posY.get(), duplicates.get(), nNodes, N,
                                                                 maxDepth, bottom, radius, blockOfRows);
        DAAL_CHECK_STATUS_VAR(status);
        //std::cout << "qtree" << std::endl;

        status = summarizationKernelImpl<IdxType, DataType, cpu>(count.get(), child.get(), mass.get(), posX.get(), posY.get(), duplicates.get(),
                                                                 nNodes, N, bottom);
        DAAL_CHECK_STATUS_VAR(status);
        //std::cout << "summ kernel" << std::endl;

        status = sortKernelImpl<IdxType, cpu>(sort.get(), count.get(), start.get(), child.get(), nNodes, N, bottom);
        DAAL_CHECK_STATUS_VAR(status);
        //std::cout << "sort kernel" << std::endl;

        status = repulsionKernelImpl<IdxType, DataType, cpu>(theta, eps, sort.get(), child.get(), mass.get(), posX.get(), posY.get(), repX.get(),
                                                             repY.get(), zNorm, nNodes, N, radius, maxDepth, blockOfRows);
        DAAL_CHECK_STATUS_VAR(status);
        //std::cout << "repulsion kernel" << std::endl;

        if (((i + 1) % nIterCheck == 0) || (i == explorationIter - 1))
        {
            status = attractiveKernelImpl<true, IdxType, DataType, cpu>(val, col, row, posX.get(), posY.get(), attrX.get(), attrY.get(), zNorm,
                                                                        divergence, nNodes, N, nnz, nElements, exaggeration, eps, blockOfRows);
        }
        else
        {
            status = attractiveKernelImpl<false, IdxType, DataType, cpu>(val, col, row, posX.get(), posY.get(), attrX.get(), attrY.get(), zNorm,
                                                                         divergence, nNodes, N, nnz, nElements, exaggeration, eps, blockOfRows);
        }
        DAAL_CHECK_STATUS_VAR(status);
        //std::cout << "attractive kernel" << std::endl;

        status = integrationKernelImpl<IdxType, DataType, cpu>(eta, momentum, exaggeration, posX.get(), posY.get(), attrX.get(), attrY.get(),
                                                               repX.get(), repY.get(), gainX.get(), gainY.get(), oldForceX.get(), oldForceY.get(),
                                                               gradNorm, zNorm, nNodes, N, blockOfRows);
        DAAL_CHECK_STATUS_VAR(status);
        //std::cout << "integration kernel" << std::endl;

        if ((i + 1) % nIterCheck == 0)
        {
            if (divergence < bestDivergence)
            {
                bestDivergence = divergence;
                bestIter       = i;
            }

            if (gradNorm <= minGradNorm)
            {
                curIter = i;
                break;
            }
            curIter = i;
        }
    }

    momentum     = 0.8;
    exaggeration = 1.;

    for (IdxType i = explorationIter; i < maxIter; ++i)
    {
        status = boundingBoxKernelImpl<IdxType, DataType, cpu>(posX.get(), posY.get(), N, nNodes, radius, blockOfRows);
        DAAL_CHECK_STATUS_VAR(status);
        //std::cout << "BB" << std::endl;

        status = qTreeBuildingKernelImpl<IdxType, DataType, cpu>(sort.get(), child.get(), posX.get(), posY.get(), duplicates.get(), nNodes, N,
                                                                 maxDepth, bottom, radius, blockOfRows);
        DAAL_CHECK_STATUS_VAR(status);

        //std::cout << "qtree" << std::endl;

        status = summarizationKernelImpl<IdxType, DataType, cpu>(count.get(), child.get(), mass.get(), posX.get(), posY.get(), duplicates.get(),
                                                                 nNodes, N, bottom);
        DAAL_CHECK_STATUS_VAR(status);
        //std::cout << "summ kernel" << std::endl;

        status = sortKernelImpl<IdxType, cpu>(sort.get(), count.get(), start.get(), child.get(), nNodes, N, bottom);
        DAAL_CHECK_STATUS_VAR(status);
        //std::cout << "sort kernel" << std::endl;

        status = repulsionKernelImpl<IdxType, DataType, cpu>(theta, eps, sort.get(), child.get(), mass.get(), posX.get(), posY.get(), repX.get(),
                                                             repY.get(), zNorm, nNodes, N, radius, maxDepth, blockOfRows);
        DAAL_CHECK_STATUS_VAR(status);
        //std::cout << "repulsion kernel" << std::endl;

        if (((i + 1) % nIterCheck == 0) || (i == maxIter - 1))
        {
            status = attractiveKernelImpl<true, IdxType, DataType, cpu>(val, col, row, posX.get(), posY.get(), attrX.get(), attrY.get(), zNorm,
                                                                        divergence, nNodes, N, nnz, nElements, exaggeration, eps, blockOfRows);
        }
        else
        {
            status = attractiveKernelImpl<false, IdxType, DataType, cpu>(val, col, row, posX.get(), posY.get(), attrX.get(), attrY.get(), zNorm,
                                                                         divergence, nNodes, N, nnz, nElements, exaggeration, eps, blockOfRows);
        }
        DAAL_CHECK_STATUS_VAR(status);

        status = integrationKernelImpl<IdxType, DataType, cpu>(eta, momentum, exaggeration, posX.get(), posY.get(), attrX.get(), attrY.get(),
                                                               repX.get(), repY.get(), gainX.get(), gainY.get(), oldForceX.get(), oldForceY.get(),
                                                               gradNorm, zNorm, nNodes, N, blockOfRows);
        DAAL_CHECK_STATUS_VAR(status);

        if (((i + 1) % nIterCheck == 0) || (i == maxIter - 1))
        {
            if (divergence < bestDivergence)
            {
                bestDivergence = divergence;
                bestIter       = i;
            }
            else if (i - bestIter > nIterWithoutProgress)
            {
                curIter = i;
                break;
            }

            if (gradNorm <= minGradNorm)
            {
                curIter = i;
                break;
            }
            curIter = i;
        }
    }

    //save results
    services::internal::tmemcpy<DataType, cpu>(xInit, posX.get(), N);
    services::internal::tmemcpy<DataType, cpu>(yInit, posY.get(), N);

    //release block
    status = pTable->releaseSparseBlock(CSRBlock);
    DAAL_CHECK_STATUS_VAR(status);

    return services::Status();
}

template <typename algorithmIdxType, typename algorithmFPType>
DAAL_EXPORT void tsneGradientDescent(const NumericTablePtr initTable, const CSRNumericTablePtr pTable, const NumericTablePtr sizeIterTable,
                                     const NumericTablePtr paramTable, const NumericTablePtr resultTable)
{
#define DAAL_TSNE_GRADIENT_DESCENT(cpuId, ...) tsneGradientDescentImpl<algorithmIdxType, algorithmFPType, cpuId>(__VA_ARGS__);

    DAAL_DISPATCH_FUNCTION_BY_CPU_SAFE(DAAL_TSNE_GRADIENT_DESCENT, initTable, pTable, sizeIterTable, paramTable, resultTable);

#undef DAAL_TSNE_GRADIENT_DESCENT
}

template DAAL_EXPORT void tsneGradientDescent<int, DAAL_FPTYPE>(const NumericTablePtr initTable, const CSRNumericTablePtr pTable,
                                                                const NumericTablePtr sizeIterTable, const NumericTablePtr paramTable,
                                                                const NumericTablePtr resultTable);

} // namespace internal
} // namespace algorithms
} // namespace daal

#endif
