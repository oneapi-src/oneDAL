/* file tsne_gradient_descent_impl.i */
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

/*
//++
//  Q-tree-based 2D tSNE algorithm implementation.
//
//  REFERENCES
//
//  1. Laurens Van Der Maaten,
//     Accelerating t-SNE using tree-based algorithms,
//     Journal of Machine Learning Research 15, Issue 1 (2014), pp. 3221 - 3245.
//     DOI: 10.5555/2627435.2697068.
//  2. Burtscher, Martin & Pingali, Keshav.
//     An Efficient CUDA Implementation of the Tree-Based Barnes Hut n-Body
//     Algorithm.
//     GPU Computing Gems Emerald Edition.
//     DOI: 10.1016/B978-0-12-384988-5.00006-1. 
//--
*/

#ifndef __TSNE_GRADIENT_DESCENT_IMPL_I__
#define __TSNE_GRADIENT_DESCENT_IMPL_I__

#include "tsne_gradient_descent_kernel.h"

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

#define MAX_LEVEL 32
template <typename DataType>
struct xyType
{
    DataType x;
    DataType y;
};

template <typename IdxType, typename xyType>
struct MemoryCtxType
{
    int capacity            = 0;
    xyType * pos            = nullptr;
    uint64_t * morton_codes = nullptr;
    IdxType * z_order_idx   = nullptr;
    IdxType * t_order_idx   = nullptr;
    xyType * rep            = nullptr;

    xyType * attr = nullptr;
    xyType * gain = nullptr;
    xyType * ofor = nullptr;
};

struct qTreeNode
{
    int fpos; // sign (1 bit), nonempty children (2 bits), offset to first child (29 bit)
    int cnt;  // count of points in subspace
};

template <typename IdxType, typename xyType>
struct TreeCtxType
{
    int capacity             = 0;
    int size                 = 0;
    int layerSize[MAX_LEVEL] = {};
    int layerOffs[MAX_LEVEL] = {};
    qTreeNode * tree         = nullptr;
    xyType * cent            = nullptr;
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
services::Status boundingBoxKernelImpl(xyType<DataType> * pos, const IdxType N, DataType & radius, DataType & centerx, DataType & centery)
{
    DAAL_CHECK_MALLOC(pos);

    DataType box[4] = { pos[0].x, pos[0].x, pos[0].y, pos[0].y };

    daal::static_tls<DataType *> tlsBox([=]() {
        auto localBox = services::internal::service_malloc<DataType, cpu>(4);

        localBox[0] = daal::services::internal::MaxVal<DataType>::get();
        localBox[1] = -daal::services::internal::MaxVal<DataType>::get();
        localBox[2] = daal::services::internal::MaxVal<DataType>::get();
        localBox[3] = -daal::services::internal::MaxVal<DataType>::get();

        return localBox;
    });

    const IdxType nThreads = tlsBox.nthreads();
    // const IdxType sizeOfBlock = services::internal::min<cpu, IdxType>(256, (N + nThreads - 1) / nThreads);
    // const IdxType nBlocks     = (N + sizeOfBlock - 1) / sizeOfBlock;
    const IdxType sizeOfBlock = services::internal::min<cpu, IdxType>(256, N / nThreads + 1);
    const IdxType nBlocks     = N / sizeOfBlock + !!(N % sizeOfBlock);

    daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        const IdxType iStart = iBlock * sizeOfBlock;
        const IdxType iEnd   = services::internal::min<cpu, IdxType>(N, iStart + sizeOfBlock);
        DataType * localBox  = tlsBox.local(tid);

        for (IdxType i = iStart; i < iEnd; ++i)
        {
            localBox[0] = services::internal::min<cpu, DataType>(localBox[0], pos[i].x);
            localBox[1] = services::internal::max<cpu, DataType>(localBox[1], pos[i].x);
            localBox[2] = services::internal::min<cpu, DataType>(localBox[2], pos[i].y);
            localBox[3] = services::internal::max<cpu, DataType>(localBox[3], pos[i].y);
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

    //save results
    centerx = (box[0] + box[1]) * 0.5;
    centery = (box[2] + box[3]) * 0.5;
    radius  = services::internal::max<cpu, DataType>(box[1] - box[0], box[3] - box[2]) * 0.5005f;

    return services::Status();
}

template <typename IdxType, typename DataType, daal::CpuType cpu>
inline void buildSubtree5(TreeCtxType<IdxType, xyType<DataType> > & qTree, int level, IdxType * zOrder, IdxType * tOrder, uint64_t * mc, int * hist)
{
    const int sh   = 54 - (level << 1);
    const int bcnt = qTree.tree[0].cnt;
    const int bpos = qTree.tree[0].fpos;

    int * offs = hist + 1024 + 1024;

    int i, l, c;

    // Sort indexes for 10 bits of the morton code (5 tree levels)
    for (offs[0] = 0, i = 1; i < 1024; i++) offs[i] = offs[i - 1] + hist[i - 1];
    if (level)
    {
        for (i = bpos; i < bpos + bcnt; i++) tOrder[bpos + offs[(mc[zOrder[i]] >> sh) & 0x3FF]++] = zOrder[i];
        for (i = bpos; i < bpos + bcnt; i++) zOrder[i] = tOrder[i];
    }
    else
    {
        for (i = bpos; i < bpos + bcnt; i++) zOrder[offs[mc[i] >> sh]++] = i;
    }

    // Hierarchically aggregate histogram for 5 levels
    for (i = 0; i < 256 + 64 + 16 + 4 + 1; i++) hist[1024 + i] = hist[(i << 2) + 0] + hist[(i << 2) + 1] + hist[(i << 2) + 2] + hist[(i << 2) + 3];

    const int h_ofst[6] = { 1024 + 256 + 64 + 16 + 4, 1024 + 256 + 64 + 16, 1024 + 256 + 64, 1024 + 256, 1024, 0 };
    const int h_size[6] = { 1, 4, 16, 64, 256, 1024 };

    int nodeSize, posOffs, nodeOffs = 0, childOffs = 1;

    // Construct quadTree layer-by-layer using aggregated histograms
    for (l = 0; l < 6; l++)
    {
        posOffs            = bpos;
        qTree.layerOffs[l] = nodeOffs;
        for (c = 0; c < h_size[l]; c++)
        {
            nodeSize = hist[h_ofst[l] + c];

            if (nodeSize == 0) continue;

            if (nodeSize > 0)
            {
                qTree.tree[nodeOffs].cnt = nodeSize;
                if (l < 5)
                {
                    if (nodeSize > level + l + 1)
                    {
                        int cnt = !!hist[h_ofst[l + 1] + (c << 2) + 0] + !!hist[h_ofst[l + 1] + (c << 2) + 1] + !!hist[h_ofst[l + 1] + (c << 2) + 2]
                                  + !!hist[h_ofst[l + 1] + (c << 2) + 3];

                        // Adding internal node with 'cnt'  non-empty children
                        // Ttheir offset is 'childOffs'
                        qTree.tree[nodeOffs].fpos = 0x80000000 | ((cnt - 1) << 29) | childOffs;
                        childOffs += cnt;
                    }
                    else
                    {
                        // Adding internal leaf with size 'nodeSize'
                        qTree.tree[nodeOffs].fpos = posOffs;

                        hist[h_ofst[l + 1] + (c << 2) + 0] = -hist[h_ofst[l + 1] + (c << 2) + 0];
                        hist[h_ofst[l + 1] + (c << 2) + 1] = -hist[h_ofst[l + 1] + (c << 2) + 1];
                        hist[h_ofst[l + 1] + (c << 2) + 2] = -hist[h_ofst[l + 1] + (c << 2) + 2];
                        hist[h_ofst[l + 1] + (c << 2) + 3] = -hist[h_ofst[l + 1] + (c << 2) + 3];
                    }
                }
                else
                {
                    // Adding terminal leaf with size 'nodeSize'
                    qTree.tree[nodeOffs].fpos = posOffs;
                }
                nodeOffs++;
            }
            else
            {
                // Skipping non-empty node
                if (l < 5)
                {
                    hist[h_ofst[l + 1] + (c << 2) + 0] = -hist[h_ofst[l + 1] + (c << 2) + 0];
                    hist[h_ofst[l + 1] + (c << 2) + 1] = -hist[h_ofst[l + 1] + (c << 2) + 1];
                    hist[h_ofst[l + 1] + (c << 2) + 2] = -hist[h_ofst[l + 1] + (c << 2) + 2];
                    hist[h_ofst[l + 1] + (c << 2) + 3] = -hist[h_ofst[l + 1] + (c << 2) + 3];
                }
                nodeSize = -nodeSize;
            }
            posOffs += nodeSize;
        }
    }
    qTree.layerOffs[6] = qTree.size = nodeOffs;

    for (int i = 0; i < 6; i++) qTree.layerSize[i] = qTree.layerOffs[i + 1] - qTree.layerOffs[i];
}

template <typename IdxType, typename DataType, daal::CpuType cpu>
services::Status qTreeBuildingKernelImpl(MemoryCtxType<IdxType, xyType<DataType> > & mem, TreeCtxType<IdxType, xyType<DataType> > & qTree,
                                         const DataType & radius, const DataType & centerx, const DataType & centery)
{
    DAAL_CHECK_MALLOC(mem.pos);
    DAAL_CHECK_MALLOC(mem.morton_codes);

    int * mHist = services::internal::service_calloc<int, cpu>(1024 + 1024 + 1024);
    DAAL_CHECK_MALLOC(mHist);

    daal::static_tls<int *> tlsHist1024([=]() {
        auto localHist = services::internal::service_calloc<int, cpu>(1024);
        return localHist;
    });

    const IdxType nThreads    = tlsHist1024.nthreads();
    const IdxType sizeOfBlock = services::internal::min<cpu, IdxType>(256, (mem.capacity + nThreads - 1) / nThreads);
    const IdxType nBlocks     = (mem.capacity + sizeOfBlock - 1) / sizeOfBlock;

    daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        const IdxType iStart = iBlock * sizeOfBlock;
        const IdxType iEnd   = services::internal::min<cpu, IdxType>(mem.capacity, iStart + sizeOfBlock);
        const DataType rootx = centerx - radius;
        const DataType rooty = centery - radius;

        const double scale = 2147483648.0 / radius;

        int * hist = tlsHist1024.local(tid);

        uint64_t x, y;

        for (IdxType i = iStart; i < iEnd; i++)
        {
            x = (uint64_t)((mem.pos[i].x - rootx) * scale);
            y = (uint64_t)((mem.pos[i].y - rooty) * scale);

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

            x |= (y << 1);

            mem.morton_codes[i] = x;

            hist[x >> 54]++;
        }
    });

    tlsHist1024.reduce([&](int * ptr) -> void {
        if (!ptr) return;

        PRAGMA_VECTOR_ALWAYS
        PRAGMA_VECTOR_ALIGNED
        for (int i = 0; i < 1024; i++) mHist[i] += ptr[i];

        services::internal::service_free<int, cpu>(ptr);
    });

    /************************* Build the quadTree ***********************/
    {
        qTree.size         = 0;
        qTree.tree[0].fpos = 0;
        qTree.tree[0].cnt  = mem.capacity;
        for (int i = 0; i < 32; i++) qTree.layerSize[i] = 0;
        for (int i = 0; i < 32; i++) qTree.layerOffs[i] = 0;

        buildSubtree5<IdxType, DataType, cpu>(qTree, 0, mem.z_order_idx, mem.t_order_idx, mem.morton_codes, mHist);

        qTreeNode * subNodes                               = nullptr;
        TreeCtxType<IdxType, xyType<DataType> > * subTrees = nullptr;
        int subTreeCnt                                     = 0;

        for (int pass = 0; pass < 5; pass++)
        {
            int bLevel    = 5 * (pass + 1);
            int bLayerBeg = qTree.layerOffs[bLevel];
            int bLayerEnd = qTree.layerOffs[bLevel] + qTree.layerSize[bLevel];
            int bNodes    = 0;

            for (int c = bLayerBeg; c < bLayerEnd; c++)
                if (qTree.tree[c].cnt > bLevel + 1) bNodes++;

            // Terminate subtrees creation if there are not enough bottom nodes to split
            if (bNodes < 1) break;

            // Re/allocate worker space for bottom subtrees if needed
            if (bNodes > subTreeCnt)
            {
                if (subTrees)
                {
                    services::internal::service_free<TreeCtxType<IdxType, xyType<DataType> >, cpu>(subTrees);
                    services::internal::service_free<qTreeNode, cpu>(subNodes);
                }
                subNodes   = services::internal::service_malloc<qTreeNode, cpu>(bNodes * 2048);
                subTrees   = services::internal::service_malloc<TreeCtxType<IdxType, xyType<DataType> >, cpu>(bNodes);
                subTreeCnt = bNodes;
            }

            for (int c = bLayerBeg, bNodes = 0; c < bLayerEnd; c++)
                if (qTree.tree[c].cnt > bLevel + 1)
                {
                    subTrees[bNodes].size     = 0;
                    subTrees[bNodes].capacity = 2048;
                    subTrees[bNodes].tree     = subNodes + bNodes * 2048;
                    subTrees[bNodes].tree[0]  = qTree.tree[c];
                    bNodes++;
                }

            // Build bottom subtrees in parallel
            const IdxType nThreads    = threader_get_threads_number();
            const IdxType sizeOfBlock = 1;
            const IdxType nBlocks     = bNodes;

            daal::threader_for(nBlocks, nBlocks, [&](IdxType iSubTree) {
                int * hist = services::internal::service_calloc<int, cpu>(3072);

                const int sft  = 54 - (bLevel << 1);
                const int bcnt = subTrees[iSubTree].tree[0].cnt;
                const int bpos = subTrees[iSubTree].tree[0].fpos;

                // services::internal::service_memset<int, cpu>(hist, 0, 1024);
                for (int i = bpos; i < bpos + bcnt; i++) hist[(mem.morton_codes[mem.z_order_idx[i]] >> sft) & 0x3FF]++;

                buildSubtree5<IdxType, DataType, cpu>(subTrees[iSubTree], bLevel, mem.z_order_idx, mem.t_order_idx, mem.morton_codes, hist);

                services::internal::service_free<int, cpu>(hist);
            });

            // Reallocate the tree if needed
            int newTreeSize = qTree.size;

            for (int l = 1; l < 6; l++)
                for (int s = 0; s < bNodes; s++) newTreeSize += subTrees[s].layerSize[l];

            if (newTreeSize > qTree.capacity)
            {
                int capacity      = newTreeSize + (newTreeSize >> 2);
                qTreeNode * nodes = services::internal::service_malloc<qTreeNode, cpu>(capacity);

                services::internal::tmemcpy<qTreeNode, cpu>(nodes, qTree.tree, qTree.size);
                services::internal::service_free<qTreeNode, cpu>(qTree.tree);
                services::internal::service_free<xyType<DataType>, cpu>(qTree.cent);

                qTree.cent     = services::internal::service_malloc<xyType<DataType>, cpu>(capacity);
                qTree.tree     = nodes;
                qTree.capacity = capacity;
            }

            // Aggregate subtrees into main tree
            {
                // Replace splitted bottom leafs with top nodes from subtrees
                int nodeOffs  = qTree.layerOffs[bLevel];
                int childOffs = qTree.layerOffs[bLevel + 1];

                bNodes = 0;
                for (int c = 0; c < qTree.layerSize[bLevel]; c++)
                {
                    if (qTree.tree[nodeOffs].cnt > bLevel + 1)
                    {
                        qTree.tree[nodeOffs].fpos = subTrees[bNodes].tree[0].fpos & 0xE0000000;
                        qTree.tree[nodeOffs].fpos |= childOffs;
                        childOffs += 1 + ((qTree.tree[nodeOffs].fpos >> 29) & 0x3);
                        bNodes++;
                    }
                    nodeOffs++;
                }

                // Copy remaining nodes from subtrees recalculating offsets
                for (int l = 1; l < 6; l++)
                {
                    qTree.layerOffs[bLevel + l] = nodeOffs;
                    for (int s = 0; s < bNodes; s++)
                    {
                        for (int c = 0; c < subTrees[s].layerSize[l]; c++)
                        {
                            qTree.tree[nodeOffs] = subTrees[s].tree[subTrees[s].layerOffs[l] + c];
                            if (qTree.tree[nodeOffs].fpos < 0)
                            {
                                qTree.tree[nodeOffs].fpos &= 0xE0000000;
                                qTree.tree[nodeOffs].fpos |= childOffs;
                                childOffs += 1 + ((qTree.tree[nodeOffs].fpos >> 29) & 0x3);
                            }
                            nodeOffs++;
                        }
                    }
                }
                qTree.layerOffs[bLevel + 6] = qTree.size = nodeOffs;
                for (int i = 0; i < 6; i++) qTree.layerSize[bLevel + i] = qTree.layerOffs[bLevel + i + 1] - qTree.layerOffs[bLevel + i];
            }
        }

        if (subTreeCnt)
        {
            services::internal::service_free<qTreeNode, cpu>(subTrees[0].tree);
            services::internal::service_free<TreeCtxType<IdxType, xyType<DataType> >, cpu>(subTrees);
        }
    }

    services::internal::service_free<int, cpu>(mHist);
    return services::Status();
}

template <typename IdxType, typename DataType, daal::CpuType cpu>
services::Status summarizationKernelImpl(MemoryCtxType<IdxType, xyType<DataType> > & mem, TreeCtxType<IdxType, xyType<DataType> > & qTree)
{
    IdxType nThreads = threader_get_threads_number();
    IdxType nBlocks, lOffset, sizeOfBlock = 1;

    for (int l = 1; l < MAX_LEVEL + 1; l++)
    {
        nBlocks = qTree.layerSize[MAX_LEVEL - l];
        lOffset = qTree.layerOffs[MAX_LEVEL - l];
        if (nBlocks == 0) continue;
        daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
            IdxType iPos = lOffset + iBlock;
            DataType cx, cy;
            if (qTree.tree[iPos].fpos < 0)
            {
                int pos = qTree.tree[iPos].fpos & ~0xE0000000;
                int cnt = 1 + ((qTree.tree[iPos].fpos >> 29) & 0x3);

                cx = qTree.cent[pos].x;
                cy = qTree.cent[pos].y;
                for (int c = 1; c < cnt; c++)
                {
                    cx += qTree.cent[pos + c].x;
                    cy += qTree.cent[pos + c].y;
                }
            }
            else
            {
                cx = mem.pos[mem.z_order_idx[qTree.tree[iPos].fpos]].x;
                cy = mem.pos[mem.z_order_idx[qTree.tree[iPos].fpos]].y;

                for (int c = 1; c < qTree.tree[iPos].cnt; c++)
                {
                    cx += mem.pos[mem.z_order_idx[qTree.tree[iPos].fpos + c]].x;
                    cy += mem.pos[mem.z_order_idx[qTree.tree[iPos].fpos + c]].y;
                }
            }
            qTree.cent[iPos].x = cx;
            qTree.cent[iPos].y = cy;
        });
    }

    nThreads    = threader_get_threads_number();
    sizeOfBlock = services::internal::min<cpu, IdxType>(256, (qTree.size + nThreads - 1) / nThreads);
    nBlocks     = (qTree.size + sizeOfBlock - 1) / sizeOfBlock;

    daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        const IdxType iStart = iBlock * sizeOfBlock;
        const IdxType iEnd   = services::internal::min<cpu, IdxType>(qTree.size, iStart + sizeOfBlock);
        for (IdxType i = iStart; i < iEnd; ++i)
        {
            DataType iMass = DataType(1) / qTree.tree[i].cnt;
            qTree.cent[i].x *= iMass;
            qTree.cent[i].y *= iMass;
        }
    });

    return services::Status();
}

template <typename IdxType, typename DataType, daal::CpuType cpu>
services::Status repulsionKernelImpl(MemoryCtxType<IdxType, xyType<DataType> > & mem, TreeCtxType<IdxType, xyType<DataType> > & qTree,
                                     const DataType theta, const DataType eps, DataType & zNorm, const DataType & radius)
{
    const DataType epsInc = eps + DataType(1);

    SafeStatus safeStat;
    DataType dq[MAX_LEVEL];

    dq[0] = (radius * radius) / (theta * theta);
    for (auto i = 1; i < MAX_LEVEL; i++) dq[i] = dq[i - 1] * 0.25;
    for (auto i = 0; i < MAX_LEVEL; i++) dq[i] += epsInc;

    daal::StaticTlsSum<DataType, cpu> sumTlsData(1);

    const IdxType nThreads    = sumTlsData.nthreads();
    const IdxType sizeOfBlock = services::internal::min<cpu, IdxType>(256, (mem.capacity + nThreads - 1) / nThreads);
    const IdxType nBlocks     = (mem.capacity + sizeOfBlock - 1) / sizeOfBlock;

    IdxType * nStack = services::internal::service_malloc<IdxType, cpu>(nThreads * MAX_LEVEL * 4);
    int * nLevel     = services::internal::service_malloc<int, cpu>(nThreads * MAX_LEVEL * 4);

    daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        const IdxType iStart = iBlock * sizeOfBlock;
        const IdxType iEnd   = services::internal::min<cpu, IdxType>(mem.capacity, iStart + sizeOfBlock);
        DataType * lSum      = sumTlsData.local(tid);

        for (IdxType k = iStart; k < iEnd; ++k)
        {
            const IdxType i   = mem.z_order_idx[k]; // 4*N
            const DataType px = mem.pos[i].x;       // 4*N
            const DataType py = mem.pos[i].y;       // 4*N

            // DataType * lSum = sumTlsData.local(tid);
            DataType vx = 0.;
            DataType vy = 0.;

            IdxType * lStack = nStack + tid * MAX_LEVEL * 4; // 3*N Flops
            int * lLevel     = nLevel + tid * MAX_LEVEL * 4; // 3*N Flops

            int cnt = 1 + ((qTree.tree[0].fpos >> 29) & 0x3); // 3*N Flops
            int pos = qTree.tree[0].fpos & ~0xE0000000;       // 2*N Flops
            int idx = 0;

            switch (cnt)
            {
            case 4: lStack[idx] = pos + 3; lLevel[idx++] = 1; // 2*N Flops
            case 3: lStack[idx] = pos + 2; lLevel[idx++] = 1;
            case 2: lStack[idx] = pos + 1; lLevel[idx++] = 1;
            default:
            case 1: lStack[idx] = pos; lLevel[idx++] = 1;
            }

            while (idx > 0)
            {
                idx--;                                          // 1*N*TS Flops
                DataType dx   = px - qTree.cent[lStack[idx]].x; // 1*N*TS Flops, 4*N*TS Bytes
                DataType dy   = py - qTree.cent[lStack[idx]].y; // 1*N*TS Flops, 4*N*TS Bytes
                DataType dxy1 = dx * dx + dy * dy + epsInc;     // 4*N*TS Flops
                int mass      = qTree.tree[lStack[idx]].cnt;    // 8*N*TS Bytes
                int fpos      = qTree.tree[lStack[idx]].fpos;
                int level     = lLevel[idx];
                DataType tdist_2;

                if ((mass == 1) || (dxy1 >= dq[level])) // 3*N*TS Flops
                {
                    // Distant node, use centroid to calculate force vectors
                    tdist_2 = mass / (dxy1 * dxy1); // 1*N*TS DIV, 1*N*TS
                    lSum[0] += tdist_2 * dxy1;      // 2*N*TS
                    vx += dx * tdist_2;             // 2*N*TS
                    vy += dy * tdist_2;             // 2*N*TS
                }
                else if (fpos < 0)
                {
                    // Intermediate node, add children to stack
                    cnt = 1 + ((fpos >> 29) & 0x3); // 3*N*TS
                    pos = fpos & ~0xE0000000;       // 2*N*TS

                    switch (cnt)
                    {
                    case 4: lStack[idx] = pos + 3; lLevel[idx++] = level + 1; // 1*N*TS
                    case 3: lStack[idx] = pos + 2; lLevel[idx++] = level + 1;
                    case 2: lStack[idx] = pos + 1; lLevel[idx++] = level + 1;
                    default:
                    case 1: lStack[idx] = pos; lLevel[idx++] = level + 1;
                    }
                }
                else
                {
                    // Leaf node, process all point separatly
                    for (int c = 0; c < mass; c++)
                    {
                        // _mm_prefetch(&mem.pos[mem.z_order_idx[fpos + c + 8]], _MM_HINT_T0);
                        dx   = px - mem.pos[mem.z_order_idx[fpos + c]].x;
                        dy   = py - mem.pos[mem.z_order_idx[fpos + c]].y;
                        dxy1 = dx * dx + dy * dy + epsInc;

                        tdist_2 = 1.0 / dxy1;
                        lSum[0] += tdist_2;

                        tdist_2 *= tdist_2;
                        vx += dx * tdist_2;
                        vy += dy * tdist_2;
                    }
                }
            }

            mem.rep[i].x = vx;
            mem.rep[i].y = vy;
        }
    });

    zNorm = 0.;
    sumTlsData.reduceTo(&zNorm, 1);

    services::internal::service_free<int, cpu>(nLevel);
    services::internal::service_free<IdxType, cpu>(nStack);

    return services::Status();
}

#include <cstdio>

/* Generic template implementation of attractive kernel for all data types and various instruction set architectures */
template <bool DivComp, typename IdxType, typename DataType, daal::CpuType cpu>
struct AttractiveKernel
{
    static services::Status impl(const DataType * val, const IdxType * col, const size_t * row, MemoryCtxType<IdxType, xyType<DataType> > & mem,
                                 DataType & zNorm, DataType & divergence, const IdxType N, const IdxType nnz, const IdxType nElements,
                                 const DataType exaggeration)
    {
        DAAL_CHECK_MALLOC(val);
        DAAL_CHECK_MALLOC(col);
        DAAL_CHECK_MALLOC(row);

        const DataType multiplier = exaggeration * DataType(zNorm);
        divergence                = 0.;

        const IdxType prefetch_dist = 32;

        daal::TlsSum<DataType, cpu> divTlsData(1);
        daal::tls<DataType *> logTlsData([=]() { return services::internal::service_scalable_calloc<DataType, cpu>(nElements); });

        const IdxType nThreads    = threader_get_threads_number();
        const IdxType sizeOfBlock = services::internal::min<cpu, size_t>(256, N / nThreads + 1);
        const IdxType nBlocks     = N / sizeOfBlock + !!(N % sizeOfBlock);

        daal::threader_for(nBlocks, nBlocks, [&](IdxType iBlock) {
            const IdxType iStart = iBlock * sizeOfBlock;
            const IdxType iEnd   = services::internal::min<cpu, IdxType>(N, iStart + sizeOfBlock);
            DataType * logLocal  = logTlsData.local();
            DataType * divLocal  = divTlsData.local();

            xyType<DataType> row_point;
            IdxType iCol, prefetch_index;
            DataType y1d, y2d, sqDist, PQ;

            for (IdxType iRow = iStart; iRow < iEnd; ++iRow)
            {
                size_t iSize     = 0;
                mem.attr[iRow].x = 0.0;
                mem.attr[iRow].y = 0.0;
                row_point        = mem.pos[iRow];

                for (IdxType index = row[iRow] - 1; index < row[iRow + 1] - 1; ++index) // 4*N
                {
                    prefetch_index = index + prefetch_dist;
                    if (prefetch_index < nnz) DAAL_PREFETCH_READ_T0(&mem.pos[col[prefetch_index] - 1]);

                    iCol = col[index] - 1; // 4*NNZ byte

                    y1d = row_point.x - mem.pos[iCol].x; // 1*NNZ Flop, 4*N + 4*NNZ byte
                    y2d = row_point.y - mem.pos[iCol].y; // 1*NNZ Flop, 4*N + 4*NNZ byte
                    // const DataType sqDist = services::internal::max<cpu, DataType>(DataType(0), y1d * y1d + y2d * y2d); // To deal with NaNs     // 4*NNZ Flop
                    // const DataType PQ     = val[index] / (sqDist + 1.);            // 1*NNZ div, 1*NNZ flop, 4*NNZ byte
                    sqDist = 1.0 + y1d * y1d + y2d * y2d;
                    PQ     = val[index] / sqDist;

                    // Apply forces
                    mem.attr[iRow].x += PQ * y1d;
                    mem.attr[iRow].y += PQ * y2d;
                    if (DivComp)
                    {
                        // logLocal[iSize++] = val[index] * multiplier * (1. + sqDist);       // 3*NNZ Flop
                        logLocal[iSize++] = val[index] * multiplier * sqDist;
                    }
                }

                if (DivComp)
                {
                    Math<DataType, cpu>::vLog(iSize, logLocal, logLocal);
                    IdxType start = row[iRow] - 1;
                    for (IdxType index = 0; index < iSize; ++index)
                    {
                        divLocal[0] += val[start + index] * logLocal[index]; // 2*NNZ Flop
                    }
                }
            }
        });

        divTlsData.reduceTo(&divergence, 1);
        divergence *= exaggeration;
        logTlsData.reduce([&](DataType * buf) { services::internal::service_scalable_free<DataType, cpu>(buf); });

        //Find_Normalization
        zNorm = DataType(1) / zNorm;

        return services::Status();
    }
};

template <typename IdxType, typename DataType, daal::CpuType cpu>
services::Status integrationKernelImpl(const DataType eta, const DataType momentum, const DataType exaggeration,
                                       MemoryCtxType<IdxType, xyType<DataType> > & mem, DataType & gradNorm, const DataType & zNorm, const IdxType N,
                                       const IdxType & blockOfRows)
{
    const IdxType nThreads    = threader_get_threads_number();
    const IdxType sizeOfBlock = services::internal::min<cpu, IdxType>(256, N / nThreads + 1);
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
            const DataType dx = 4 * (exaggeration * mem.attr[i].x - zNorm * mem.rep[i].x);
            const DataType dy = 4 * (exaggeration * mem.attr[i].y - zNorm * mem.rep[i].y);
            localSum[0] += dx * dx + dy * dy;

            gx = (dx * (ux = mem.ofor[i].x) < DataType(0)) ? mem.gain[i].x + 0.2 : mem.gain[i].x * 0.8;
            if (gx < 0.01) gx = 0.01;

            gy = (dy * (uy = mem.ofor[i].y) < DataType(0)) ? mem.gain[i].y + 0.2 : mem.gain[i].y * 0.8;
            if (gy < 0.01) gy = 0.01;

            mem.gain[i].x = gx;
            mem.gain[i].y = gy;

            mem.ofor[i].x = ux = momentum * ux - eta * gx * dx;
            mem.ofor[i].y = uy = momentum * uy - eta * gy * dy;

            mem.pos[i].x += ux;
            mem.pos[i].y += uy;
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
    const IdxType nNodes          = N <= 50 ? 4 * N : 2 * N; // A small number of points may require more memory to store tree nodes
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
    IdxType maxDepth  = 1;
    IdxType bottom    = nNodes;
    IdxType nElements = 0;
    IdxType bestIter  = 0;

    DataType radius  = 0.;
    DataType centerx = 0.;
    DataType centery = 0.;

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

    IdxType * col_i32 = services::internal::service_malloc<IdxType, cpu>(nnz);

    for (int i = 0; i < nnz; i++)
    {
        col_i32[i] = (IdxType)col[i];
    }

    MemoryCtxType<IdxType, xyType<DataType> > mem;

    // allocate and init memory for auxiliary arrays: posx & posy, morton codes and indices
    mem.capacity = N;

    mem.pos = services::internal::service_malloc<xyType<DataType>, cpu>(mem.capacity);
    DAAL_CHECK_MALLOC(mem.pos);
    mem.morton_codes = services::internal::service_malloc<uint64_t, cpu>(mem.capacity);
    DAAL_CHECK_MALLOC(mem.morton_codes);
    mem.z_order_idx = services::internal::service_malloc<int, cpu>(mem.capacity);
    DAAL_CHECK_MALLOC(mem.z_order_idx);
    mem.t_order_idx = services::internal::service_malloc<int, cpu>(mem.capacity);
    DAAL_CHECK_MALLOC(mem.t_order_idx);
    mem.rep = services::internal::service_malloc<xyType<DataType>, cpu>(mem.capacity);
    DAAL_CHECK_MALLOC(mem.rep);
    mem.attr = services::internal::service_malloc<xyType<DataType>, cpu>(mem.capacity);
    DAAL_CHECK_MALLOC(mem.attr);
    mem.gain = services::internal::service_calloc<xyType<DataType>, cpu>(mem.capacity);
    DAAL_CHECK_MALLOC(mem.gain);
    mem.ofor = services::internal::service_calloc<xyType<DataType>, cpu>(mem.capacity);
    DAAL_CHECK_MALLOC(mem.ofor);

    for (size_t i = 0; i < N; i++)
    {
        mem.pos[i].x = xInit[i];
        mem.pos[i].y = yInit[i];
    }

    TreeCtxType<IdxType, xyType<DataType> > qTree;
    // allocate enough memory to store top 5 levels of qTree
    qTree.capacity = 1024;
    qTree.tree     = services::internal::service_malloc<qTreeNode, cpu>(qTree.capacity);
    DAAL_CHECK_MALLOC(qTree.tree);
    qTree.cent = services::internal::service_malloc<xyType<DataType>, cpu>(qTree.capacity);
    DAAL_CHECK_MALLOC(qTree.cent);

    double boundingBox   = 0.;
    double treeBuild     = 0.;
    double summarization = 0.;
    double repulsion     = 0.;
    double attractive    = 0.;
    double integration   = 0.;

    status = maxRowElementsImpl<IdxType, cpu>(row, N, nElements, blockOfRows);
    DAAL_CHECK_STATUS_VAR(status);

    gradNorm = 0.0;

    //start iterations
    for (IdxType i = 0; i < explorationIter; ++i)
    {
        status = boundingBoxKernelImpl<IdxType, DataType, cpu>(mem.pos, N, radius, centerx, centery);
        DAAL_CHECK_STATUS_VAR(status);

        status = qTreeBuildingKernelImpl<IdxType, DataType, cpu>(mem, qTree, radius, centerx, centery);
        DAAL_CHECK_STATUS_VAR(status);

        status = summarizationKernelImpl<IdxType, DataType, cpu>(mem, qTree);
        DAAL_CHECK_STATUS_VAR(status);

        status = repulsionKernelImpl<IdxType, DataType, cpu>(mem, qTree, theta, eps, zNorm, radius);
        DAAL_CHECK_STATUS_VAR(status);

        if (((i + 1) % nIterCheck == 0) || (i == explorationIter - 1))
        {
            status = AttractiveKernel<true, IdxType, DataType, cpu>::impl(val, col_i32, row, mem, zNorm, divergence, N, nnz, nElements, exaggeration);
        }
        else
        {
            status =
                AttractiveKernel<false, IdxType, DataType, cpu>::impl(val, col_i32, row, mem, zNorm, divergence, N, nnz, nElements, exaggeration);
        }

        DAAL_CHECK_STATUS_VAR(status);

        status = integrationKernelImpl<IdxType, DataType, cpu>(eta, momentum, exaggeration, mem, gradNorm, zNorm, N, blockOfRows);
        DAAL_CHECK_STATUS_VAR(status);

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
        status = boundingBoxKernelImpl<IdxType, DataType, cpu>(mem.pos, N, radius, centerx, centery);
        DAAL_CHECK_STATUS_VAR(status);

        status = qTreeBuildingKernelImpl<IdxType, DataType, cpu>(mem, qTree, radius, centerx, centery);
        DAAL_CHECK_STATUS_VAR(status);

        status = summarizationKernelImpl<IdxType, DataType, cpu>(mem, qTree);
        DAAL_CHECK_STATUS_VAR(status);

        status = repulsionKernelImpl<IdxType, DataType, cpu>(mem, qTree, theta, eps, zNorm, radius);

        DAAL_CHECK_STATUS_VAR(status);

        if (((i + 1) % nIterCheck == 0) || (i == explorationIter - 1))
        {
            status = AttractiveKernel<true, IdxType, DataType, cpu>::impl(val, col_i32, row, mem, zNorm, divergence, N, nnz, nElements, exaggeration);
        }
        else
        {
            status =
                AttractiveKernel<false, IdxType, DataType, cpu>::impl(val, col_i32, row, mem, zNorm, divergence, N, nnz, nElements, exaggeration);
        }

        DAAL_CHECK_STATUS_VAR(status);

        status = integrationKernelImpl<IdxType, DataType, cpu>(eta, momentum, exaggeration, mem, gradNorm, zNorm, N, blockOfRows);
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
    for (size_t i = 0; i < N; i++)
    {
        xInit[i] = mem.pos[i].x;
        yInit[i] = mem.pos[i].y;
    }

    //release block
    status = pTable->releaseSparseBlock(CSRBlock);
    DAAL_CHECK_STATUS_VAR(status);

    services::internal::service_free<qTreeNode, cpu>(qTree.tree);
    services::internal::service_free<xyType<DataType>, cpu>(qTree.cent);
    services::internal::service_free<int, cpu>(mem.t_order_idx);
    services::internal::service_free<int, cpu>(mem.z_order_idx);
    services::internal::service_free<uint64_t, cpu>(mem.morton_codes);
    services::internal::service_free<xyType<DataType>, cpu>(mem.pos);
    services::internal::service_free<xyType<DataType>, cpu>(mem.rep);

    services::internal::service_free<IdxType, cpu>(col_i32);
    return services::Status();
}

} // namespace internal
} // namespace algorithms
} // namespace daal

#endif // __INTERNAL_TSNE_GRADIENT_DESCENT_IMPL_I__
