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

template <typename IdxType, typename DataType, daal::CpuType cpu>
services::Status qTreeBuildingKernelImpl(IdxType * child, const DataType * posX, const DataType * posY, IdxType * duplicates, const IdxType nNodes,
                                         const IdxType N, IdxType & maxDepth, IdxType & bottom, const DataType & radius)
{
    // internal variables
    IdxType j               = 0;
    IdxType depth           = 0;
    IdxType ch              = 0;
    IdxType n               = 0;
    IdxType locked          = 0;
    IdxType patch           = 0;
    DataType x              = 0.;
    DataType y              = 0.;
    DataType r              = 0.;
    DataType px             = 0.;
    DataType py             = 0.;
    const DataType localEps = 1e-6;

    //initialize array
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(IdxType, (nNodes + 1), 4);
    services::internal::service_memset<IdxType, cpu>(child, -1, (nNodes + 1) * 4);
    services::internal::service_memset<IdxType, cpu>(duplicates, 1, N);
    bottom = nNodes;

    // cache root data
    const DataType rootX = posX[nNodes];
    const DataType rootY = posY[nNodes];

    IdxType localMaxDepth = 1;
    maxDepth              = 1;
    IdxType skip          = 1;

    const IdxType inc = 1;
    IdxType i         = 0;

    // iterate over all bodies assigned to thread
    while (i < N)
    {
        if (skip != 0)
        {
            // new body, so start traversing at root
            skip  = 0;
            n     = nNodes;
            depth = 1;
            r     = radius * 0.5;

            /* Select child node 'j'
                          rootx < px  rootx > px
             * rooty < py   1 -> 3    0 -> 2
             * rooty > py   1 -> 1    0 -> 0
             */
            x = rootX + ((rootX < (px = posX[i])) ? (j = 1, r) : (j = 0, -r));

            y = rootY + ((rootY < (py = posY[i])) ? (j |= 2, r) : (-r));
        }

        // follow path to leaf cell
        while ((ch = child[n * 4 + j]) >= N)
        {
            n = ch;
            depth++;
            r *= 0.5;

            x += ((x < px) ? (j = 1, r) : (j = 0, -r));

            y += ((y < py) ? (j |= 2, r) : (-r));
        }

        // (ch)ild will be '-1' (nullptr), '-2' (locked), or an Integer corresponding to a body offset
        // in the lower [0, N) blocks of child
        if (ch != -2)
        {
            // skip if child pointer was locked when we examined it, and try again later.
            locked = n * 4 + j;
            // store the locked position in case we need to patch in a cell later.

            if (ch == -1)
            {
                // Child is a nullptr ('-1'), so we write our body index to the leaf, and move on to the next body.
                if (child[locked] == -1)
                {
                    child[locked] = i;
                    if (depth > localMaxDepth) localMaxDepth = depth;

                    i += inc; // move on to next body
                    skip = 1;
                }
            }
            else
            {
                // Child node isn't empty, so we store the current value of the child, lock the leaf, and patch in a new cell
                // Some points may be duplicated, so we count the number of duplicate points
                if (posX[i] - posX[ch] <= localEps && posY[i] - posY[ch] <= localEps && posX[i] - posX[ch] >= -localEps
                    && posY[i] - posY[ch] >= -localEps)
                {
                    duplicates[ch]++;
                    i += inc;
                    skip = 1;
                    continue;
                }
                if (child[locked] == ch)
                {
                    patch = -1;
                    while (ch >= 0)
                    {
                        depth++;

                        const IdxType cell = bottom - 1;
                        bottom += IdxType(-1);
                        if (cell == N)
                        {
                            bottom = nNodes;
                        }
                        else if (cell < N)
                        {
                            depth--;
                            continue;
                        }

                        if (patch != -1)
                        {
                            child[n * 4 + j] = cell;
                        }

                        if (cell > patch)
                        {
                            patch = cell;
                        }

                        // Insert migrated child node
                        j = (x < posX[ch]) ? 1 : 0;
                        if (y < posY[ch])
                        {
                            j |= 2;
                        }

                        child[cell * 4 + j] = ch;
                        n                   = cell;
                        r *= 0.5;

                        x += ((x < px) ? (j = 1, r) : (j = 0, -r));

                        y += ((y < py) ? (j |= 2, r) : (-r));

                        ch = child[n * 4 + j];
                        if (r <= radius * 1e-10)
                        {
                            break;
                        }
                    }

                    child[n * 4 + j] = i;

                    if (depth > localMaxDepth) localMaxDepth = depth;

                    i += inc; // move on to next body
                    skip = 2;
                }
            }
        }
        if (skip == 2)
        {
            child[locked] = patch;
        }
    }

    // record maximum tree depth
    if (localMaxDepth > 32) localMaxDepth = 32;

    maxDepth = (maxDepth < localMaxDepth) ? localMaxDepth : maxDepth;
    return services::Status();
}

template <typename IdxType, typename DataType, daal::CpuType cpu>
services::Status summarizationKernelImpl(IdxType * count, IdxType * child, DataType * mass, DataType * posX, DataType * posY, IdxType * duplicates,
                                         const IdxType nNodes, const IdxType N, const IdxType & bottom)
{
    bool flag = false;
    DataType cm, px, py;
    IdxType curChild[4];
    DataType curMass[4];

    const IdxType inc = 1;
    auto k            = bottom;

    //initialize array
    services::internal::service_memset<DataType, cpu>(mass, DataType(1), k);
    services::internal::service_memset<DataType, cpu>(&mass[k], DataType(-1), nNodes - k + 1);

    const auto restart = k;
    // iterate over all cells assigned to thread
    while (k <= nNodes)
    {
        if (mass[k] < 0.)
        {
            for (IdxType i = 0; i < 4; i++)
            {
                const auto ch = child[k * 4 + i];
                curChild[i]   = ch;
                if (ch >= 0) curMass[i] = mass[ch];
            }

            // all children are ready
            cm       = 0.;
            px       = 0.;
            py       = 0.;
            auto cnt = 0;

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
                    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(DataType, posX[ch], m);
                    DAAL_OVERFLOW_CHECK_BY_ADDING(DataType, px, posX[ch] * m);
                    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(DataType, posY[ch], m);
                    DAAL_OVERFLOW_CHECK_BY_ADDING(DataType, py, posY[ch] * m);
                    px += posX[ch] * m;
                    py += posY[ch] * m;
                }
            }
            count[k]         = cnt;
            const DataType m = cm ? DataType(1) / cm : DataType(1);

            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(DataType, px, m);
            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(DataType, py, m);
            posX[k] = px * m;
            posY[k] = py * m;

            mass[k] = cm;
        }

        k += inc; // move on to next cell
    }
    return services::Status();
}

template <typename IdxType, daal::CpuType cpu>
services::Status sortKernelImpl(IdxType * sort, const IdxType * count, IdxType * start, IdxType * child, const IdxType nNodes, const IdxType N,
                                const IdxType & bottom)
{
    //initialize array
    services::internal::service_memset<IdxType, cpu>(start, -1, nNodes);
    start[nNodes] = 0;

    const IdxType dec = 1;
    IdxType k         = nNodes;
    IdxType begin;
    IdxType limiter = 0;

    // iterate over all cells assigned to thread
    while (k >= bottom)
    {
        // To control possible infinite loops
        if (++limiter > nNodes) break;

        // Not a child so skip
        if ((begin = start[k]) < 0) continue;

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
                if (ch >= N)
                {
                    // child is a cell
                    start[ch] = begin;
                    begin += count[ch]; // add #bodies in subtree
                }
                else if (begin <= nNodes && begin >= 0)
                {
                    // child is a body
                    sort[begin++] = ch;
                }
                j++;
            }
        }
        k -= dec; // move on to next cell
    }
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

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(DataType, DataType(1), radiusSquared / thetaSquared);
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
                        DAAL_CHECK_THR(dxy1 * dxy1 != DataType(0), services::ErrorBufferSizeIntegerOverflow)
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

            gx =
                (dx * (ux = oldForceX[i]) < DataType(0)) ? gainX[i] + DataType(0.2) : gainX[i] * DataType(0.8);
            if (gx < DataType(0.01)) gx = DataType(0.01);

            gy =
                (dy * (uy = oldForceY[i]) < DataType(0)) ? gainY[i] + DataType(0.2) : gainY[i] * DataType(0.8);
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

        status = qTreeBuildingKernelImpl<IdxType, DataType, cpu>(child.get(), posX.get(), posY.get(), duplicates.get(), nNodes, N, maxDepth, bottom,
                                                                 radius);
        DAAL_CHECK_STATUS_VAR(status);

        status = summarizationKernelImpl<IdxType, DataType, cpu>(count.get(), child.get(), mass.get(), posX.get(), posY.get(), duplicates.get(),
                                                                 nNodes, N, bottom);
        DAAL_CHECK_STATUS_VAR(status);

        status = sortKernelImpl<IdxType, cpu>(sort.get(), count.get(), start.get(), child.get(), nNodes, N, bottom);
        DAAL_CHECK_STATUS_VAR(status);

        status = repulsionKernelImpl<IdxType, DataType, cpu>(theta, eps, sort.get(), child.get(), mass.get(), posX.get(), posY.get(), repX.get(),
                                                             repY.get(), zNorm, nNodes, N, radius, maxDepth, blockOfRows);
        DAAL_CHECK_STATUS_VAR(status);

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

        status = integrationKernelImpl<IdxType, DataType, cpu>(eta, momentum, exaggeration, posX.get(), posY.get(), attrX.get(), attrY.get(),
                                                               repX.get(), repY.get(), gainX.get(), gainY.get(), oldForceX.get(), oldForceY.get(),
                                                               gradNorm, zNorm, nNodes, N, blockOfRows);
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
        status = boundingBoxKernelImpl<IdxType, DataType, cpu>(posX.get(), posY.get(), N, nNodes, radius, blockOfRows);
        DAAL_CHECK_STATUS_VAR(status);

        status = qTreeBuildingKernelImpl<IdxType, DataType, cpu>(child.get(), posX.get(), posY.get(), duplicates.get(), nNodes, N, maxDepth, bottom,
                                                                 radius);
        DAAL_CHECK_STATUS_VAR(status);

        status = summarizationKernelImpl<IdxType, DataType, cpu>(count.get(), child.get(), mass.get(), posX.get(), posY.get(), duplicates.get(),
                                                                 nNodes, N, bottom);
        DAAL_CHECK_STATUS_VAR(status);

        status = sortKernelImpl<IdxType, cpu>(sort.get(), count.get(), start.get(), child.get(), nNodes, N, bottom);
        DAAL_CHECK_STATUS_VAR(status);

        status = repulsionKernelImpl<IdxType, DataType, cpu>(theta, eps, sort.get(), child.get(), mass.get(), posX.get(), posY.get(), repX.get(),
                                                             repY.get(), zNorm, nNodes, N, radius, maxDepth, blockOfRows);
        DAAL_CHECK_STATUS_VAR(status);

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
