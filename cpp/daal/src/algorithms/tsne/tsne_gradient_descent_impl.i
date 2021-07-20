/** file tsne_gradient_descent_impl.i */
/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef __INTERNAL_TSNE_GRADIENT_DESCENT_IMPL_I__
#define __INTERNAL_TSNE_GRADIENT_DESCENT_IMPL_I__

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
#include <iostream>
#include <ctime>
#include <chrono>
#include <fstream>

using namespace daal::data_management;
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace internal
{
template <typename IdxType, daal::CpuType cpu>
services::Status maxRowElementsImpl(const size_t * row, const IdxType N, IdxType & nElements)
{
    nElements                 = 0;
    const IdxType nThreads    = threader_get_threads_number();
    const IdxType nBlocks     = ((nThreads < N) ? nThreads : 1);
    const IdxType sizeOfBlock = N / nBlocks + !!(N % nBlocks);
    size_t * maxElements      = services::internal::service_calloc<size_t, cpu>(nThreads);
    DAAL_CHECK_MALLOC(maxElements);

    daal::threader_for(nBlocks, nBlocks, [&](IdxType iBlock) {
        const IdxType iStart = iBlock * sizeOfBlock;
        const IdxType iEnd   = (((iBlock + 1) * sizeOfBlock > N) ? N : iStart + sizeOfBlock);
        size_t max           = 0;
        for (IdxType i = iStart; i < iEnd; ++i)
        {
            max = services::internal::max<cpu, size_t>(max, row[i + 1] - row[i]);
        }
        maxElements[iBlock] = max;
    });

    for (IdxType i = 0; i < nBlocks; ++i)
    {
        nElements = services::internal::max<cpu, IdxType>(nElements, (IdxType)maxElements[i]);
    }

    services::internal::service_free<size_t, cpu>(maxElements);
    return services::Status();
}

template <typename IdxType, typename DataType, daal::CpuType cpu>
services::Status boundingBoxKernelImpl(DataType * posx, DataType * posy, const IdxType N, const IdxType nNodes, DataType & radius)
{
    DAAL_CHECK_MALLOC(posx);
    DAAL_CHECK_MALLOC(posy);

    const IdxType nThreads    = threader_get_threads_number();
    const IdxType nBlocks     = ((nThreads < N) ? nThreads : 1);
    const IdxType sizeOfBlock = N / nBlocks + !!(N % nBlocks);
    DataType * xMinValues     = services::internal::service_calloc<DataType, cpu>(nThreads);
    DataType * xMaxValues     = services::internal::service_calloc<DataType, cpu>(nThreads);
    DataType * yMinValues     = services::internal::service_calloc<DataType, cpu>(nThreads);
    DataType * yMaxValues     = services::internal::service_calloc<DataType, cpu>(nThreads);
    DAAL_CHECK_MALLOC(xMinValues);
    DAAL_CHECK_MALLOC(xMaxValues);
    DAAL_CHECK_MALLOC(yMinValues);
    DAAL_CHECK_MALLOC(yMaxValues);

    daal::threader_for(nBlocks, nBlocks, [&](IdxType iBlock) {
        const IdxType iStart = iBlock * sizeOfBlock;
        const IdxType iEnd   = (((iBlock + 1) * sizeOfBlock > N) ? N : iStart + sizeOfBlock);

        DataType xMin = posx[iStart];
        DataType xMax = posx[iStart];
        DataType yMin = posy[iStart];
        DataType yMax = posy[iStart];
        for (IdxType i = iStart + 1; i < iEnd; ++i)
        {
            xMin = services::internal::min<cpu, DataType>(xMin, posx[i]);
            xMax = services::internal::max<cpu, DataType>(xMax, posx[i]);
            yMin = services::internal::min<cpu, DataType>(yMin, posy[i]);
            yMax = services::internal::max<cpu, DataType>(yMax, posy[i]);
        }
        xMinValues[iBlock] = xMin;
        xMaxValues[iBlock] = xMax;
        yMinValues[iBlock] = yMin;
        yMaxValues[iBlock] = yMax;
    });
    DataType xMin = xMinValues[0];
    DataType xMax = xMaxValues[0];
    DataType yMin = yMinValues[0];
    DataType yMax = yMaxValues[0];

    for (IdxType i = 1; i < nBlocks; ++i)
    {
        xMin = services::internal::min<cpu, DataType>(xMin, xMinValues[i]);
        xMax = services::internal::max<cpu, DataType>(xMax, xMaxValues[i]);
        yMin = services::internal::min<cpu, DataType>(yMin, yMinValues[i]);
        yMax = services::internal::max<cpu, DataType>(yMax, yMaxValues[i]);
    }

    //scale the maximum to get all points strictly in the bounding box
    if (xMax >= 0.)
        xMax = services::internal::max<cpu, DataType>(xMax * (1. + 1e-3), xMax + 1e-3);
    else
        xMax = services::internal::max<cpu, DataType>(xMax * (1. - 1e-3), xMax + 1e-3);
    if (yMax >= 0.)
        yMax = services::internal::max<cpu, DataType>(yMax * (1. + 1e-3), yMax + 1e-3);
    else
        yMax = services::internal::max<cpu, DataType>(yMax * (1. - 1e-3), yMax + 1e-3);

    //save results
    //radius       = services::internal::max<cpu, DataType>(xMax - xMin, yMax - yMin) * 0.5 + 1e-5;
    radius       = services::internal::max<cpu, DataType>(xMax - xMin, yMax - yMin) * 0.5;
    posx[nNodes] = (xMin + xMax) * 0.5;
    posy[nNodes] = (yMin + yMax) * 0.5;

    //free memory
    services::internal::service_free<DataType, cpu>(xMinValues);
    services::internal::service_free<DataType, cpu>(xMaxValues);
    services::internal::service_free<DataType, cpu>(yMinValues);
    services::internal::service_free<DataType, cpu>(yMaxValues);

    return services::Status();
}

template <typename IdxType, typename DataType, daal::CpuType cpu>
services::Status qTreeBuildingKernelImpl(IdxType * child, const DataType * posx, const DataType * posy, const IdxType nNodes, const IdxType N,
                                         IdxType & maxDepth, IdxType & bottom, const DataType & radius)
{
    // internal variables
    IdxType j      = 0;
    IdxType depth  = 0;
    IdxType ch     = 0;
    IdxType n      = 0;
    IdxType locked = 0;
    IdxType patch  = 0;
    DataType x     = 0.;
    DataType y     = 0.;
    DataType r     = 0.;
    DataType px    = 0.;
    DataType py    = 0.;

    DAAL_CHECK_MALLOC(posx);
    DAAL_CHECK_MALLOC(posy);
    DAAL_CHECK_MALLOC(child);

    //initialize array
    services::internal::service_memset<IdxType, cpu>(child, -1, (nNodes + 1) * 4);
    bottom = nNodes;

    // cache root data
    const DataType rootx = posx[nNodes];
    const DataType rooty = posy[nNodes];

    IdxType localmaxDepth = 1;
    IdxType skip          = 1;

    const IdxType n_threads = 1;
    const IdxType i_thread  = 0;
    const IdxType inc       = n_threads;
    IdxType i               = i_thread;

    // iterate over all bodies assigned to thread
    while (i < N)
    {
        if (skip != 0)
        {
            // new body, so start traversing at root
            skip  = 0;
            n     = nNodes;
            depth = 1;
            r     = radius * 0.5f;

            /* Select child node 'j'
                          rootx < px  rootx > px
             * rooty < py   1 -> 3    0 -> 2
             * rooty > py   1 -> 1    0 -> 0
             */
            x = rootx + ((rootx < (px = posx[i])) ? (j = 1, r) : (j = 0, -r));

            y = rooty + ((rooty < (py = posy[i])) ? (j |= 2, r) : (-r));
        }

        // follow path to leaf cell
        while ((ch = child[n * 4 + j]) >= N)
        {
            n = ch;
            depth++;
            r *= 0.5f;

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
                    if (depth > localmaxDepth) localmaxDepth = depth;

                    i += inc; // move on to next body
                    skip = 1;
                }
            }
            else
            {
                // Child node isn't empty, so we store the current value of the child, lock the leaf, and patch in a new cell
                if (child[locked] == ch)
                {
                    child[locked] = -2;
                    patch         = -1;

                    while (ch >= 0)
                    {
                        depth++;

                        const IdxType cell = bottom - 1;
                        bottom += ((IdxType)-1);
                        if (cell == N)
                        {
                            bottom = nNodes;
                        }
                        else if (cell < N)
                        {
                            depth--;
                            continue;
                        }

                        if (patch != -1) child[n * 4 + j] = cell;

                        if (cell > patch) patch = cell;

                        // Insert migrated child node
                        j = (x < posx[ch]) ? 1 : 0;
                        if (y < posy[ch]) j |= 2;

                        child[cell * 4 + j] = ch;
                        n                   = cell;
                        r *= 0.5f;

                        x += ((x < px) ? (j = 1, r) : (j = 0, -r));

                        y += ((y < py) ? (j |= 2, r) : (-r));

                        ch = child[n * 4 + j];
                        if (r <= 1e-10)
                        {
                            break;
                        }
                    }

                    child[n * 4 + j] = i;

                    if (depth > localmaxDepth) localmaxDepth = depth;

                    i += inc; // move on to next body
                    skip = 2;
                }
            }
        }

        if (skip == 2) child[locked] = patch;
    }

    // record maximum tree depth
    if (localmaxDepth > 32) localmaxDepth = 32;

    maxDepth = (maxDepth < localmaxDepth) ? localmaxDepth : maxDepth;
    return services::Status();
}

template <typename IdxType, typename DataType, daal::CpuType cpu>
services::Status summarizationKernelImpl(IdxType * count, IdxType * child, DataType * mass, DataType * posx, DataType * posy, const IdxType nNodes,
                                         const IdxType N, const IdxType & bottom)
{
    DAAL_CHECK_MALLOC(count);
    DAAL_CHECK_MALLOC(child);
    DAAL_CHECK_MALLOC(mass);
    DAAL_CHECK_MALLOC(posx);
    DAAL_CHECK_MALLOC(posy);

    bool flag = 0;
    DataType cm, px, py;
    IdxType curChild[4];
    DataType curMass[4];

    const auto inc = 1;
    auto k         = bottom;

    //initialize array
    services::internal::service_memset<DataType, cpu>(mass, 1., k);
    services::internal::service_memset<DataType, cpu>(&mass[k], -1., nNodes - k + 1);

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
                    const DataType m = (ch >= N) ? (cnt += count[ch], curMass[i]) : (cnt++, mass[ch]);
                    // add child's contribution
                    cm += m;
                    px += posx[ch] * m;
                    py += posy[ch] * m;
                }
            }

            count[k]         = cnt;
            const DataType m = cm ? 1. / cm : 0.;
            posx[k]          = px * m;
            posy[k]          = py * m;

            mass[k] = cm;
        }

        k += inc; // move on to next cell
    }
    k = restart;

    IdxType j = 0;
    // iterate over all cells assigned to thread
    while (k <= nNodes)
    {
        if (j == 0)
        {
            j = 4;
            for (IdxType i = 0; i < 4; i++)
            {
                const auto ch = child[k * 4 + i];

                curChild[i] = ch;
                if ((ch < N) or ((curMass[i] = mass[ch]) >= 0)) j--;
            }
        }
        else
        {
            j = 4;
            for (IdxType i = 0; i < 4; i++)
            {
                const auto ch = curChild[i];

                if ((ch < N) or (curMass[i] >= 0) or ((curMass[i] = mass[ch]) >= 0)) j--;
            }
        }

        if (j == 0)
        {
            // all children are ready
            cm       = 0.;
            px       = 0.;
            py       = 0.;
            auto cnt = 0;

            for (IdxType i = 0; i < 4; i++)
            {
                const auto ch = curChild[i];
                if (ch >= 0)
                {
                    const auto m = (ch >= N) ? (cnt += count[ch], curMass[i]) : (cnt++, mass[ch]);
                    // add child's contribution
                    cm += m;
                    px += posx[ch] * m;
                    py += posy[ch] * m;
                }
            }

            count[k]         = cnt;
            const DataType m = 1. / cm;
            posx[k]          = px * m;
            posy[k]          = py * m;
            flag             = 1;
        }

        if (flag != 0)
        {
            mass[k] = cm;
            k += inc;
            flag = 0;
        }
    }
    return services::Status();
}

template <typename IdxType, daal::CpuType cpu>
services::Status sortKernelImpl(IdxType * sort, const IdxType * count, IdxType * start, IdxType * child, const IdxType nNodes, const IdxType N,
                                const IdxType & bottom)
{
    DAAL_CHECK_MALLOC(count);
    DAAL_CHECK_MALLOC(sort);
    DAAL_CHECK_MALLOC(start);
    DAAL_CHECK_MALLOC(child);

    //initialize array
    services::internal::service_memset<IdxType, cpu>(start, -1, nNodes);
    start[nNodes] = 0;

    const IdxType dec = 1;
    IdxType k         = nNodes + 1 - dec;
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
                else if (begin <= nNodes and begin >= 0)
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
                                     const DataType * posx, const DataType * posy, DataType * repx, DataType * repy, DataType & zNorm,
                                     const IdxType nNodes, const IdxType N, const DataType & radius, const IdxType & maxDepth)
{
    DAAL_CHECK_MALLOC(sort);
    DAAL_CHECK_MALLOC(child);
    DAAL_CHECK_MALLOC(mass);
    DAAL_CHECK_MALLOC(posx);
    DAAL_CHECK_MALLOC(posy);
    DAAL_CHECK_MALLOC(repx);
    DAAL_CHECK_MALLOC(repy);

    //initialize arrays
    services::internal::service_memset<DataType, cpu>(repx, 0., nNodes + 1);
    services::internal::service_memset<DataType, cpu>(repy, 0., nNodes + 1);
    zNorm = 0.;

    const IdxType fourNNodes      = 4 * nNodes;
    const DataType thetaSquared   = theta * theta;
    const DataType radiusdSquared = radius * radius;
    const DataType EPS_PLUS_1     = eps + 1.;
    DataType * dq                 = services::internal::service_calloc<DataType, cpu>(maxDepth);

    daal::StaticTlsSum<DataType, cpu> sumTlsData(1);
    daal::static_tls<IdxType *> posTlsData([=]() { return services::internal::service_scalable_calloc<IdxType, cpu>(maxDepth); });
    daal::static_tls<IdxType *> nodeTlsData([=]() { return services::internal::service_scalable_calloc<IdxType, cpu>(maxDepth); });
    const IdxType nThreads    = posTlsData.nthreads();
    const IdxType sizeOfBlock = services::internal::min<cpu, IdxType>(256, N / nThreads + 1);
    const IdxType nBlocks     = N / sizeOfBlock + !!(N % sizeOfBlock);

    dq[0] = radiusdSquared / thetaSquared;
    for (auto i = 1; i < maxDepth; i++)
    {
        dq[i] = dq[i - 1] * 0.25;
        dq[i - 1] += eps;
    }
    dq[maxDepth - 1] += eps;

    // Add one so EPS_PLUS_1 can be compared
    for (auto i = 0; i < maxDepth; i++) dq[i] += 1.;

    // iterate over all bodies assigned to thread
    const auto MAX_SIZE = fourNNodes + 4;

    daal::static_threader_for(nBlocks, [&](IdxType iBlock, IdxType tid) {
        const IdxType iStart = iBlock * sizeOfBlock;
        const IdxType iEnd   = services::internal::min<cpu, IdxType>(N, iStart + sizeOfBlock);

        IdxType * pos       = posTlsData.local(tid);
        IdxType * node      = nodeTlsData.local(tid);
        DataType * localSum = sumTlsData.local(tid);
        for (IdxType k = iStart; k < iEnd; ++k)
        {
            const auto i = sort[k];

            const DataType px = posx[i];
            const DataType py = posy[i];

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
                    if (index < 0 or index >= MAX_SIZE) break;

                    const auto n = child[index]; // load child pointer

                    // Non child
                    if (n < 0 or n > nNodes) break;

                    const DataType dx   = px - posx[n];
                    const DataType dy   = py - posy[n];
                    const DataType dxy1 = dx * dx + dy * dy + EPS_PLUS_1;

                    if ((n < N) or (dxy1 >= dq[depth]))
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
            repx[i] += vx;
            repy[i] += vy;
        }
    });
    posTlsData.reduce([&](IdxType * buf) { services::internal::service_scalable_free<IdxType, cpu>(buf); });
    nodeTlsData.reduce([&](IdxType * buf) { services::internal::service_scalable_free<IdxType, cpu>(buf); });
    sumTlsData.reduceTo(&zNorm, 1);
    services::internal::service_free<DataType, cpu>(dq);

    return services::Status();
}

template <bool DivComp, typename IdxType, typename DataType, daal::CpuType cpu>
services::Status attractiveKernelImpl(const DataType * val, const size_t * col, const size_t * row, const DataType * posx, const DataType * posy,
                                      DataType * attrx, DataType * attry, DataType & zNorm, DataType & divergence, const IdxType nNodes,
                                      const IdxType N, const IdxType nnz, const IdxType nElements, const DataType exaggeration)
{
    DAAL_CHECK_MALLOC(val);
    DAAL_CHECK_MALLOC(col);
    DAAL_CHECK_MALLOC(row);
    DAAL_CHECK_MALLOC(posx);
    DAAL_CHECK_MALLOC(posy);
    DAAL_CHECK_MALLOC(attrx);
    DAAL_CHECK_MALLOC(attry);

    //initialize arrays
    services::internal::service_memset<DataType, cpu>(attrx, 0., N);
    services::internal::service_memset<DataType, cpu>(attry, 0., N);

    const DataType multiplier = exaggeration * (DataType)zNorm;
    divergence                = 0.;

    daal::StaticTlsSum<DataType, cpu> divTlsData(1);
    daal::static_tls<DataType *> logTlsData([=]() { return services::internal::service_scalable_calloc<DataType, cpu>(nElements); });

    const size_t nThreads    = logTlsData.nthreads();
    const size_t sizeOfBlock = services::internal::min<cpu, size_t>(256, N / nThreads + 1);
    const size_t nBlocks     = (size_t)N / sizeOfBlock + !!((size_t)N % sizeOfBlock);

    daal::static_threader_for(nBlocks, [&](size_t iBlock, size_t tid) {
        const size_t iStart = iBlock * sizeOfBlock;
        const size_t iEnd   = services::internal::min<cpu, size_t>((size_t)N, iStart + sizeOfBlock);
        DataType * log      = logTlsData.local(tid);
        DataType * div      = divTlsData.local(tid);
        for (size_t iRow = iStart; iRow < iEnd; ++iRow)
        {
            size_t iSize = 0;
            for (size_t index = row[iRow] - 1; index < row[iRow + 1] - 1; ++index)
            {
                const size_t iCol = col[index] - 1;

                const DataType y1d              = posx[iRow] - posx[iCol];
                const DataType y2d              = posy[iRow] - posy[iCol];
                DataType squared_euclidean_dist = y1d * y1d + y2d * y2d;
                if (!(squared_euclidean_dist >= 0)) squared_euclidean_dist = 0.;
                const DataType PQ = val[index] / (squared_euclidean_dist + 1.);

                // Apply forces
                attrx[iRow] += PQ * (posx[iRow] - posx[iCol]);
                attry[iRow] += PQ * (posy[iRow] - posy[iCol]);
                if (DivComp)
                {
                    log[iSize++] = val[index] * multiplier * (1. + squared_euclidean_dist);
                }
            }
            if (DivComp)
            {
                Math<DataType, cpu>::vLog(iSize, log, log);
                size_t start = row[iRow] - 1;
                for (size_t index = 0; index < iSize; ++index)
                {
                    div[0] += val[start + index] * log[index];
                }
            }
        }
    });
    divTlsData.reduceTo(&divergence, 1);
    divergence *= exaggeration;

    logTlsData.reduce([&](DataType * buf) { services::internal::service_scalable_free<DataType, cpu>(buf); });

    //Find_Normalization
    zNorm = 1. / (zNorm - N);

    return services::Status();
}

template <typename IdxType, typename DataType, daal::CpuType cpu>
services::Status integrationKernelImpl(const DataType eta, const DataType momentum, const DataType exaggeration, DataType * posx, DataType * posy,
                                       const DataType * attrx, const DataType * attry, const DataType * repx, const DataType * repy, DataType * gainx,
                                       DataType * gainy, DataType * oldForcex, DataType * oldForcey, DataType & gradNorm, const DataType & zNorm,
                                       const IdxType nNodes, const IdxType N)
{
    DAAL_CHECK_MALLOC(attrx);
    DAAL_CHECK_MALLOC(attry);
    DAAL_CHECK_MALLOC(repx);
    DAAL_CHECK_MALLOC(repy);
    DAAL_CHECK_MALLOC(posx);
    DAAL_CHECK_MALLOC(posy);
    DAAL_CHECK_MALLOC(gainx);
    DAAL_CHECK_MALLOC(gainy);
    DAAL_CHECK_MALLOC(oldForcex);
    DAAL_CHECK_MALLOC(oldForcey);

    typedef daal::services::internal::SignBit<DataType, cpu> SignBitType;

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
            const DataType dx = exaggeration * attrx[i] - zNorm * repx[i];
            const DataType dy = exaggeration * attry[i] - zNorm * repy[i];
            localSum[0] += dx * dx + dy * dy;

            gx = (SignBitType::get(dx) != SignBitType::get(ux = oldForcex[i])) ? gainx[i] + 0.2 : gainx[i] * 0.8;
            if (gx < 0.01) gx = 0.01;

            gy = (SignBitType::get(dy) != SignBitType::get(uy = oldForcey[i])) ? gainy[i] + 0.2 : gainy[i] * 0.8;
            if (gy < 0.01) gy = 0.01;

            gainx[i] = gx;
            gainy[i] = gy;

            oldForcex[i] = ux = momentum * ux - 4. * eta * gx * dx;
            oldForcey[i] = uy = momentum * uy - 4. * eta * gy * dy;

            posx[i] += ux;
            posy[i] += uy;
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
    auto begin = std::chrono::high_resolution_clock::now();

    // sizes and number of iterations
    daal::internal::ReadColumns<IdxType, cpu> sizeIterDataBlock(*sizeIterTable, 0, 0, 4);
    const IdxType * sizeIter = sizeIterDataBlock.get();
    DAAL_CHECK_MALLOC(sizeIter);
    const IdxType N                    = sizeIter[0];
    const IdxType nnz                  = sizeIter[1];
    const IdxType nIterWithoutProgress = sizeIter[2];
    const IdxType maxIter              = sizeIter[3];
    const IdxType nNodes               = N * 2;
    const IdxType nIterCheck           = 50;
    const IdxType explorationIter      = 250;
    const IdxType verbose              = 0;

    // parameters
    daal::internal::ReadColumns<DataType, cpu> paramDataBlock(*paramTable, 0, 0, 4);
    const DataType * params = paramDataBlock.get();
    DAAL_CHECK_MALLOC(params);
    const DataType eps         = 0.0025;
    DataType momentum          = 0.5;
    DataType exaggeration      = params[0];
    const DataType eta         = params[1];
    const DataType minGradNorm = params[2];
    const DataType theta       = params[3];

    // results
    daal::internal::WriteColumns<DataType, cpu> resultDataBlock(*resultTable, 0, 0, 3);
    DataType * results = resultDataBlock.get();
    DAAL_CHECK_MALLOC(results);
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

    // allocate and init memory for auxiliary arrays: posx & posy
    DataType * posx = services::internal::service_scalable_calloc<DataType, cpu>(nNodes + 1);
    DAAL_CHECK_MALLOC(posx);
    services::internal::tmemcpy<DataType, cpu>(posx, xInit, N);
    DataType * posy = services::internal::service_scalable_calloc<DataType, cpu>(nNodes + 1);
    DAAL_CHECK_MALLOC(posy);
    services::internal::tmemcpy<DataType, cpu>(posy, yInit, N);

    // allocate and init memory for auxiliary arrays
    IdxType * child = services::internal::service_scalable_calloc<IdxType, cpu>((nNodes + 1) * 4);
    DAAL_CHECK_MALLOC(child);
    IdxType * count = services::internal::service_scalable_calloc<IdxType, cpu>(nNodes + 1);
    DAAL_CHECK_MALLOC(count);
    DataType * mass = services::internal::service_scalable_calloc<DataType, cpu>(nNodes + 1);
    DAAL_CHECK_MALLOC(mass);
    IdxType * sort = services::internal::service_scalable_calloc<IdxType, cpu>(nNodes + 1);
    DAAL_CHECK_MALLOC(sort);
    IdxType * start = services::internal::service_scalable_calloc<IdxType, cpu>(nNodes + 1);
    DAAL_CHECK_MALLOC(start);
    DataType * repx = services::internal::service_scalable_calloc<DataType, cpu>(nNodes + 1);
    DAAL_CHECK_MALLOC(repx);
    DataType * repy = services::internal::service_scalable_calloc<DataType, cpu>(nNodes + 1);
    DAAL_CHECK_MALLOC(repy);
    DataType * attrx = services::internal::service_scalable_calloc<DataType, cpu>(N);
    DAAL_CHECK_MALLOC(attrx);
    DataType * attry = services::internal::service_scalable_calloc<DataType, cpu>(N);
    DAAL_CHECK_MALLOC(attry);
    DataType * gainx = services::internal::service_scalable_calloc<DataType, cpu>(N);
    DAAL_CHECK_MALLOC(gainx);
    DataType * gainy = services::internal::service_scalable_calloc<DataType, cpu>(N);
    DAAL_CHECK_MALLOC(gainy);
    DataType * oldForcex = services::internal::service_scalable_calloc<DataType, cpu>(N);
    DAAL_CHECK_MALLOC(oldForcex);
    DataType * oldForcey = services::internal::service_scalable_calloc<DataType, cpu>(N);
    DAAL_CHECK_MALLOC(oldForcey);

    auto init = std::chrono::high_resolution_clock::now();
    if (verbose == 1) std::cout << "Alloc and init gradient time = " << std::chrono::duration<double, std::milli>(init - begin).count() << std::endl;

    double boundingBox   = 0.;
    double treeBuild     = 0.;
    double summarization = 0.;
    double sorting       = 0.;
    double repulsion     = 0.;
    double attractive    = 0.;
    double integration   = 0.;

    status = maxRowElementsImpl<IdxType, cpu>(row, N, nElements);
    DAAL_CHECK_STATUS_VAR(status);

    //start iterations
    for (IdxType i = 0; i < explorationIter; ++i)
    {
        auto kernel0 = std::chrono::high_resolution_clock::now();
        status       = boundingBoxKernelImpl<IdxType, DataType, cpu>(posx, posy, N, nNodes, radius);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel1 = std::chrono::high_resolution_clock::now();
        boundingBox += std::chrono::duration<double, std::milli>(kernel1 - kernel0).count();

        status = qTreeBuildingKernelImpl<IdxType, DataType, cpu>(child, posx, posy, nNodes, N, maxDepth, bottom, radius);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel2 = std::chrono::high_resolution_clock::now();
        treeBuild += std::chrono::duration<double, std::milli>(kernel2 - kernel1).count();

        status = summarizationKernelImpl<IdxType, DataType, cpu>(count, child, mass, posx, posy, nNodes, N, bottom);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel3 = std::chrono::high_resolution_clock::now();
        summarization += std::chrono::duration<double, std::milli>(kernel3 - kernel2).count();

        status = sortKernelImpl<IdxType, cpu>(sort, count, start, child, nNodes, N, bottom);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel4 = std::chrono::high_resolution_clock::now();
        sorting += std::chrono::duration<double, std::milli>(kernel4 - kernel3).count();

        status =
            repulsionKernelImpl<IdxType, DataType, cpu>(theta, eps, sort, child, mass, posx, posy, repx, repy, zNorm, nNodes, N, radius, maxDepth);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel5 = std::chrono::high_resolution_clock::now();
        repulsion += std::chrono::duration<double, std::milli>(kernel5 - kernel4).count();

        if (((i + 1) % nIterCheck == 0) || (i == explorationIter - 1))
        {
            status = attractiveKernelImpl<true, IdxType, DataType, cpu>(val, col, row, posx, posy, attrx, attry, zNorm, divergence, nNodes, N, nnz,
                                                                        nElements, exaggeration);
        }
        else
        {
            status = attractiveKernelImpl<false, IdxType, DataType, cpu>(val, col, row, posx, posy, attrx, attry, zNorm, divergence, nNodes, N, nnz,
                                                                         nElements, exaggeration);
        }
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel6 = std::chrono::high_resolution_clock::now();
        attractive += std::chrono::duration<double, std::milli>(kernel6 - kernel5).count();

        status = integrationKernelImpl<IdxType, DataType, cpu>(eta, momentum, exaggeration, posx, posy, attrx, attry, repx, repy, gainx, gainy,
                                                               oldForcex, oldForcey, gradNorm, zNorm, nNodes, N);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel7 = std::chrono::high_resolution_clock::now();
        integration += std::chrono::duration<double, std::milli>(kernel7 - kernel6).count();
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
        auto kernel0 = std::chrono::high_resolution_clock::now();
        status       = boundingBoxKernelImpl<IdxType, DataType, cpu>(posx, posy, N, nNodes, radius);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel1 = std::chrono::high_resolution_clock::now();
        boundingBox += std::chrono::duration<double, std::milli>(kernel1 - kernel0).count();

        status = qTreeBuildingKernelImpl<IdxType, DataType, cpu>(child, posx, posy, nNodes, N, maxDepth, bottom, radius);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel2 = std::chrono::high_resolution_clock::now();
        treeBuild += std::chrono::duration<double, std::milli>(kernel2 - kernel1).count();

        status = summarizationKernelImpl<IdxType, DataType, cpu>(count, child, mass, posx, posy, nNodes, N, bottom);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel3 = std::chrono::high_resolution_clock::now();
        summarization += std::chrono::duration<double, std::milli>(kernel3 - kernel2).count();

        status = sortKernelImpl<IdxType, cpu>(sort, count, start, child, nNodes, N, bottom);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel4 = std::chrono::high_resolution_clock::now();
        sorting += std::chrono::duration<double, std::milli>(kernel4 - kernel3).count();

        status =
            repulsionKernelImpl<IdxType, DataType, cpu>(theta, eps, sort, child, mass, posx, posy, repx, repy, zNorm, nNodes, N, radius, maxDepth);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel5 = std::chrono::high_resolution_clock::now();
        repulsion += std::chrono::duration<double, std::milli>(kernel5 - kernel4).count();

        if (((i + 1) % nIterCheck == 0) || (i == maxIter - 1))
        {
            status = attractiveKernelImpl<true, IdxType, DataType, cpu>(val, col, row, posx, posy, attrx, attry, zNorm, divergence, nNodes, N, nnz,
                                                                        nElements, exaggeration);
        }
        else
        {
            status = attractiveKernelImpl<false, IdxType, DataType, cpu>(val, col, row, posx, posy, attrx, attry, zNorm, divergence, nNodes, N, nnz,
                                                                         nElements, exaggeration);
        }
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel6 = std::chrono::high_resolution_clock::now();
        attractive += std::chrono::duration<double, std::milli>(kernel6 - kernel5).count();

        status = integrationKernelImpl<IdxType, DataType, cpu>(eta, momentum, exaggeration, posx, posy, attrx, attry, repx, repy, gainx, gainy,
                                                               oldForcex, oldForcey, gradNorm, zNorm, nNodes, N);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel7 = std::chrono::high_resolution_clock::now();
        integration += std::chrono::duration<double, std::milli>(kernel7 - kernel6).count();
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
    services::internal::tmemcpy<DataType, cpu>(xInit, posx, N);
    services::internal::tmemcpy<DataType, cpu>(yInit, posy, N);

    //release block
    status = pTable->releaseSparseBlock(CSRBlock);
    DAAL_CHECK_STATUS_VAR(status);

    auto end = std::chrono::high_resolution_clock::now();
    if (verbose == 1)
    {
        std::cout << "Full gradient time = " << std::chrono::duration<double, std::milli>(end - begin).count() << std::endl;
        std::cout << "     BoundingBox time = " << boundingBox << std::endl;
        std::cout << "     TreeBuilding time = " << treeBuild << std::endl;
        std::cout << "     Summarization time = " << summarization << std::endl;
        std::cout << "     Sort time = " << sorting << std::endl;
        std::cout << "     Repulsion time = " << repulsion << std::endl;
        std::cout << "     Attractive time = " << attractive << std::endl;
        std::cout << "     Integration time = " << integration << std::endl;
        std::cout << "     Divergence = " << divergence << std::endl;
        std::cout << "     Gradient norm = " << gradNorm << std::endl;
        std::cout << "     Last iteration = " << curIter << std::endl;
    }

    return services::Status();
}

} // namespace internal
} // namespace algorithms
} // namespace daal

#endif
