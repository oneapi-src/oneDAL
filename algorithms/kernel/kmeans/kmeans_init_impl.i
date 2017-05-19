/* file: kmeans_init_impl.i */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of Lloyd method for K-means algorithm.
//--
*/

#include "algorithm.h"
#include "numeric_table.h"
#include "threading.h"
#include "daal_defines.h"
#include "service_memory.h"
#include "service_numeric_table.h"
#include "service_rng.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{
namespace internal
{

using namespace daal::services;
using namespace daal::internal;
using namespace daal::services::internal;

template <Method method, typename algorithmFPType, CpuType cpu>
Status init( size_t p, size_t n, size_t nRowsTotal, size_t nClusters, algorithmFPType *clusters,
    NumericTable *ntData, unsigned int seed, size_t& clustersFound)
{
    if(method == deterministicDense || method == deterministicCSR)
    {
        ReadRows<algorithmFPType, cpu> mtData(ntData, 0, nClusters);
        DAAL_CHECK_BLOCK_STATUS(mtData);
        const algorithmFPType *data = mtData.get();
        for (size_t i = 0; i < nClusters && i < n; i++)
        {
            for (size_t j = 0; j < p; j++)
            {
                clusters[i * p + j] = data[i * p + j];
            }
        }
        clustersFound = nClusters;
        return Status();
    }
    if(method == randomDense || method == randomCSR)
    {
        TArray<int, cpu> aIndices(nClusters);
        DAAL_CHECK(aIndices.get(), ErrorMemoryAllocationFailed);
        int *indices = aIndices.get();

        ReadRows<algorithmFPType, cpu> mtData;
        BaseRNGs<cpu> baseRng(seed);
        RNGs<int, cpu> rng;

        size_t k = 0;
        for(size_t i = 0; i < nClusters; i++)
        {
            DAAL_CHECK(rng.uniform(1, &indices[i], baseRng, i, (int)nRowsTotal) == 0, ErrorIncorrectErrorcodeFromGenerator);
            size_t c = (size_t)indices[i];
            int value = indices[i];
            for(size_t j = i; j > 0; j--)
            {
                if(value == indices[j-1])
                {
                    c = (size_t)(j-1);
                    value = c;
                }
            }
            if(c>=n )
                continue;
            const algorithmFPType* data = mtData.set(ntData, c, 1);
            DAAL_CHECK_BLOCK_STATUS(mtData);
            for(size_t j = 0; j < p; j++)
            {
                clusters[k * p + j] = data[j];
            }
            k++;
        }
        clustersFound = k;
        return Status();
    }
    DAAL_ASSERT(false && "should never happen");
    return Status();
}

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status KMeansinitKernel<method, algorithmFPType, cpu>::compute( size_t na, const NumericTable *const *a,
                                                                          size_t nr, const NumericTable *const *r, const Parameter *par)
{
    NumericTable *ntData     = const_cast<NumericTable *>( a[0] );
    NumericTable *ntClusters = const_cast<NumericTable *>( r[0] );

    const size_t p = ntData->getNumberOfColumns();
    const size_t n = ntData->getNumberOfRows();
    const size_t nClusters = par->nClusters;

    WriteOnlyRows<algorithmFPType, cpu> clustersBD(ntClusters, 0, nClusters);
    algorithmFPType *clusters = clustersBD.get();

    size_t clustersFound = 0;
    return init<method, algorithmFPType, cpu>(p, n, n, nClusters, clusters, ntData, par->seed, clustersFound);
}

template <typename algorithmFPType, CpuType cpu>
Status initDistrDeterministic(const NumericTable* pData, const Parameter *par, size_t& nClustersFound, NumericTablePtr& pRes)
{
    nClustersFound = 0;
    if(par->nClusters <= par->offset)
        return Status(); //ok

    nClustersFound = par->nClusters - par->offset;
    const size_t nRows = pData->getNumberOfRows();
    if(nClustersFound > nRows)
        nClustersFound = nRows;
    if(!pRes)
    {
        pRes.reset(new HomogenNumericTableCPU<algorithmFPType, cpu>(pData->getNumberOfColumns(), nClustersFound));
        DAAL_CHECK(pRes.get(), ErrorMemoryAllocationFailed);
    }

    WriteOnlyRows<algorithmFPType, cpu> resBD(*pRes, 0, nClustersFound);
    DAAL_CHECK_BLOCK_STATUS(resBD);
    ReadRows<algorithmFPType, cpu> dataBD(*const_cast<NumericTable*>(pData), 0, nClustersFound);
    DAAL_CHECK_BLOCK_STATUS(dataBD);
    const size_t sz = pData->getNumberOfColumns()*nClustersFound*sizeof(algorithmFPType);
    daal::services::daal_memcpy_s(resBD.get(), sz, dataBD.get(), sz);
    return Status();
}

template <typename algorithmFPType, CpuType cpu>
Status generateRandomIndices(const Parameter *par, size_t nRows, size_t& nClustersFound, int* clusters)
{
    TArray<int, cpu> aIndices(par->nClusters);
    int* indices = aIndices.get();
    DAAL_CHECK(indices, ErrorMemoryAllocationFailed);
    BaseRNGs<cpu> baseRng(par->seed);
    RNGs<int, cpu> rng;
    nClustersFound = 0;
    for(size_t i = 0; i < par->nClusters; i++)
    {
        DAAL_CHECK(rng.uniform(1, &indices[i], baseRng, i, (int)par->nRowsTotal) == 0, ErrorIncorrectErrorcodeFromGenerator);
        size_t c = (size_t)indices[i];
        int value = indices[i];
        for(size_t j = i; j > 0; j--)
        {
            if(value == indices[j - 1])
            {
                c = (size_t)(j - 1);
                value = c;
            }
        }

        if(c < par->offset || c >= par->offset + nRows)
            continue;
        clusters[nClustersFound] = c - par->offset;
        nClustersFound++;
    }
    return Status();
}

template <typename algorithmFPType, CpuType cpu>
Status initDistrRandom(const NumericTable* pData, const Parameter *par,
    size_t& nClustersFound, NumericTablePtr& pRes)
{
    TArray<int, cpu> clusters(par->nClusters);
    DAAL_CHECK(clusters.get(), ErrorMemoryAllocationFailed);
    Status s = generateRandomIndices<algorithmFPType, cpu>(par, pData->getNumberOfRows(), nClustersFound, clusters.get());
    if(!s)
        return s;

    if(!nClustersFound)
        return s; //ok

    const size_t p = pData->getNumberOfColumns();
    if(!pRes)
    {
        pRes.reset(new HomogenNumericTableCPU<algorithmFPType, cpu>(p, nClustersFound));
        DAAL_CHECK(pRes.get(), ErrorMemoryAllocationFailed);
    }

    WriteOnlyRows<algorithmFPType, cpu> resBD(*pRes, 0, nClustersFound);
    DAAL_CHECK_BLOCK_STATUS(resBD);

    auto aClusters = resBD.get();
    ReadRows<algorithmFPType, cpu> dataBD;
    for(size_t i = 0; i < nClustersFound; ++i)
    {
        auto pRow = dataBD.set(const_cast<NumericTable*>(pData), clusters.get()[i], 1);
        DAAL_CHECK_BLOCK_STATUS(dataBD);
        for(size_t j = 0; j < p; j++)
            aClusters[i * p + j] = pRow[j];
    }
    return s;
}

template <typename algorithmFPType, CpuType cpu>
Status initDistrPlusPlus(const NumericTable* pData, const Parameter *par,
    size_t& nClustersFound, NumericTablePtr& pRes)
{
    nClustersFound = 0;

    BaseRNGs<cpu> baseRng(par->seed);
    RNGs<int, cpu> rng;

    int index = 0;
    rng.uniform(1, &index, baseRng, 0, (int)par->nRowsTotal);
    size_t c(index);
    if(c < par->offset)
        return Status(); //ok
    if(c >= par->offset + pData->getNumberOfRows())
        return Status(); //ok

    ReadRows<algorithmFPType, cpu> dataBD(*const_cast<NumericTable*>(pData), c - par->offset, 1);
    DAAL_CHECK_BLOCK_STATUS(dataBD);

    if(!pRes)
    {
        pRes.reset(new HomogenNumericTableCPU<algorithmFPType, cpu>(pData->getNumberOfColumns(), 1));
        DAAL_CHECK(pRes.get(), ErrorMemoryAllocationFailed);
    }

    nClustersFound = 1;
    const size_t p = pData->getNumberOfColumns();
    WriteOnlyRows<algorithmFPType, cpu> resBD(*pRes, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(resBD);
    daal::services::daal_memcpy_s(resBD.get(), sizeof(algorithmFPType)*p, dataBD.get(), sizeof(algorithmFPType)*p);
    return Status();
}

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status KMeansinitStep1LocalKernel<method, algorithmFPType, cpu>::compute(const NumericTable* pData, const Parameter *par,
    NumericTable* pNumPartialClusters, NumericTablePtr& pPartialClusters)
{
    size_t nClustersFound = 0;
    services::Status s;
    if((method == deterministicDense) || (method == deterministicCSR))
        s = initDistrDeterministic<algorithmFPType, cpu>(pData, par, nClustersFound, pPartialClusters);
    else if((method == randomDense) || (method == randomCSR))
        s = initDistrRandom<algorithmFPType, cpu>(pData, par, nClustersFound, pPartialClusters);
    else if(isPlusPlusMethod(method))
        s = initDistrPlusPlus<algorithmFPType, cpu>(pData, par, nClustersFound, pPartialClusters);
    else
        DAAL_ASSERT(false && "should never happen");
    if(!s)
        return s;
    WriteOnlyRows<int, cpu> npcBD(*pNumPartialClusters, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(npcBD);
    *npcBD.get() = (int)nClustersFound;
    return s;
}

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status KMeansinitStep2MasterKernel<method, algorithmFPType, cpu>::finalizeCompute(size_t na, const NumericTable *const *a,
    NumericTable* ntClusters, const Parameter *par)
{
    const size_t nBlocks = na / 2;
    const size_t p = ntClusters->getNumberOfColumns();
    const size_t nClusters = par->nClusters;

    WriteOnlyRows<algorithmFPType, cpu> mtClusters(ntClusters, 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtClusters);
    algorithmFPType *clusters = mtClusters.get();

    size_t k = 0;
    for( size_t i = 0; i<nBlocks; i++ )
    {
        if(!a[i * 2 + 1])
            continue; //can be null
        ReadRows<int, cpu> mtInClustersN(*const_cast<NumericTable*>(a[i * 2 + 0]), 0, nClusters);
        DAAL_CHECK_BLOCK_STATUS(mtInClustersN);
        const int* inClustersN = mtInClustersN.get();
        ReadRows<algorithmFPType, cpu> mtInClusters(*const_cast<NumericTable*>(a[i * 2 + 1]), 0, 1);
        DAAL_CHECK_BLOCK_STATUS(mtInClusters);
        const algorithmFPType *inClusters = mtInClusters.get();

        const size_t inK = *inClustersN;
        for( size_t j=0; j<inK; j++ )
        {
            for( size_t h=0; h<p; h++ )
            {
                clusters[k*p + h] = inClusters[j*p + h];
            }
            k++;
        }
    }
    return Status();
}

} // namespace daal::algorithms::kmeans::init::internal
} // namespace daal::algorithms::kmeans::init
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
