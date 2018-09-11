/* file: kmeans_init_impl.i */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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
#include "uniform_kernel.h"
#include "uniform_impl.i"

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
using namespace daal::algorithms::distributions::uniform::internal;

template <Method method, typename algorithmFPType, CpuType cpu>
Status init( size_t p, size_t n, size_t nRowsTotal, size_t nClusters, algorithmFPType *clusters,
    NumericTable *ntData, unsigned int seed, engines::BatchBase &engine, size_t& clustersFound)
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

        size_t k = 0;
        Status s;
        for(size_t i = 0; i < nClusters; i++)
        {
            DAAL_CHECK_STATUS(s, (UniformKernelDefault<int, cpu>::compute(i, (int)nRowsTotal, engine, 1, indices + i)));
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
        return s;
    }
    DAAL_ASSERT(false && "should never happen");
    return Status();
}

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status KMeansinitKernel<method, algorithmFPType, cpu>::compute( size_t na, const NumericTable *const *a,
                                                                          size_t nr, const NumericTable *const *r, const Parameter *par, engines::BatchBase &engine)
{
    NumericTable *ntData     = const_cast<NumericTable *>( a[0] );
    NumericTable *ntClusters = const_cast<NumericTable *>( r[0] );

    const size_t p = ntData->getNumberOfColumns();
    const size_t n = ntData->getNumberOfRows();
    const size_t nClusters = par->nClusters;

    WriteOnlyRows<algorithmFPType, cpu> clustersBD(ntClusters, 0, nClusters);
    algorithmFPType *clusters = clustersBD.get();

    size_t clustersFound = 0;
    return init<method, algorithmFPType, cpu>(p, n, n, nClusters, clusters, ntData, par->seed, engine, clustersFound);
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
    Status st;
    if(!pRes)
    {
        pRes = HomogenNumericTableCPU<algorithmFPType, cpu>::create(pData->getNumberOfColumns(), nClustersFound, &st);
        DAAL_CHECK_STATUS_VAR(st);
    }

    WriteOnlyRows<algorithmFPType, cpu> resBD(*pRes, 0, nClustersFound);
    DAAL_CHECK_BLOCK_STATUS(resBD);
    ReadRows<algorithmFPType, cpu> dataBD(*const_cast<NumericTable*>(pData), 0, nClustersFound);
    DAAL_CHECK_BLOCK_STATUS(dataBD);
    const size_t sz = pData->getNumberOfColumns()*nClustersFound*sizeof(algorithmFPType);
    daal::services::daal_memcpy_s(resBD.get(), sz, dataBD.get(), sz);
    return st;
}

template <typename algorithmFPType, CpuType cpu>
Status generateRandomIndices(const Parameter *par, size_t nRows, size_t& nClustersFound, int* clusters, engines::BatchBase &engine)
{
    TArray<int, cpu> aIndices(par->nClusters);
    int* indices = aIndices.get();
    DAAL_CHECK(indices, ErrorMemoryAllocationFailed);
    nClustersFound = 0;

    Status s;
    for(size_t i = 0; i < par->nClusters; i++)
    {
        DAAL_CHECK_STATUS(s, (UniformKernelDefault<int, cpu>::compute(i, (int)par->nRowsTotal, engine, 1, indices + i)));
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
    return s;
}

template <typename algorithmFPType, CpuType cpu>
Status initDistrRandom(const NumericTable* pData, const Parameter *par,
    size_t& nClustersFound, NumericTablePtr& pRes, engines::BatchBase &engine)
{
    TArray<int, cpu> clusters(par->nClusters);
    DAAL_CHECK(clusters.get(), ErrorMemoryAllocationFailed);
    Status s = generateRandomIndices<algorithmFPType, cpu>(par, pData->getNumberOfRows(), nClustersFound, clusters.get(), engine);
    if(!s)
        return s;

    if(!nClustersFound)
        return s; //ok

    const size_t p = pData->getNumberOfColumns();
    if(!pRes)
    {
        pRes = HomogenNumericTableCPU<algorithmFPType, cpu>::create(p, nClustersFound, &s);
        DAAL_CHECK_STATUS_VAR(s);
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
    size_t& nClustersFound, NumericTablePtr& pRes, engines::BatchBase &engine)
{
    nClustersFound = 0;
    int index = 0;

    Status s;
    DAAL_CHECK_STATUS(s, (UniformKernelDefault<int, cpu>::compute(0, (int)par->nRowsTotal, engine, 1, &index)));

    size_t c(index);
    if(c < par->offset)
        return Status(); //ok
    if(c >= par->offset + pData->getNumberOfRows())
        return Status(); //ok

    ReadRows<algorithmFPType, cpu> dataBD(*const_cast<NumericTable*>(pData), c - par->offset, 1);
    DAAL_CHECK_BLOCK_STATUS(dataBD);

    if(!pRes)
    {
        pRes = HomogenNumericTableCPU<algorithmFPType, cpu>::create(pData->getNumberOfColumns(), 1, &s);
        DAAL_CHECK_STATUS_VAR(s);
    }

    nClustersFound = 1;
    const size_t p = pData->getNumberOfColumns();
    WriteOnlyRows<algorithmFPType, cpu> resBD(*pRes, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(resBD);
    daal::services::daal_memcpy_s(resBD.get(), sizeof(algorithmFPType)*p, dataBD.get(), sizeof(algorithmFPType)*p);
    return s;
}

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status KMeansinitStep1LocalKernel<method, algorithmFPType, cpu>::compute(const NumericTable* pData, const Parameter *par,
    NumericTable* pNumPartialClusters, NumericTablePtr& pPartialClusters, engines::BatchBase &engine)
{
    size_t nClustersFound = 0;
    services::Status s;
    if((method == deterministicDense) || (method == deterministicCSR))
        s = initDistrDeterministic<algorithmFPType, cpu>(pData, par, nClustersFound, pPartialClusters);
    else if((method == randomDense) || (method == randomCSR))
        s = initDistrRandom<algorithmFPType, cpu>(pData, par, nClustersFound, pPartialClusters, engine);
    else if(isPlusPlusMethod(method))
        s = initDistrPlusPlus<algorithmFPType, cpu>(pData, par, nClustersFound, pPartialClusters, engine);
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
