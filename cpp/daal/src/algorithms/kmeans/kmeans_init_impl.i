/* file: kmeans_init_impl.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "src/threading/threading.h"
#include "services/daal_defines.h"
#include "src/externals/service_memory.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/distributions/uniform/uniform_kernel.h"
#include "src/algorithms/distributions/uniform/uniform_impl.i"
#include "src/services/service_data_utils.h"
#include <iostream>

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
Status init(size_t p, size_t n, size_t nRowsTotal, size_t nClusters, algorithmFPType * clusters, NumericTable * ntData, unsigned int seed,
            engines::BatchBase & engine, size_t & clustersFound)
{
    std::cout << "init begin" << std::endl;
    if (method == deterministicDense || method == deterministicCSR)
    {
        ReadRows<algorithmFPType, cpu> mtData(ntData, 0, nClusters);
        DAAL_CHECK_BLOCK_STATUS(mtData);
        const algorithmFPType * data = mtData.get();
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
    if (method == randomDense || method == randomCSR)
    {
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nClusters, sizeof(int));

        TArray<int, cpu> aIndices(nClusters);
        DAAL_CHECK(aIndices.get(), ErrorMemoryAllocationFailed);
        int * indices = aIndices.get();
        ReadRows<algorithmFPType, cpu> mtData;

        size_t k = 0;
        Status s;
        for (size_t i = 0; i < nClusters; i++)
        {
            DAAL_ASSERT(nRowsTotal <= services::internal::MaxVal<int>::get())
            DAAL_CHECK_STATUS(s, (UniformKernelDefault<int, cpu>::compute(i, (int)nRowsTotal, engine, 1, indices + i)));
            DAAL_ASSERT(indices[i] >= 0)
            size_t c    = (size_t)indices[i];
            int & value = indices[i];
            for (size_t j = i; j > 0; j--)
            {
                if (value == indices[j - 1])
                {
                    c     = (size_t)(j - 1);
                    value = c;
                }
            }
            if (c >= n) continue;
            const algorithmFPType * data = mtData.set(ntData, c, 1);
            DAAL_CHECK_BLOCK_STATUS(mtData);
            for (size_t j = 0; j < p; j++)
            {
                clusters[k * p + j] = data[j];
            }
            k++;
        }
        clustersFound = k;
        return s;
    }
    DAAL_ASSERT(false && "should never happen");
    std::cout << "init end" << std::endl;
    return Status();
}

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status KMeansInitKernel<method, algorithmFPType, cpu>::compute(size_t na, const NumericTable * const * a, size_t nr,
                                                                         const NumericTable * const * r, const Parameter * par,
                                                                         engines::BatchBase & engine)
{
    std::cout << "compute begin" << std::endl;
    NumericTable * ntData     = const_cast<NumericTable *>(a[0]);
    NumericTable * ntClusters = const_cast<NumericTable *>(r[0]);

    const size_t p         = ntData->getNumberOfColumns();
    const size_t n         = ntData->getNumberOfRows();
    const size_t nClusters = par->nClusters;

    WriteOnlyRows<algorithmFPType, cpu> clustersBD(ntClusters, 0, nClusters);
    algorithmFPType * clusters = clustersBD.get();

    size_t clustersFound = 0;
    std::cout << "compute end" << std::endl;
    return init<method, algorithmFPType, cpu>(p, n, n, nClusters, clusters, ntData, par->seed, engine, clustersFound);
}

template <typename algorithmFPType, CpuType cpu>
Status initDistrDeterministic(const NumericTable * pData, const Parameter * par, size_t & nClustersFound, NumericTablePtr & pRes)
{
    std::cout << "init distr begin" << std::endl;
    nClustersFound = 0;
    if (par->nClusters <= par->offset) return Status(); //ok

    nClustersFound     = par->nClusters - par->offset;
    const size_t nRows = pData->getNumberOfRows();
    if (nClustersFound > nRows) nClustersFound = nRows;
    Status st;
    if (!pRes)
    {
        pRes = HomogenNumericTableCPU<algorithmFPType, cpu>::create(pData->getNumberOfColumns(), nClustersFound, &st);
        DAAL_CHECK_STATUS_VAR(st);
    }

    WriteOnlyRows<algorithmFPType, cpu> resBD(*pRes, 0, nClustersFound);
    DAAL_CHECK_BLOCK_STATUS(resBD);
    ReadRows<algorithmFPType, cpu> dataBD(*const_cast<NumericTable *>(pData), 0, nClustersFound);
    DAAL_CHECK_BLOCK_STATUS(dataBD);
    const size_t sz = pData->getNumberOfColumns() * nClustersFound * sizeof(algorithmFPType);
    int result      = daal::services::internal::daal_memcpy_s(resBD.get(), sz, dataBD.get(), sz);
    std::cout << "init distr end" << std::endl;
    return (!result) ? st : services::Status(services::ErrorMemoryCopyFailedInternal);
}

template <typename algorithmFPType, CpuType cpu>
Status generateRandomIndices(const Parameter * par, size_t nRows, size_t & nClustersFound, int * clusters, engines::BatchBase & engine)
{
    std::cout << "gen random indicies begin" << std::endl;
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, par->nClusters, sizeof(int));

    TArray<int, cpu> aIndices(par->nClusters);
    int * indices = aIndices.get();
    DAAL_CHECK(indices, ErrorMemoryAllocationFailed);
    nClustersFound = 0;

    Status s;
    for (size_t i = 0; i < par->nClusters; i++)
    {
        DAAL_ASSERT(par->nRowsTotal <= services::internal::MaxVal<int>::get())
        DAAL_CHECK_STATUS(s, (UniformKernelDefault<int, cpu>::compute(i, (int)par->nRowsTotal, engine, 1, indices + i)));
        size_t c  = (size_t)indices[i];
        int value = indices[i];
        for (size_t j = i; j > 0; j--)
        {
            if (value == indices[j - 1])
            {
                c     = (size_t)(j - 1);
                value = c;
            }
        }

        if (c < par->offset || c >= par->offset + nRows) continue;
        clusters[nClustersFound] = c - par->offset;
        nClustersFound++;
    }
    std::cout << "gen random indicies end" << std::endl;
    return s;
}

template <typename algorithmFPType, CpuType cpu>
Status initDistrRandom(const NumericTable * pData, const Parameter * par, size_t & nClustersFound, NumericTablePtr & pRes,
                       engines::BatchBase & engine)
{
    std::cout << "initDistrRandom begin" << std::endl;
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, par->nClusters, sizeof(int));

    TArray<int, cpu> clusters(par->nClusters);
    DAAL_CHECK(clusters.get(), ErrorMemoryAllocationFailed);
    Status s = generateRandomIndices<algorithmFPType, cpu>(par, pData->getNumberOfRows(), nClustersFound, clusters.get(), engine);
    if (!s) return s;

    if (!nClustersFound) return s; //ok

    const size_t p = pData->getNumberOfColumns();
    if (!pRes)
    {
        pRes = HomogenNumericTableCPU<algorithmFPType, cpu>::create(p, nClustersFound, &s);
        DAAL_CHECK_STATUS_VAR(s);
    }

    WriteOnlyRows<algorithmFPType, cpu> resBD(*pRes, 0, nClustersFound);
    DAAL_CHECK_BLOCK_STATUS(resBD);

    auto aClusters = resBD.get();
    ReadRows<algorithmFPType, cpu> dataBD;
    for (size_t i = 0; i < nClustersFound; ++i)
    {
        auto pRow = dataBD.set(const_cast<NumericTable *>(pData), clusters.get()[i], 1);
        DAAL_CHECK_BLOCK_STATUS(dataBD);
        for (size_t j = 0; j < p; j++) aClusters[i * p + j] = pRow[j];
    }
    std::cout << "initDistrRandom end" << std::endl;
    return s;
}

template <typename algorithmFPType, CpuType cpu>
Status initDistrPlusPlus(const NumericTable * pData, const Parameter * par, size_t & nClustersFound, NumericTablePtr & pRes,
                         engines::BatchBase & engine)
{
    std::cout << "initDistrPlusPlus begin" << std::endl;
    nClustersFound = 0;
    int index      = 0;

    Status s;
    DAAL_CHECK(par->nRowsTotal <= services::internal::MaxVal<int>::get(), ErrorIncorrectNumberOfRows)
    DAAL_CHECK_STATUS(s, (UniformKernelDefault<int, cpu>::compute(0, (int)par->nRowsTotal, engine, 1, &index)));

    DAAL_ASSERT(index >= 0)
    size_t c(index);
    if (c < par->offset) return Status();                             //ok
    if (c >= par->offset + pData->getNumberOfRows()) return Status(); //ok

    ReadRows<algorithmFPType, cpu> dataBD(*const_cast<NumericTable *>(pData), c - par->offset, 1);
    DAAL_CHECK_BLOCK_STATUS(dataBD);

    if (!pRes)
    {
        pRes = HomogenNumericTableCPU<algorithmFPType, cpu>::create(pData->getNumberOfColumns(), 1, &s);
        DAAL_CHECK_STATUS_VAR(s);
    }

    nClustersFound = 1;
    const size_t p = pData->getNumberOfColumns();
    WriteOnlyRows<algorithmFPType, cpu> resBD(*pRes, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(resBD);
    int result = daal::services::internal::daal_memcpy_s(resBD.get(), sizeof(algorithmFPType) * p, dataBD.get(), sizeof(algorithmFPType) * p);
    std::cout << "initDistrPlusPlus end" << std::endl;
    return (!result) ? s : services::Status(services::ErrorMemoryCopyFailedInternal);
}

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status KMeansInitStep1LocalKernel<method, algorithmFPType, cpu>::compute(const NumericTable * pData, const Parameter * par,
                                                                                   NumericTable * pNumPartialClusters,
                                                                                   NumericTablePtr & pPartialClusters, engines::BatchBase & engine)
{
    std::cout << "compute 2 begin" << std::endl;
    size_t nClustersFound = 0;
    services::Status s;
    if ((method == deterministicDense) || (method == deterministicCSR))
        s = initDistrDeterministic<algorithmFPType, cpu>(pData, par, nClustersFound, pPartialClusters);
    else if ((method == randomDense) || (method == randomCSR))
        s = initDistrRandom<algorithmFPType, cpu>(pData, par, nClustersFound, pPartialClusters, engine);
    else if (isPlusPlusMethod(method))
        s = initDistrPlusPlus<algorithmFPType, cpu>(pData, par, nClustersFound, pPartialClusters, engine);
    else
        DAAL_ASSERT(false && "should never happen");
    if (!s) return s;
    WriteOnlyRows<int, cpu> npcBD(*pNumPartialClusters, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(npcBD);
    DAAL_CHECK(nClustersFound <= services::internal::MaxVal<int>::get(), ErrorIncorrectNumberOfPartialClusters)
    *npcBD.get() = (int)nClustersFound;
    std::cout << "compute 2 end" << std::endl;
    return s;
}

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status KMeansInitStep2MasterKernel<method, algorithmFPType, cpu>::finalizeCompute(size_t na, const NumericTable * const * a,
                                                                                            NumericTable * ntClusters, const Parameter * par)
{
    std::cout << "finalizeCompute begin" << std::endl;
    const size_t nBlocks   = na / 2;
    const size_t p         = ntClusters->getNumberOfColumns();
    const size_t nClusters = par->nClusters;

    WriteOnlyRows<algorithmFPType, cpu> mtClusters(ntClusters, 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtClusters);
    algorithmFPType * clusters = mtClusters.get();

    size_t k = 0;
    for (size_t i = 0; i < nBlocks; i++)
    {
        if (!a[i * 2 + 1]) continue; //can be null
        ReadRows<int, cpu> mtInClustersN(*const_cast<NumericTable *>(a[i * 2 + 0]), 0, nClusters);
        DAAL_CHECK_BLOCK_STATUS(mtInClustersN);
        const int * inClustersN = mtInClustersN.get();
        ReadRows<algorithmFPType, cpu> mtInClusters(*const_cast<NumericTable *>(a[i * 2 + 1]), 0, 1);
        DAAL_CHECK_BLOCK_STATUS(mtInClusters);
        const algorithmFPType * inClusters = mtInClusters.get();

        const size_t inK = *inClustersN;
        for (size_t j = 0; j < inK; j++)
        {
            for (size_t h = 0; h < p; h++)
            {
                clusters[k * p + h] = inClusters[j * p + h];
            }
            k++;
        }
    }
    std::cout << "finalizeCompute begin" << std::endl;
    return Status();
}

} // namespace internal
} // namespace init
} // namespace kmeans
} // namespace algorithms
} // namespace daal
