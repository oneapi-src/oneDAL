/* file: kmeans_lloyd_distr_step2_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#include "kmeans_lloyd_impl.i"

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace internal
{
#define __DAAL_FABS(a) (((a) > (algorithmFPType)0.0) ? (a) : (-(a)))

template <Method method, typename algorithmFPType, CpuType cpu>
Status KMeansDistributedStep2Kernel<method, algorithmFPType, cpu>::compute(size_t na, const NumericTable * const * a, size_t nr,
                                                                           const NumericTable * const * r, const Parameter * par)
{
    const size_t nClusters = par->nClusters;
    const size_t p         = r[1]->getNumberOfColumns();
    int result             = 0;

    WriteOnlyRows<int, cpu> mtClusterS0(*const_cast<NumericTable *>(r[0]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtClusterS0);
    /* TODO: That should be size_t or double */
    int * clusterS0 = mtClusterS0.get();
    WriteOnlyRows<algorithmFPType, cpu> mtClusterS1(*const_cast<NumericTable *>(r[1]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtClusterS1);
    algorithmFPType * clusterS1 = mtClusterS1.get();
    WriteOnlyRows<algorithmFPType, cpu> mtTargetFunc(*const_cast<NumericTable *>(r[2]), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(mtTargetFunc);
    algorithmFPType * goalFunc = mtTargetFunc.get();

    WriteOnlyRows<algorithmFPType, cpu> mtCValues(*const_cast<NumericTable *>(r[3]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtCValues);
    algorithmFPType * cValues = mtCValues.get();
    WriteOnlyRows<algorithmFPType, cpu> mtCCentroids(*const_cast<NumericTable *>(r[4]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtCCentroids);
    algorithmFPType * cCentroids = mtCCentroids.get();

    const size_t nBlocks = na / 5;

    /* TODO: initialization  */
    for (size_t j = 0; j < nClusters; j++)
    {
        clusterS0[j] = 0;
    }

    for (size_t j = 0; j < nClusters * p; j++)
    {
        clusterS1[j] = 0;
    }

    goalFunc[0] = 0;

    for (size_t j = 0; j < nClusters; j++)
    {
        cValues[j] = (algorithmFPType)-1.0;
    }

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nClusters, sizeof(algorithmFPType));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nClusters, sizeof(size_t));

    TArray<algorithmFPType, cpu> tmpValues(nClusters);
    TArray<size_t, cpu> tmpIndices(nClusters);
    TArray<size_t, cpu> cIndices(nClusters);
    DAAL_CHECK_MALLOC(tmpValues.get() && tmpIndices.get() && cIndices.get());

    for (size_t i = 0; i < nBlocks; i++)
    {
        ReadRows<int, cpu> mtInClusterS0(*const_cast<NumericTable *>(a[i * 5 + 0]), 0, nClusters);
        DAAL_CHECK_BLOCK_STATUS(mtInClusterS0);
        ReadRows<algorithmFPType, cpu> mtInClusterS1(*const_cast<NumericTable *>(a[i * 5 + 1]), 0, nClusters);
        DAAL_CHECK_BLOCK_STATUS(mtInClusterS1);
        ReadRows<algorithmFPType, cpu> mtInTargetFunc(*const_cast<NumericTable *>(a[i * 5 + 2]), 0, 1);
        DAAL_CHECK_BLOCK_STATUS(mtInTargetFunc);
        ReadRows<algorithmFPType, cpu> mtInCValues(*const_cast<NumericTable *>(a[i * 5 + 3]), 0, nClusters);
        DAAL_CHECK_BLOCK_STATUS(mtInCValues);

        const int * inClusterS0              = mtInClusterS0.get();
        const algorithmFPType * inClusterS1  = mtInClusterS1.get();
        const algorithmFPType * inTargetFunc = mtInTargetFunc.get();
        const algorithmFPType * inCValues    = mtInCValues.get();

        for (size_t j = 0; j < nClusters; j++)
        {
            clusterS0[j] += inClusterS0[j];
        }

        for (size_t j = 0; j < nClusters * p; j++)
        {
            clusterS1[j] += inClusterS1[j];
        }

        goalFunc[0] += inTargetFunc[0];

        size_t cPos = 0, clPos = 0, cNum = 0;
        while (cNum < nClusters)
        {
            if (cValues[cPos] < (algorithmFPType)0.0 && inCValues[clPos] < (algorithmFPType)0.0)
            {
                break;
            }
            if (cValues[cPos] > inCValues[clPos])
            {
                tmpValues[cNum]  = cValues[cPos];
                tmpIndices[cNum] = cIndices[cPos];
                cNum++;
                cPos++;
            }
            else
            {
                tmpValues[cNum]  = inCValues[clPos];
                tmpIndices[cNum] = i * nClusters + clPos;
                cNum++;
                clPos++;
            }
        }
        result |= daal::services::internal::daal_memcpy_s(cValues, cNum * sizeof(algorithmFPType), tmpValues.get(), cNum * sizeof(algorithmFPType));
        result |= daal::services::internal::daal_memcpy_s(cIndices.get(), cNum * sizeof(size_t), tmpIndices.get(), cNum * sizeof(size_t));
    }

    for (size_t i = 0; i < nClusters; i++)
    {
        if (cValues[i] < (algorithmFPType)0.0)
        {
            break;
        }
        size_t block      = cIndices[i] / nClusters;
        size_t posInBlock = cIndices[i] % nClusters;

        ReadRows<algorithmFPType, cpu> mtInCCentroids(*const_cast<NumericTable *>(a[block * 5 + 4]), posInBlock, 1);
        DAAL_CHECK_BLOCK_STATUS(mtInCCentroids);
        const algorithmFPType * inCCentroids = mtInCCentroids.get();
        result |= daal::services::internal::daal_memcpy_s(&cCentroids[i * p], p * sizeof(algorithmFPType), inCCentroids, p * sizeof(algorithmFPType));
    }

    return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
}

template <Method method, typename algorithmFPType, CpuType cpu>
Status KMeansDistributedStep2Kernel<method, algorithmFPType, cpu>::finalizeCompute(size_t na, const NumericTable * const * a, size_t nr,
                                                                                   const NumericTable * const * r, const Parameter * par)
{
    const size_t p         = a[1]->getNumberOfColumns();
    const size_t nClusters = par->nClusters;
    int result             = 0;

    ReadRows<int, cpu> mtInClusterS0(*const_cast<NumericTable *>(a[0]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtInClusterS0);
    ReadRows<algorithmFPType, cpu> mtInClusterS1(*const_cast<NumericTable *>(a[1]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtInClusterS1);
    ReadRows<algorithmFPType, cpu> mtInTargetFunc(*const_cast<NumericTable *>(a[2]), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(mtInTargetFunc);

    ReadRows<algorithmFPType, cpu> mtCValues(*const_cast<NumericTable *>(a[3]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtCValues);
    ReadRows<algorithmFPType, cpu> mtCCentroids(*const_cast<NumericTable *>(a[4]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtCCentroids);

    /* TODO: That should be size_t or double */
    const int * clusterS0             = mtInClusterS0.get();
    const algorithmFPType * clusterS1 = mtInClusterS1.get();
    const algorithmFPType * inTarget  = mtInTargetFunc.get();

    const algorithmFPType * cValues    = mtCValues.get();
    const algorithmFPType * cCentroids = mtCCentroids.get();

    WriteOnlyRows<algorithmFPType, cpu> mtClusters(*const_cast<NumericTable *>(r[0]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtClusters);
    WriteOnlyRows<algorithmFPType, cpu> mtTargetFunct(*const_cast<NumericTable *>(r[1]), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(mtTargetFunct);

    algorithmFPType * clusters  = mtClusters.get();
    algorithmFPType * outTarget = mtTargetFunct.get();

    *outTarget = *inTarget;

    size_t cPos = 0;

    for (size_t i = 0; i < nClusters; i++)
    {
        if (clusterS0[i] > 0)
        {
            algorithmFPType coeff = 1.0 / clusterS0[i];

            for (size_t j = 0; j < p; j++)
            {
                clusters[i * p + j] = clusterS1[i * p + j] * coeff;
            }
        }
        else
        {
            DAAL_CHECK(!(cValues[cPos] < (algorithmFPType)0.0), services::ErrorKMeansNumberOfClustersIsTooLarge);
            outTarget[0] -= cValues[cPos];
            result |= daal::services::internal::daal_memcpy_s(&clusters[i * p], p * sizeof(algorithmFPType), &cCentroids[cPos * p],
                                                              p * sizeof(algorithmFPType));
            cPos++;
        }
    }

    return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
}

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal
