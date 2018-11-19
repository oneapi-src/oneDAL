/* file: kmeans_lloyd_batch_impl.i */
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

template <Method method, typename algorithmFPType, CpuType cpu>
Status KMeansBatchKernel<method, algorithmFPType, cpu>::compute(const NumericTable *const *a,
    const NumericTable *const *r, const Parameter *par)
{
    NumericTable *ntData     = const_cast<NumericTable *>( a[0] );
    const size_t nIter = par->maxIterations;
    const size_t p = ntData->getNumberOfColumns();
    const size_t nClusters = par->nClusters;

    TArray<int, cpu> clusterS0(nClusters);
    TArray<algorithmFPType, cpu> clusterS1(nClusters*p);
    DAAL_CHECK(clusterS0.get() && clusterS1.get(), services::ErrorMemoryAllocationFailed);

    /* Categorial variables check and support: begin */
    int catFlag = 0;
    for(size_t i = 0; i < p; i++)
    {
        if (ntData->getFeatureType(i) == features::DAAL_CATEGORICAL)
        {
            catFlag = 1;
            break;
        }
    }
    TArray<algorithmFPType, cpu> catCoef(catFlag ? p : 0);
    if(catFlag)
    {
        DAAL_CHECK(catCoef.get(), services::ErrorMemoryAllocationFailed);
        for(size_t i = 0; i < p; i++)
        {
            if (ntData->getFeatureType(i) == features::DAAL_CATEGORICAL)
            {
                catCoef[i] = par->gamma;
            }
            else
            {
                catCoef[i] = (algorithmFPType)1.0;
            }
        }
    }

    ReadRows<algorithmFPType, cpu> mtInClusters(*const_cast<NumericTable*>(a[1]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtInClusters);
    WriteOnlyRows<algorithmFPType, cpu> mtClusters(*const_cast<NumericTable*>(r[0]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtClusters);

    algorithmFPType *inClusters = const_cast<algorithmFPType*>(mtInClusters.get());
    algorithmFPType *clusters = mtClusters.get();

    TArray<double, cpu> dS1(method == defaultDense ? p : 0);
    if (method == defaultDense)
    {
        DAAL_CHECK(dS1.get(), services::ErrorMemoryAllocationFailed);
    }

    TArray<algorithmFPType, cpu> cValues(nClusters);
    TArray<size_t, cpu> cIndices(nClusters);

    Status s;
    algorithmFPType oldTargetFunc(0.0);
    size_t kIter;
    for(kIter = 0; kIter < nIter; kIter++)
    {
        SharedPtr<task_t<algorithmFPType, cpu> > task = task_t<algorithmFPType, cpu>::create(p, nClusters, inClusters);
        DAAL_CHECK(task.get(), services::ErrorMemoryAllocationFailed);
        DAAL_ASSERT(task);

        s = task->template addNTToTaskThreaded<method>(ntData, catCoef.get());
        if(!s)
        {
            task->kmeansClearClusters(&oldTargetFunc);
            break;
        }

        task->template kmeansComputeCentroids<method>(clusterS0.get(), clusterS1.get(), dS1.get());

        size_t cNum;
        DAAL_CHECK_STATUS(s, task->kmeansComputeCentroidsCandidates(cValues.get(), cIndices.get(), cNum));
        size_t cPos = 0;

        algorithmFPType newCentersGoalFunc = (algorithmFPType)0.0;

        for (size_t i = 0; i < nClusters; i++)
        {
            if ( clusterS0[i] > 0 )
            {
                const algorithmFPType coeff = 1.0 / clusterS0[i];

                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j < p; j++)
                {
                    clusters[i * p + j] = clusterS1[i * p + j] * coeff;
                }
            }
            else
            {
                DAAL_CHECK(cPos < cNum, services::ErrorKMeansNumberOfClustersIsTooLarge);
                newCentersGoalFunc += cValues[cPos];
                ReadRows<algorithmFPType, cpu> mtRow(ntData, cIndices[cPos], 1);
                const algorithmFPType *row = mtRow.get();
                daal::services::daal_memcpy_s(&clusters[i * p], p * sizeof(algorithmFPType), row, p * sizeof(algorithmFPType));
                cPos++;
            }
        }

        if ( par->accuracyThreshold > (algorithmFPType)0.0 )
        {
            algorithmFPType newTargetFunc = (algorithmFPType)0.0;

            task->kmeansClearClusters(&newTargetFunc);
            newTargetFunc -= newCentersGoalFunc;

            if ( internal::Math<algorithmFPType, cpu>::sFabs(oldTargetFunc - newTargetFunc) < par->accuracyThreshold )
            {
                kIter++;
                break;
            }

            oldTargetFunc = newTargetFunc;
        }
        else
        {
            task->kmeansClearClusters(&oldTargetFunc);
            oldTargetFunc -= newCentersGoalFunc;
        }
        inClusters = clusters;
    }

    if(!nIter)
    {
        clusters = inClusters;
    }

    NumericTable * assignments = par->assignFlag ? const_cast<NumericTable *>(r[1]) : nullptr;
    algorithmFPType targetFunc = algorithmFPType(0);

    s = RecalculationObservations<method, algorithmFPType, cpu>(p, nClusters, inClusters, ntData, catCoef.get(), assignments, targetFunc);

    WriteOnlyRows<int, cpu> mtIterations(*const_cast<NumericTable *>(r[3]), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(mtIterations);
    *mtIterations.get() = kIter;

    WriteOnlyRows<algorithmFPType, cpu> mtTarget(*const_cast<NumericTable *>(r[2]), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(mtTarget);
    *mtTarget.get() = targetFunc;

    return s;
}

} // namespace daal::algorithms::kmeans::internal
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
