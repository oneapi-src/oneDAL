/* file: kmeans_lloyd_batch_impl.i */
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

#define __DAAL_FABS(a) (((a)>(algorithmFPType)0.0)?(a):(-(a)))

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status KMeansBatchKernel<method, algorithmFPType, cpu>::compute(const NumericTable *const *a,
    const NumericTable *const *r, const Parameter *par)
{
    NumericTable *ntData     = const_cast<NumericTable *>( a[0] );
    const size_t nIter = par->maxIterations;
    const size_t p = ntData->getNumberOfColumns();
    const size_t n = ntData->getNumberOfRows();
    const size_t nClusters = par->nClusters;

    TArray<size_t, cpu> clusterS0(nClusters);
    TArray<algorithmFPType, cpu> clusterS1(nClusters*p);
    DAAL_CHECK(clusterS0.get() && clusterS1.get(), services::ErrorMemoryAllocationFailed);

    /* Categorial variables check and support: begin */
    int catFlag = 0;
    for(size_t i = 0; i < p; i++)
    {
        if (ntData->getFeatureType(i) == data_feature_utils::DAAL_CATEGORICAL)
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
            if (ntData->getFeatureType(i) == data_feature_utils::DAAL_CATEGORICAL)
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

    services::Status s;
    algorithmFPType oldTargetFunc(0.0);
    size_t kIter;
    for(kIter = 0; kIter < nIter; kIter++)
    {
        void *task = nullptr;
        services::Status s1 = kmeansInitTask<algorithmFPType, cpu>(p, nClusters, inClusters, task);
        DAAL_CHECK_STATUS(s, s1);
        DAAL_ASSERT(task);

        s = addNTToTaskThreaded<method, algorithmFPType, cpu, 0>(task, ntData, catCoef.get());
        if(!s)
        {
            kmeansClearClusters<algorithmFPType, cpu>(task, &oldTargetFunc);
            break;
        }
        for (size_t i = 0; i < nClusters; i++)
        {
            for (size_t j = 0; j < p; j++)
            {
                clusterS1[i * p + j] = 0.0;
            }

            clusterS0[i] = kmeansUpdateCluster<algorithmFPType, cpu>( task, i, &clusterS1[i * p] );
        }

        for (size_t i = 0; i < nClusters; i++)
        {
            if ( clusterS0[i] > 0 )
            {
                algorithmFPType coeff = 1.0 / clusterS0[i];

                for (size_t j = 0; j < p; j++)
                {
                    clusters[i * p + j] = clusterS1[i * p + j] * coeff;
                }
            }
        }

        if ( par->accuracyThreshold > (algorithmFPType)0.0 )
        {
            algorithmFPType newTargetFunc = (algorithmFPType)0.0;

            kmeansClearClusters<algorithmFPType, cpu>(task, &newTargetFunc);

            if ( __DAAL_FABS(oldTargetFunc - newTargetFunc) < par->accuracyThreshold )
            {
                kIter++;
                break;
            }

            oldTargetFunc = newTargetFunc;
        }
        else
        {
            kmeansClearClusters<algorithmFPType, cpu>(task, &oldTargetFunc);
        }

        inClusters = clusters;
    }

    if( s.ok() && par->assignFlag )
    {
        if(!nIter)
            clusters = inClusters;

        void *task = nullptr;
        services::Status s1 = kmeansInitTask<algorithmFPType, cpu>(p, nClusters, inClusters, task);
        DAAL_CHECK_STATUS(s, s1);
        DAAL_ASSERT(task);

        s = getNTAssignmentsThreaded<method, algorithmFPType, cpu>(task, ntData, r[1], catCoef.get());
        kmeansClearClusters<algorithmFPType, cpu>(task, 0);
    }

    WriteOnlyRows<int, cpu> mtIterations(*const_cast<NumericTable *>(r[3]), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(mtIterations);
    *mtIterations.get() = kIter;

    WriteOnlyRows<algorithmFPType, cpu> mtTarget(*const_cast<NumericTable *>(r[2]), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(mtTarget);
    *mtTarget.get() = oldTargetFunc;
    return s;
}

} // namespace daal::algorithms::kmeans::internal
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
