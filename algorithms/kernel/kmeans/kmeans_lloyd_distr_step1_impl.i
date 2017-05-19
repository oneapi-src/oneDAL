/* file: kmeans_lloyd_distr_step1_impl.i */
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
services::Status KMeansDistributedStep1Kernel<method, algorithmFPType, cpu>::compute( size_t na, const NumericTable *const *a,
                                                                                      size_t nr, const NumericTable *const *r, const Parameter *par)
{
    NumericTable *ntData = const_cast<NumericTable *>(a[0]);
    NumericTable* ntAssignments = const_cast<NumericTable*>(r[3]);

    const size_t p = ntData->getNumberOfColumns();
    const size_t n = ntData->getNumberOfRows();
    const size_t nClusters = par->nClusters;

    ReadRows<algorithmFPType, cpu> mtInitClusters(*const_cast<NumericTable*>(a[1]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtInitClusters);
    algorithmFPType *initClusters = const_cast<algorithmFPType*>(mtInitClusters.get());
    WriteOnlyRows<int, cpu> mtClusterS0(*const_cast<NumericTable*>(r[0]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtClusterS0);
    /* TODO: That should be size_t or double */
    int * clusterS0 = mtClusterS0.get();
    WriteOnlyRows<algorithmFPType, cpu> mtClusterS1(*const_cast<NumericTable*>(r[1]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtClusterS1);
    algorithmFPType *clusterS1 = mtClusterS1.get();
    WriteOnlyRows<algorithmFPType, cpu> mtTargetFunc(*const_cast<NumericTable*>(r[2]), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(mtTargetFunc);
    algorithmFPType *goalFunc = mtTargetFunc.get();

    /* Categorial variables check and support: begin */
    int catFlag = 0;
    for(size_t i = 0; i < p; i++)
    {
        if(ntData->getFeatureType(i) == data_feature_utils::DAAL_CATEGORICAL)
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
            if(ntData->getFeatureType(i) == data_feature_utils::DAAL_CATEGORICAL)
            {
                catCoef[i] = par->gamma;
            }
            else
            {
                catCoef[i] = (algorithmFPType)1.0;
            }
        }
    }

    services::Status s;
    algorithmFPType oldTargetFunc = (algorithmFPType)0.0;
    {
        void *task = nullptr;
        services::Status s1 = kmeansInitTask<algorithmFPType, cpu>(p, nClusters, initClusters, task);
        DAAL_CHECK_STATUS(s, s1);
        DAAL_ASSERT(task);

        if( par->assignFlag )
        {
            s = addNTToTaskThreaded<method, algorithmFPType, cpu, 1>(task, ntData, catCoef.get(), ntAssignments);
        }
        else
        {
            s = addNTToTaskThreaded<method, algorithmFPType, cpu, 0>(task, ntData, catCoef.get());
        }
        if(!s)
        {
            kmeansClearClusters<algorithmFPType, cpu>(task, &oldTargetFunc);
            return s;
        }

        for (size_t i = 0; i < nClusters; i++)
        {
            for (size_t j = 0; j < p; j++)
            {
                clusterS1[i * p + j] = 0.0;
            }

            clusterS0[i] = kmeansUpdateCluster<algorithmFPType, cpu>( task, i, &clusterS1[i * p] );
        }
        kmeansClearClusters<algorithmFPType, cpu>(task, goalFunc);
    }
    return s;
}

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status KMeansDistributedStep1Kernel<method, algorithmFPType, cpu>::finalizeCompute( size_t na, const NumericTable *const *a,
                                                                                              size_t nr, const NumericTable *const *r, const Parameter *par)
{
    if(!par->assignFlag)
        return services::Status();

        NumericTable *ntPartialAssignments = const_cast<NumericTable *>(a[0]);
        NumericTable *ntAssignments = const_cast<NumericTable *>(r[0]);
    const size_t n = ntPartialAssignments->getNumberOfRows();

    ReadRows<int, cpu> inBlock(*ntPartialAssignments, 0, n);
    DAAL_CHECK_BLOCK_STATUS(inBlock);
    const int* inAssignments = inBlock.get();

    WriteOnlyRows<int, cpu> outBlock(*ntAssignments, 0, n);
    DAAL_CHECK_BLOCK_STATUS(outBlock);
    int* outAssignments = outBlock.get();

      PRAGMA_IVDEP
        for(size_t i=0; i<n; i++)
        {
            outAssignments[i] = inAssignments[i];
        }
    return services::Status();
}

} // namespace daal::algorithms::kmeans::internal
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
