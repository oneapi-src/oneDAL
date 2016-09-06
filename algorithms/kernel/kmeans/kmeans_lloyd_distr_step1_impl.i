/* file: kmeans_lloyd_distr_step1_impl.i */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
#include "service_micro_table.h"

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

#define __DAAL_FABS(a) (((a)>(interm)0.0)?(a):(-(a)))

template <Method method, typename interm, CpuType cpu>
void KMeansDistributedStep1Kernel<method, interm, cpu>::compute( size_t na, const NumericTable *const *a,
                                                                     size_t nr, const NumericTable *const *r, const Parameter *par)
{
    const NumericTable *ntData  = a[0];

    BlockMicroTable<interm, readOnly,  cpu> mtData        ( ntData );
    BlockMicroTable<interm, readOnly,  cpu> mtInitClusters( a[1] );
    BlockMicroTable<int,    writeOnly, cpu> mtClusterS0   ( r[0] );
    BlockMicroTable<interm, writeOnly, cpu> mtClusterS1   ( r[1] );
    BlockMicroTable<interm, writeOnly, cpu> mtTargetFunc  ( r[2] );

    NumericTable* ntAssignments = const_cast<NumericTable*>(r[3]);

    size_t p = mtData.getFullNumberOfColumns();
    size_t n = mtData.getFullNumberOfRows();
    size_t nClusters = par->nClusters;

    interm *initClusters;

    /* TODO: That should be size_t or double */
    int    *clusterS0;
    interm *clusterS1;
    interm *goalFunc;

    mtInitClusters.getBlockOfRows(0, nClusters, &initClusters);
    mtClusterS0   .getBlockOfRows(0, nClusters, &clusterS0   );
    mtClusterS1   .getBlockOfRows(0, nClusters, &clusterS1   );
    mtTargetFunc  .getBlockOfRows(0, 1,         &goalFunc  );

    /* Categorial variables check and support: begin */
    int catFlag = 0;
    interm *catCoef = 0;
    for (size_t i = 0; i < p; i++)
    {
        if (ntData->getFeatureType(i) == data_feature_utils::DAAL_CATEGORICAL)
        {
            catFlag = 1;
            break;
        }
    }

    if (catFlag)
    {
        catCoef = new interm[p];

        for (size_t i = 0; i < p; i++)
        {
            if (ntData->getFeatureType(i) == data_feature_utils::DAAL_CATEGORICAL)
            {
                catCoef[i] = par->gamma;
            }
            else
            {
                catCoef[i] = (interm)1.0;
            }
        }
    }

    interm oldTargetFunc = (interm)0.0;

    {
        void *task = kmeansInitTask<interm, cpu>(p, nClusters, initClusters, this->_errors);
        if(!task)
        {
           if (catFlag)
           {
             delete[] catCoef;
           }

           mtInitClusters.release();
           mtClusterS0   .release();
           mtClusterS1   .release();
           mtTargetFunc  .release();

            return;
        }

        if( par->assignFlag )
        {
            addNTToTaskThreaded<method, interm, cpu, 1>(task, ntData, catCoef, ntAssignments);
        }
        else
        {
            addNTToTaskThreaded<method, interm, cpu, 0>(task, ntData, catCoef);
        }

        for (size_t i = 0; i < nClusters; i++)
        {
            for (size_t j = 0; j < p; j++)
            {
                clusterS1[i * p + j] = 0.0;
            }

            clusterS0[i] = kmeansUpdateCluster<interm, cpu>( task, i, &clusterS1[i * p] );
        }
        kmeansClearClusters<interm, cpu>(task, goalFunc);
    }

    if (catFlag)
    {
        delete[] catCoef;
    }

    mtInitClusters.release();
    mtClusterS0   .release();
    mtClusterS1   .release();
    mtTargetFunc  .release();
}

template <Method method, typename interm, CpuType cpu>
void KMeansDistributedStep1Kernel<method, interm, cpu>::finalizeCompute( size_t na, const NumericTable *const *a,
                                                                             size_t nr, const NumericTable *const *r, const Parameter *par)
{
    if( par->assignFlag )
    {
        NumericTable *ntPartialAssignments = const_cast<NumericTable *>(a[0]);
        NumericTable *ntAssignments = const_cast<NumericTable *>(r[0]);

        size_t n = ntPartialAssignments->getNumberOfRows();

        BlockDescriptor<int> inBlock;
        BlockDescriptor<int> outBlock;
        ntPartialAssignments->getBlockOfRows(0,n,readOnly,inBlock);
        ntAssignments->getBlockOfRows(0,n,writeOnly,outBlock);

        int* inAssignments = inBlock.getBlockPtr();
        int* outAssignments = outBlock.getBlockPtr();

      PRAGMA_IVDEP
        for(size_t i=0; i<n; i++)
        {
            outAssignments[i] = inAssignments[i];
        }

        ntAssignments->releaseBlockOfRows(outBlock);
        ntPartialAssignments->releaseBlockOfRows(inBlock);
    }
}

} // namespace daal::algorithms::kmeans::internal
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
