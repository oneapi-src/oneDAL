/* file: kmeans_lloyd_batch_impl.i */
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
void KMeansBatchKernel<method, interm, cpu>::compute( size_t na, const NumericTable *const *a,
                                                           size_t nr, const NumericTable *const *r, const Parameter *par)
{
    NumericTable *ntData     = const_cast<NumericTable *>( a[0] );

    size_t nIter = par->maxIterations;

    size_t p = ntData->getNumberOfColumns();
    size_t n = ntData->getNumberOfRows();
    size_t nClusters = par->nClusters;

    size_t *clusterS0 = (size_t *)daal::services::daal_malloc( sizeof(size_t) * nClusters );
    if(!clusterS0)
    {
        this->_errors->add(services::ErrorMemoryAllocationFailed);
        return;
    }

    interm *clusterS1 = (interm *)daal::services::daal_malloc( sizeof(interm) * nClusters * p );
    if(!clusterS1)
    {
        this->_errors->add(services::ErrorMemoryAllocationFailed);
        daal::services::daal_free( clusterS0 );
        return;
    }

    /* Categorial variables check and support: begin */
    int catFlag = 0;
    interm *catCoef = 0;
    for(size_t i = 0; i < p; i++)
    {
        if (ntData->getFeatureType(i) == data_feature_utils::DAAL_CATEGORICAL)
        {
            catFlag = 1;
            break;
        }
    }

    if(catFlag)
    {
        catCoef = new interm[p];

        for(size_t i = 0; i < p; i++)
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

    BlockMicroTable<interm, readOnly,  cpu> mtInClusters( a[1] );
    BlockMicroTable<interm, writeOnly, cpu> mtClusters( r[0] );

    interm *inClusters;
    interm *clusters;

    mtInClusters.getBlockOfRows( 0, nClusters, &inClusters );
    mtClusters  .getBlockOfRows( 0, nClusters, &clusters   );

    size_t kIter;
    interm oldTargetFunc = (interm)0.0;

    for(kIter = 0; kIter < nIter; kIter++)
    {
        void *task = kmeansInitTask<interm, cpu>(p, nClusters, inClusters, this->_errors);
        if(!task){ daal::services::daal_free( clusterS0 ); daal::services::daal_free( clusterS1 ); return; }

        addNTToTaskThreaded<method, interm, cpu, 0>(task, ntData, catCoef );

        for (size_t i = 0; i < nClusters; i++)
        {
            for (size_t j = 0; j < p; j++)
            {
                clusterS1[i * p + j] = 0.0;
            }

            clusterS0[i] = kmeansUpdateCluster<interm, cpu>( task, i, &clusterS1[i * p] );
        }

        for (size_t i = 0; i < nClusters; i++)
        {
            if ( clusterS0[i] > 0 )
            {
                interm coeff = 1.0 / clusterS0[i];

                for (size_t j = 0; j < p; j++)
                {
                    clusters[i * p + j] = clusterS1[i * p + j] * coeff;
                }
            }
        }

        if ( par->accuracyThreshold > (interm)0.0 )
        {
            interm newTargetFunc = (interm)0.0;

            kmeansClearClusters<interm, cpu>(task, &newTargetFunc);

            if ( __DAAL_FABS(oldTargetFunc - newTargetFunc) < par->accuracyThreshold )
            {
                kIter++;
                break;
            }

            oldTargetFunc = newTargetFunc;
        }
        else
        {
            kmeansClearClusters<interm, cpu>(task, &oldTargetFunc);
        }

        inClusters = clusters;
    }

    if( par->assignFlag )
    {
        if(!nIter)
        {
            clusters = inClusters;
        }

        void *task = kmeansInitTask<interm, cpu>(p, nClusters, clusters, this->_errors);
        if(!task){ daal::services::daal_free( clusterS0 ); daal::services::daal_free( clusterS1 ); return; }

        getNTAssignmentsThreaded<method, interm, cpu>(task, ntData, r[1], catCoef);
        kmeansClearClusters<interm, cpu>(task, 0);
    }

    daal::services::daal_free( clusterS0 );
    daal::services::daal_free( clusterS1 );

    mtInClusters.release();
    mtClusters  .release();

    int* nIterations;
    BlockMicroTable<int, writeOnly, cpu> mtIterations( r[3] );
    mtIterations.getBlockOfRows(0, 1, &nIterations);
    *nIterations = kIter;
    mtIterations.release();

    interm* goal;
    BlockMicroTable<interm, writeOnly, cpu> mtTarget( r[2] );
    mtTarget.getBlockOfRows(0, 1, &goal);
    *goal = oldTargetFunc;
    mtTarget.release();

    if (catFlag)
    {
        delete[] catCoef;
    }
}

} // namespace daal::algorithms::kmeans::internal
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
