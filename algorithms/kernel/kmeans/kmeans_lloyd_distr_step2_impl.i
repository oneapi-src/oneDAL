/* file: kmeans_lloyd_distr_step2_impl.i */
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
void KMeansDistributedStep2Kernel<method, interm, cpu>::compute( size_t na, const NumericTable *const *a,
                                                                      size_t nr, const NumericTable *const *r, const Parameter *par)
{
    BlockMicroTable<int,    writeOnly, cpu> mtClusterS0   ( r[0] );
    BlockMicroTable<interm, writeOnly, cpu> mtClusterS1   ( r[1] );
    BlockMicroTable<interm, writeOnly, cpu> mtTargetFunc  ( r[2] );

    size_t nBlocks = na/3;

    size_t p = mtClusterS1.getFullNumberOfColumns();
    size_t nClusters = par->nClusters;

    /* TODO: That should be size_t or double */
    int    *clusterS0;
    interm *clusterS1;
    interm *goalFunc;

    mtClusterS0   .getBlockOfRows(0, nClusters, &clusterS0);
    mtClusterS1   .getBlockOfRows(0, nClusters, &clusterS1);
    mtTargetFunc  .getBlockOfRows(0, 1,         &goalFunc);

    /* TODO: initialization  */
    for(size_t j=0; j<nClusters; j++)
    {
        clusterS0[j] = 0;
    }

    for(size_t j=0; j<nClusters*p; j++)
    {
        clusterS1[j] = 0;
    }

    goalFunc[0] = 0;

    for(size_t i=0; i<nBlocks; i++)
    {
        int    *inClusterS0;
        interm *inClusterS1;
        interm *inTargetFunc;

        BlockMicroTable<int,    readOnly, cpu> mtInClusterS0   ( a[i*3+0] );
        BlockMicroTable<interm, readOnly, cpu> mtInClusterS1   ( a[i*3+1] );
        BlockMicroTable<interm, readOnly, cpu> mtInTargetFunc  ( a[i*3+2] );

        mtInClusterS0 .getBlockOfRows(0, nClusters, &inClusterS0);
        mtInClusterS1 .getBlockOfRows(0, nClusters, &inClusterS1);
        mtInTargetFunc.getBlockOfRows(0, 1,         &inTargetFunc);

        for(size_t j=0; j<nClusters; j++)
        {
            clusterS0[j] += inClusterS0[j];
        }

        for(size_t j=0; j<nClusters*p; j++)
        {
            clusterS1[j] += inClusterS1[j];
        }

        goalFunc[0] += inTargetFunc[0];

        mtInClusterS0 .release();
        mtInClusterS1 .release();
        mtInTargetFunc.release();
    }

    mtClusterS0   .release();
    mtClusterS1   .release();
    mtTargetFunc  .release();
}

template <Method method, typename interm, CpuType cpu>
void KMeansDistributedStep2Kernel<method, interm, cpu>::finalizeCompute( size_t na, const NumericTable *const *a,
                                                                              size_t nr, const NumericTable *const *r, const Parameter *par)
{
    BlockMicroTable<int,    readOnly, cpu> mtClusterS0   ( a[0] );
    BlockMicroTable<interm, readOnly, cpu> mtClusterS1   ( a[1] );
    BlockMicroTable<interm, readOnly, cpu> mtInTargetFunc( a[2] );

    BlockMicroTable<interm, writeOnly, cpu> mtClusters ( r[0] );
    BlockMicroTable<interm, writeOnly, cpu> mtTargetFunct ( r[1] );

    size_t nBlocks = na/3;

    size_t p = mtClusterS1.getFullNumberOfColumns();
    size_t nClusters = par->nClusters;

    /* TODO: That should be size_t or double */
    int    *clusterS0;
    interm *clusterS1;
    interm *inTarget;

    interm *clusters;
    interm *outTarget;


    mtClusterS0.getBlockOfRows(0, nClusters, &clusterS0);
    mtClusterS1.getBlockOfRows(0, nClusters, &clusterS1);
    mtClusters .getBlockOfRows(0, nClusters, &clusters);

    mtInTargetFunc.getBlockOfRows(0, 1, &inTarget);
    mtTargetFunct .getBlockOfRows(0, 1, &outTarget);

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

    *outTarget = *inTarget;

    mtClusterS0.release();
    mtClusterS1.release();
    mtClusters .release();

    mtInTargetFunc.release();
    mtTargetFunct .release();
}

} // namespace daal::algorithms::kmeans::internal
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
