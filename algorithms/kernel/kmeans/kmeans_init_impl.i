/* file: kmeans_init_impl.i */
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
#include "service_rng.h"

using namespace daal::internal;
using namespace daal::services::internal;

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

template <Method method, typename interm, CpuType cpu>
bool init( size_t p, size_t n, size_t nRowsTotal, size_t nClusters, size_t offset, interm *clusters,
    BlockMicroTable<interm, readOnly, cpu> &mtData, unsigned int seed, size_t& clustersFound)
{
    interm *data;

    if(method == deterministicDense || method == deterministicCSR)
    {
        if( nClusters <= offset )
        {
            clustersFound = 0;
            return true;
        }

        nClusters -= offset;
        mtData.getBlockOfRows( 0, nClusters, &data );

        for (size_t i = 0; i < nClusters && i < n; i++)
        {
            for (size_t j = 0; j < p; j++)
            {
                clusters[i * p + j] = data[i * p + j];
            }
        }

        mtData.release();

        clustersFound = nClusters;
    }
    else if(method == randomDense || method == randomCSR)
    {
        int *indices = (int *)daal::services::daal_malloc( sizeof(int) * nClusters );
        if( !indices )
        {
            return false;
        }

        IntRng<int, cpu> rng(seed);

        size_t k = 0;
        for(size_t i = 0; i < nClusters; i++)
        {
            rng.uniform(1, i, (int)nRowsTotal, &indices[i]);

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

            if( c<offset || c>=offset+n ) continue;

            mtData.getBlockOfRows( c-offset, 1, &data );

            for(size_t j = 0; j < p; j++)
            {
                clusters[k * p + j] = data[j];
            }
            k++;

            mtData.release();
        }
        clustersFound = k;

        daal::services::daal_free( indices );
    }

    return true;
}

template <Method method, typename interm, CpuType cpu>
void KMeansinitKernel<method, interm, cpu>::compute( size_t na, const NumericTable *const *a,
                                                     size_t nr, const NumericTable *const *r, const Parameter *par)
{
    NumericTable *ntData     = const_cast<NumericTable *>( a[0] );
    NumericTable *ntClusters = const_cast<NumericTable *>( r[0] );

    size_t p = ntData->getNumberOfColumns();
    size_t n = ntData->getNumberOfRows();
    size_t nClusters = par->nClusters;

    BlockMicroTable<interm, readOnly,  cpu> mtData( ntData );
    BlockMicroTable<interm, writeOnly, cpu> mtClusters( ntClusters );

    interm *clusters;

    mtClusters.getBlockOfRows( 0, nClusters, &clusters );

    size_t clustersFound = 0;
    if( !init<method, interm, cpu>( p, n, n, nClusters, 0, clusters, mtData, par->seed, clustersFound) )
    {
        this->_errors->add(services::ErrorMemoryAllocationFailed); return;
    }

    mtClusters.release();
}

template <Method method, typename interm, CpuType cpu>
void KMeansinitStep1LocalKernel<method, interm, cpu>::compute( size_t na, const NumericTable *const *a,
                                                                     size_t nr, const NumericTable *const *r, const Parameter *par)
{
    NumericTable *ntData     = const_cast<NumericTable *>( a[0] );
    NumericTable *ntClustersN= const_cast<NumericTable *>( r[0] );
    NumericTable *ntClusters = const_cast<NumericTable *>( r[1] );

    size_t p = ntData->getNumberOfColumns();
    size_t n = ntData->getNumberOfRows();
    size_t nClusters  = par->nClusters;
    size_t offset     = par->offset;
    size_t nRowsTotal = par->nRowsTotal;

    BlockMicroTable<interm, readOnly,  cpu> mtData( ntData );
    BlockMicroTable<interm, writeOnly, cpu> mtClusters( ntClusters );
    BlockMicroTable<int, writeOnly, cpu> mtClustersN( ntClustersN );

    interm *data;
    int    *clustersN;
    interm *clusters;

    mtClusters.getBlockOfRows( 0, nClusters, &clusters );
    mtClustersN.getBlockOfRows( 0, 1, &clustersN );

    size_t clustersFound = 0;
    if( !init<method, interm, cpu>( p, n, nRowsTotal, nClusters, offset, clusters, mtData, par->seed, clustersFound) )
    {
        this->_errors->add(services::ErrorMemoryAllocationFailed);
        mtClustersN.release();
        mtClusters.release();
        return;
    }
    *clustersN = (int)clustersFound;

    mtClustersN.release();
    mtClusters.release();
}

template <Method method, typename interm, CpuType cpu>
void KMeansinitStep1LocalKernel<method, interm, cpu>::finalizeCompute( size_t na, const NumericTable *const *a,
                                                                                 size_t nr, const NumericTable *const *r, const Parameter *par)
{}

template <Method method, typename interm, CpuType cpu>
void KMeansinitStep2MasterKernel<method, interm, cpu>::compute( size_t na, const NumericTable *const *a,
                                                                      size_t nr, const NumericTable *const *r, const Parameter *par)
{
    NumericTable *ntClustersN= const_cast<NumericTable *>( r[0] );
    NumericTable *ntClusters = const_cast<NumericTable *>( r[1] );

    size_t nBlocks = na / 2;
    size_t p = r[1]->getNumberOfColumns();
    size_t nClusters = par->nClusters;

    BlockMicroTable<interm, writeOnly, cpu> mtClusters( ntClusters );
    BlockMicroTable<int, writeOnly, cpu> mtClustersN( ntClustersN );

    int    *clustersN;
    interm *clusters;

    mtClusters.getBlockOfRows( 0, nClusters, &clusters );
    mtClustersN.getBlockOfRows( 0, 1, &clustersN );

    size_t k = 0;

    for( size_t i = 0; i<nBlocks; i++ )
    {
        BlockMicroTable<int,    readOnly, cpu> mtInClustersN( a[i*2 + 0] );
        BlockMicroTable<interm, readOnly, cpu> mtInClusters ( a[i*2 + 1] );

        int    *inClustersN;
        interm *inClusters;

        mtInClusters.getBlockOfRows( 0, nClusters, &inClusters );
        mtInClustersN.getBlockOfRows( 0, 1, &inClustersN );

        size_t inK = *inClustersN;
        for( size_t j=0; j<inK; j++ )
        {
            for( size_t h=0; h<p; h++ )
            {
                clusters[k*p + h] = inClusters[j*p + h];
            }
            k++;
        }

        mtInClustersN.release();
        mtInClusters.release();
    }

    *clustersN = k;

    mtClustersN.release();
    mtClusters.release();
}

template <Method method, typename interm, CpuType cpu>
void KMeansinitStep2MasterKernel<method, interm, cpu>::finalizeCompute( size_t na, const NumericTable *const *a,
                                                                              size_t nr, const NumericTable *const *r, const Parameter *par)
{
    NumericTable *ntInClusters  = const_cast<NumericTable *>( a[1] );
    NumericTable *ntClusters    = const_cast<NumericTable *>( r[0] );

    size_t p = ntClusters->getNumberOfColumns();
    size_t nClusters = par->nClusters;

    BlockMicroTable<interm, readOnly,  cpu> mtInClusters( ntInClusters );
    BlockMicroTable<interm, writeOnly, cpu> mtClusters( ntClusters );

    interm *inClusters;
    interm *clusters;

    mtInClusters.getBlockOfRows( 0, nClusters, &inClusters );
    mtClusters.getBlockOfRows( 0, nClusters, &clusters );

    for (size_t i = 0; i < nClusters; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            clusters[i * p + j] = inClusters[i * p + j];
        }
    }

    mtInClusters.release();
    mtClusters.release();
}

} // namespace daal::algorithms::kmeans::init::internal
} // namespace daal::algorithms::kmeans::init
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
