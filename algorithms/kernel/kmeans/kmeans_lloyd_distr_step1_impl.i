/* file: kmeans_lloyd_distr_step1_impl.i */
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

#define __DAAL_FABS(a) (((a)>(algorithmFPType)0.0)?(a):(-(a)))

template <Method method, typename algorithmFPType, CpuType cpu>
Status KMeansDistributedStep1Kernel<method, algorithmFPType, cpu>::compute( size_t na, const NumericTable *const *a,
                                                                                      size_t nr, const NumericTable *const *r, const Parameter *par)
{
    NumericTable *ntData = const_cast<NumericTable *>(a[0]);
    NumericTable* ntAssignments = const_cast<NumericTable*>(r[5]);

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

    WriteOnlyRows<algorithmFPType, cpu> mtCValues(*const_cast<NumericTable*>(r[3]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtCValues);
    algorithmFPType *cValues = mtCValues.get();
    WriteOnlyRows<algorithmFPType, cpu> mtCCentroids(*const_cast<NumericTable*>(r[4]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtCCentroids);
    algorithmFPType *cCentroids = mtCCentroids.get();

    /* Categorial variables check and support: begin */
    int catFlag = 0;
    for(size_t i = 0; i < p; i++)
    {
        if(ntData->getFeatureType(i) == features::DAAL_CATEGORICAL)
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
            if(ntData->getFeatureType(i) == features::DAAL_CATEGORICAL)
            {
                catCoef[i] = par->gamma;
            }
            else
            {
                catCoef[i] = (algorithmFPType)1.0;
            }
        }
    }

    TArray<size_t, cpu> cIndices(nClusters);
    DAAL_CHECK_MALLOC(cIndices.get());

    Status s;
    algorithmFPType oldTargetFunc = (algorithmFPType)0.0;
    {
        SharedPtr<task_t<algorithmFPType, cpu> > task = task_t<algorithmFPType, cpu>::create(p, nClusters, initClusters);
        DAAL_CHECK(task.get(), services::ErrorMemoryAllocationFailed);
        DAAL_ASSERT(task);

        if( par->assignFlag )
        {
            s = task->template addNTToTaskThreaded<method>(ntData, catCoef.get(), ntAssignments);
        }
        else
        {
            s = task->template addNTToTaskThreaded<method>(ntData, catCoef.get());
        }
        if(!s)
        {
            task->kmeansClearClusters(&oldTargetFunc);
            return s;
        }

        TArray<double, cpu> dS1(method == defaultDense ? p : 0);
        if (method == defaultDense)
        {
            DAAL_CHECK(dS1.get(), services::ErrorMemoryAllocationFailed);
        }

        task->template kmeansComputeCentroids<method>(clusterS0, clusterS1, dS1.get());

        size_t cNum;
        DAAL_CHECK_STATUS(s, task->kmeansComputeCentroidsCandidates(cValues, cIndices.get(), cNum));
        for (size_t i = 0; i < cNum; i++)
        {
            ReadRows<algorithmFPType, cpu> mtRow(ntData, cIndices.get()[i], 1);
            const algorithmFPType *row = mtRow.get();
            daal::services::daal_memcpy_s(&cCentroids[i * p], p * sizeof(algorithmFPType), row, p * sizeof(algorithmFPType));
        }
        for (size_t i = cNum; i < nClusters; i++)
        {
            cValues[i] = (algorithmFPType)-1.0;
        }

        task->kmeansClearClusters(goalFunc);
    }
    return s;
}

template <Method method, typename algorithmFPType, CpuType cpu>
Status KMeansDistributedStep1Kernel<method, algorithmFPType, cpu>::finalizeCompute( size_t na, const NumericTable *const *a,
                                                                                              size_t nr, const NumericTable *const *r, const Parameter *par)
{
    if(!par->assignFlag)
        return Status();

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
    return Status();
}

} // namespace daal::algorithms::kmeans::internal
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
