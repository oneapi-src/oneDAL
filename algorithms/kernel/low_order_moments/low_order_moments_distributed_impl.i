/* file: low_order_moments_distributed_impl.i */
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
//  Implementation of LowOrderMomentsDistributedKernel
//--
*/

#ifndef __LOW_ORDER_MOMENTS_DISTRIBUTED_IMPL_I__
#define __LOW_ORDER_MOMENTS_DISTRIBUTED_IMPL_I__

#include "low_order_moments_kernel.h"
#include "low_order_moments_impl.i"

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{
namespace internal
{
using namespace daal::services;
template<typename algorithmFPType, Method method, CpuType cpu>
services::Status LowOrderMomentsDistributedKernel<algorithmFPType, method, cpu>::compute(
            data_management::DataCollection *partialResultsCollection,
            PartialResult *partialResult, const Parameter *parameter)
{
    TArray<int, cpu> partialNObservations(partialResultsCollection->size());
    if (!partialNObservations.get())
        return Status(services::ErrorMemoryAllocationFailed);

    mergeNObservations<algorithmFPType, cpu>(partialResultsCollection, partialResult, partialNObservations.get());
    mergeMinAndMax<algorithmFPType, cpu>(partialResultsCollection, partialResult);
    mergeSums<algorithmFPType, cpu>(partialResultsCollection, partialResult, partialNObservations.get());
    return Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status LowOrderMomentsDistributedKernel<algorithmFPType, method, cpu>::finalizeCompute(
            NumericTable *nObservationsTable,
            NumericTable *sumTable, NumericTable *sumSqTable, NumericTable *sumSqCenTable,
            NumericTable *meanTable, NumericTable *raw2MomTable, NumericTable *varianceTable,
            NumericTable *stDevTable, NumericTable *variationTable,
            const Parameter *parameter)
{
    LowOrderMomentsFinalizeTask<algorithmFPType, cpu> task(
        nObservationsTable, sumTable, sumSqTable, sumSqCenTable, meanTable,
        raw2MomTable, varianceTable, stDevTable, variationTable);

    finalize<algorithmFPType, cpu>(task);
    return Status();
}

}
}
}
}

#endif
