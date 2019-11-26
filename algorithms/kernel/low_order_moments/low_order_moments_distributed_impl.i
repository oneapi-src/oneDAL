/* file: low_order_moments_distributed_impl.i */
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
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status LowOrderMomentsDistributedKernel<algorithmFPType, method, cpu>::compute(data_management::DataCollection * partialResultsCollection,
                                                                                         PartialResult * partialResult, const Parameter * parameter)
{
    services::Status status;

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, partialResultsCollection->size(), sizeof(int));

    TArray<int, cpu> partialNObservations(partialResultsCollection->size());
    if (!partialNObservations.get()) return Status(services::ErrorMemoryAllocationFailed);

    mergeNObservations<algorithmFPType, cpu>(partialResultsCollection, partialResult, partialNObservations.get());
    status |= mergeMinAndMax<algorithmFPType, cpu>(partialResultsCollection, partialResult);
    status |= mergeSums<algorithmFPType, cpu>(partialResultsCollection, partialResult, partialNObservations.get());
    return status;
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status LowOrderMomentsDistributedKernel<algorithmFPType, method, cpu>::finalizeCompute(
    NumericTable * nObservationsTable, NumericTable * sumTable, NumericTable * sumSqTable, NumericTable * sumSqCenTable, NumericTable * meanTable,
    NumericTable * raw2MomTable, NumericTable * varianceTable, NumericTable * stDevTable, NumericTable * variationTable, const Parameter * parameter)
{
    LowOrderMomentsFinalizeTask<algorithmFPType, cpu> task(nObservationsTable, sumTable, sumSqTable, sumSqCenTable, meanTable, raw2MomTable,
                                                           varianceTable, stDevTable, variationTable);

    finalize<algorithmFPType, cpu>(task);
    return Status();
}

} // namespace internal
} // namespace low_order_moments
} // namespace algorithms
} // namespace daal

#endif
