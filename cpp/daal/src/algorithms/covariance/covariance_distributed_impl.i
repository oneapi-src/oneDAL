/* file: covariance_distributed_impl.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Covariance matrix computation algorithm implementation in distributed mode
//--
*/

#ifndef __COVARIANCE_DISTRIBUTED_IMPL_I__
#define __COVARIANCE_DISTRIBUTED_IMPL_I__

#include "src/algorithms/covariance/covariance_kernel.h"
#include "src/algorithms/covariance/covariance_impl.i"

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace internal
{
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status CovarianceDistributedKernel<algorithmFPType, method, cpu>::compute(DataCollection * partialResultsCollection,
                                                                                    NumericTable * nObservationsTable,
                                                                                    NumericTable * crossProductTable, NumericTable * sumTable,
                                                                                    const Parameter * parameter,
                                                                                    const Hyperparameter * hyperparameter)
{
    const size_t collectionSize = partialResultsCollection->size();
    const size_t nFeatures      = crossProductTable->getNumberOfColumns();

    DEFINE_TABLE_BLOCK(WriteOnlyRows, sumBlock, sumTable);
    DEFINE_TABLE_BLOCK(WriteOnlyRows, crossProductBlock, crossProductTable);
    DEFINE_TABLE_BLOCK(WriteOnlyRows, nObservationsBlock, nObservationsTable);

    algorithmFPType * sums          = sumBlock.get();
    algorithmFPType * crossProduct  = crossProductBlock.get();
    algorithmFPType * nObservations = nObservationsBlock.get();

    algorithmFPType zero = 0.0;
    daal::services::internal::service_memset<algorithmFPType, cpu>(crossProduct, zero, nFeatures * nFeatures);
    daal::services::internal::service_memset<algorithmFPType, cpu>(sums, zero, nFeatures);
    *nObservations = zero;

    for (size_t i = 0; i < collectionSize; i++)
    {
        PartialResult * patrialResult            = static_cast<PartialResult *>((*partialResultsCollection)[i].get());
        NumericTable * partialSumsTable          = patrialResult->get(covariance::sum).get();
        NumericTable * partialCrossProductTable  = patrialResult->get(covariance::crossProduct).get();
        NumericTable * partialNObservationsTable = patrialResult->get(covariance::nObservations).get();

        DEFINE_TABLE_BLOCK(ReadRows, partialSumsBlock, partialSumsTable);
        DEFINE_TABLE_BLOCK(ReadRows, partialCrossProductBlock, partialCrossProductTable);
        DEFINE_TABLE_BLOCK(ReadRows, partialNObservationsBlock, partialNObservationsTable);

        mergeCrossProductAndSums<algorithmFPType, cpu>(nFeatures, partialCrossProductBlock.get(), partialSumsBlock.get(),
                                                       partialNObservationsBlock.get(), crossProduct, sums, nObservations, hyperparameter);
    }

    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status CovarianceDistributedKernel<algorithmFPType, method, cpu>::finalizeCompute(NumericTable * nObservationsTable,
                                                                                            NumericTable * crossProductTable, NumericTable * sumTable,
                                                                                            NumericTable * covTable, NumericTable * meanTable,
                                                                                            const Parameter * parameter,
                                                                                            const Hyperparameter * hyperparameter)
{
    return finalizeCovariance<algorithmFPType, cpu>(nObservationsTable, crossProductTable, sumTable, covTable, meanTable, parameter, hyperparameter);
}

} // namespace internal
} // namespace covariance
} // namespace algorithms
} // namespace daal

#endif
