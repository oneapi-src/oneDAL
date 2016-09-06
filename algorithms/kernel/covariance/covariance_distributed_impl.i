/* file: covariance_distributed_impl.i */
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
//  Covariance matrix computation algorithm implementation in distributed mode
//--
*/

#ifndef __COVARIANCE_DISTRIBUTED_IMPL_I__
#define __COVARIANCE_DISTRIBUTED_IMPL_I__

#include "covariance_kernel.h"
#include "covariance_impl.i"

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void CovarianceDistributedKernel<algorithmFPType, method, cpu>::compute(
            DataCollection *partialResultsCollection,
            NumericTable *nObservationsTable, NumericTable *crossProductTable,
            NumericTable *sumTable, const Parameter *parameter)
{
    size_t collectionSize = partialResultsCollection->size();

    size_t nFeatures = crossProductTable->getNumberOfColumns();

    BlockDescriptor<algorithmFPType> crossProductBD, sumBD, nObservationsBD;
    algorithmFPType *crossProduct, *sums, *nObservations;
    getDenseCrossProductAndSums<algorithmFPType, cpu>(writeOnly,
        crossProductTable, crossProductBD, &crossProduct, sumTable, sumBD, &sums,
        nObservationsTable, nObservationsBD, &nObservations);

    algorithmFPType zero = 0.0;
    daal::services::internal::service_memset<algorithmFPType, cpu>(crossProduct, zero, nFeatures * nFeatures);
    daal::services::internal::service_memset<algorithmFPType, cpu>(sums, zero, nFeatures);
    *nObservations = zero;

    NumericTable *partialCrossProductTable, *partialSumsTable, *partialNObservationsTable;
    BlockDescriptor<algorithmFPType> partialCrossProductBD, partialSumBD, partialNObservationsBD;
    algorithmFPType *partialCrossProduct, *partialSums, *partialNObservations;
    for (size_t i = 0; i < collectionSize; i++)
    {
        PartialResult* patrialResult = static_cast<PartialResult*>((*partialResultsCollection)[i].get());
        partialCrossProductTable  = patrialResult->get(covariance::crossProduct).get();
        partialSumsTable          = patrialResult->get(covariance::sum).get();
        partialNObservationsTable = patrialResult->get(covariance::nObservations).get();
        getDenseCrossProductAndSums<algorithmFPType, cpu>(readOnly,
            partialCrossProductTable, partialCrossProductBD, &partialCrossProduct,
            partialSumsTable, partialSumBD, &partialSums,
            partialNObservationsTable, partialNObservationsBD, &partialNObservations);

        mergeCrossProductAndSums<algorithmFPType, cpu>(nFeatures, partialCrossProduct,
            partialSums, partialNObservations, crossProduct, sums, nObservations);

        releaseDenseCrossProductAndSums<algorithmFPType, cpu>(partialCrossProductTable, partialCrossProductBD,
            partialSumsTable, partialSumBD, partialNObservationsTable, partialNObservationsBD);
    }
    releaseDenseCrossProductAndSums<algorithmFPType, cpu>(crossProductTable, crossProductBD, sumTable, sumBD,
        nObservationsTable, nObservationsBD);
}

template<typename algorithmFPType, Method method, CpuType cpu>
void CovarianceDistributedKernel<algorithmFPType, method, cpu>::finalizeCompute(
            NumericTable *nObservationsTable, NumericTable *crossProductTable,
            NumericTable *sumTable, NumericTable *covTable,
            NumericTable *meanTable, const Parameter *parameter)
{
    finalizeCovariance<algorithmFPType, cpu>(crossProductTable, sumTable, nObservationsTable,
        covTable, meanTable, parameter, this->_errors.get());
}

}
}
}
}

#endif
