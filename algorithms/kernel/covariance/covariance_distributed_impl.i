/* file: covariance_distributed_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
services::Status CovarianceDistributedKernel<algorithmFPType, method, cpu>::compute(
    DataCollection *partialResultsCollection,
    NumericTable *nObservationsTable,
    NumericTable *crossProductTable,
    NumericTable *sumTable,
    const Parameter *parameter)
{
    const size_t collectionSize = partialResultsCollection->size();
    const size_t nFeatures      = crossProductTable->getNumberOfColumns();

    DEFINE_TABLE_BLOCK( WriteOnlyRows, sumBlock,           sumTable           );
    DEFINE_TABLE_BLOCK( WriteOnlyRows, crossProductBlock,  crossProductTable  );
    DEFINE_TABLE_BLOCK( WriteOnlyRows, nObservationsBlock, nObservationsTable );

    algorithmFPType *sums          = sumBlock.get();
    algorithmFPType *crossProduct  = crossProductBlock.get();
    algorithmFPType *nObservations = nObservationsBlock.get();

    algorithmFPType zero = 0.0;
    daal::services::internal::service_memset<algorithmFPType, cpu>(crossProduct, zero, nFeatures * nFeatures);
    daal::services::internal::service_memset<algorithmFPType, cpu>(sums, zero, nFeatures);
    *nObservations = zero;

    for (size_t i = 0; i < collectionSize; i++)
    {
        PartialResult *patrialResult = static_cast<PartialResult*>((*partialResultsCollection)[i].get());
        NumericTable  *partialSumsTable          = patrialResult->get(covariance::sum).get();
        NumericTable  *partialCrossProductTable  = patrialResult->get(covariance::crossProduct).get();
        NumericTable  *partialNObservationsTable = patrialResult->get(covariance::nObservations).get();

        DEFINE_TABLE_BLOCK( ReadRows, partialSumsBlock,          partialSumsTable          );
        DEFINE_TABLE_BLOCK( ReadRows, partialCrossProductBlock,  partialCrossProductTable  );
        DEFINE_TABLE_BLOCK( ReadRows, partialNObservationsBlock, partialNObservationsTable );

        mergeCrossProductAndSums<algorithmFPType, cpu>(nFeatures, partialCrossProductBlock.get(),
            partialSumsBlock.get(), partialNObservationsBlock.get(), crossProduct, sums, nObservations);
    }

    return services::Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status CovarianceDistributedKernel<algorithmFPType, method, cpu>::finalizeCompute(
    NumericTable *nObservationsTable,
    NumericTable *crossProductTable,
    NumericTable *sumTable,
    NumericTable *covTable,
    NumericTable *meanTable,
    const Parameter *parameter)
{
    return finalizeCovariance<algorithmFPType, cpu>(nObservationsTable,
        crossProductTable, sumTable, covTable, meanTable, parameter);
}

} // internal
} // covariance
} // algorithms
} // daal

#endif
