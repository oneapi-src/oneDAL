/* file: covariance_dense_distr_step2_oneapi_impl.i */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef __COVARIANCE_DENSE_DISTR_STEP2_ONEAPI_IMPL_I__
#define __COVARIANCE_DENSE_DISTR_STEP2_ONEAPI_IMPL_I__

#include "src/algorithms/covariance/oneapi/covariance_dense_distr_step2_oneapi.h"
#include "src/algorithms/covariance/oneapi/covariance_oneapi_impl.i"

using namespace daal::services::internal::sycl;

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace oneapi
{
namespace internal
{
template <typename algorithmFPType, Method method>
services::Status CovarianceDenseDistrStep2KernelOneAPI<algorithmFPType, method>::compute(DataCollection * partialResultsCollection,
                                                                                         NumericTable * nObsTable, NumericTable * crossProductTable,
                                                                                         NumericTable * sumTable, const Parameter * parameter)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(computeDistr);

    auto & context = services::internal::getDefaultContext();

    services::Status status;

    const size_t collectionSize = partialResultsCollection->size();
    const size_t nFeatures      = crossProductTable->getNumberOfColumns();

    BlockDescriptor<algorithmFPType> sumBlock;
    BlockDescriptor<algorithmFPType> crossProductBlock;
    BlockDescriptor<algorithmFPType> nObservationsBlock;

    status |= sumTable->getBlockOfRows(0, sumTable->getNumberOfRows(), readWrite, sumBlock);
    DAAL_CHECK_STATUS_VAR(status);
    status |= crossProductTable->getBlockOfRows(0, crossProductTable->getNumberOfRows(), readWrite, crossProductBlock);
    DAAL_CHECK_STATUS_VAR(status);
    status |= nObsTable->getBlockOfRows(0, nObsTable->getNumberOfRows(), readWrite, nObservationsBlock);
    DAAL_CHECK_STATUS_VAR(status);

    const algorithmFPType zero = 0.0;
    context.fill(sumBlock.getBuffer(), zero, status);
    DAAL_CHECK_STATUS_VAR(status);
    context.fill(crossProductBlock.getBuffer(), zero, status);
    DAAL_CHECK_STATUS_VAR(status);
    context.fill(nObservationsBlock.getBuffer(), zero, status);
    DAAL_CHECK_STATUS_VAR(status);

    for (size_t i = 0; i < collectionSize; i++)
    {
        PartialResult * patrialResult            = static_cast<PartialResult *>((*partialResultsCollection)[i].get());
        NumericTable * partialSumsTable          = patrialResult->get(covariance::sum).get();
        NumericTable * partialCrossProductTable  = patrialResult->get(covariance::crossProduct).get();
        NumericTable * partialNObservationsTable = patrialResult->get(covariance::nObservations).get();

        BlockDescriptor<algorithmFPType> partialSumsBlock;
        BlockDescriptor<algorithmFPType> partialCrossProductBlock;
        BlockDescriptor<algorithmFPType> partialNObservationsBlock;

        status |= partialSumsTable->getBlockOfRows(0, partialSumsTable->getNumberOfRows(), readWrite, partialSumsBlock);
        DAAL_CHECK_STATUS_VAR(status);
        status |= partialCrossProductTable->getBlockOfRows(0, partialCrossProductTable->getNumberOfRows(), readWrite, partialCrossProductBlock);
        DAAL_CHECK_STATUS_VAR(status);
        status |= partialNObservationsTable->getBlockOfRows(0, partialNObservationsTable->getNumberOfRows(), readWrite, partialNObservationsBlock);
        DAAL_CHECK_STATUS_VAR(status);

        if (i == 0)
        {
            context.copy(crossProductBlock.getBuffer(), 0, partialCrossProductBlock.getBuffer(), 0, nFeatures * nFeatures, status);
            DAAL_CHECK_STATUS_VAR(status);
            context.copy(sumBlock.getBuffer(), 0, partialSumsBlock.getBuffer(), 0, nFeatures, status);
            DAAL_CHECK_STATUS_VAR(status);
        }
        else
        {
            const auto partialNObservationsBlockHost = partialNObservationsBlock.getBuffer().toHost(data_management::readOnly, status);
            DAAL_CHECK_STATUS_VAR(status);

            const auto nObservationsBlockHost = nObservationsBlock.getBuffer().toHost(data_management::readOnly, status);
            DAAL_CHECK_STATUS_VAR(status);

            status |= mergeCrossProduct<algorithmFPType>(nFeatures, partialCrossProductBlock.getBuffer(), partialSumsBlock.getBuffer(),
                                                         *partialNObservationsBlockHost, crossProductBlock.getBuffer(), sumBlock.getBuffer(),
                                                         *nObservationsBlockHost);
            DAAL_CHECK_STATUS_VAR(status);

            status |= mergeSums<algorithmFPType, method>(nFeatures, partialSumsBlock.getBuffer(), sumBlock.getBuffer());
            DAAL_CHECK_STATUS_VAR(status);
        }

        status |= BlasGpu<algorithmFPType>::xaxpy(1, 1, partialNObservationsBlock.getBuffer(), 1, nObservationsBlock.getBuffer(), 1);

        status |= partialSumsTable->releaseBlockOfRows(partialSumsBlock);
        DAAL_CHECK_STATUS_VAR(status);
        status |= partialCrossProductTable->releaseBlockOfRows(partialCrossProductBlock);
        DAAL_CHECK_STATUS_VAR(status);
        status |= partialNObservationsTable->releaseBlockOfRows(partialNObservationsBlock);
        DAAL_CHECK_STATUS_VAR(status);
    }

    status |= sumTable->releaseBlockOfRows(sumBlock);
    DAAL_CHECK_STATUS_VAR(status);
    status |= crossProductTable->releaseBlockOfRows(crossProductBlock);
    DAAL_CHECK_STATUS_VAR(status);
    status |= nObsTable->releaseBlockOfRows(nObservationsBlock);
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

template <typename algorithmFPType, Method method>
services::Status CovarianceDenseDistrStep2KernelOneAPI<algorithmFPType, method>::finalizeCompute(NumericTable * nObservationsTable,
                                                                                                 NumericTable * crossProductTable,
                                                                                                 NumericTable * sumTable, NumericTable * covTable,
                                                                                                 NumericTable * meanTable,
                                                                                                 const Parameter * parameter)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(finalizeComputeDistr);

    services::Status status;

    const size_t nFeatures = crossProductTable->getNumberOfColumns();

    BlockDescriptor<algorithmFPType> sumBlock;
    BlockDescriptor<algorithmFPType> covBlock;
    BlockDescriptor<algorithmFPType> meanBlock;
    BlockDescriptor<algorithmFPType> crossProductBlock;
    BlockDescriptor<algorithmFPType> nObservationsBlock;

    status |= sumTable->getBlockOfRows(0, sumTable->getNumberOfRows(), readWrite, sumBlock);
    DAAL_CHECK_STATUS_VAR(status);
    status |= covTable->getBlockOfRows(0, covTable->getNumberOfRows(), readWrite, covBlock);
    DAAL_CHECK_STATUS_VAR(status);
    status |= meanTable->getBlockOfRows(0, meanTable->getNumberOfRows(), readWrite, meanBlock);
    DAAL_CHECK_STATUS_VAR(status);
    status |= crossProductTable->getBlockOfRows(0, crossProductTable->getNumberOfRows(), readWrite, crossProductBlock);
    DAAL_CHECK_STATUS_VAR(status);
    status |= nObservationsTable->getBlockOfRows(0, nObservationsTable->getNumberOfRows(), readWrite, nObservationsBlock);
    DAAL_CHECK_STATUS_VAR(status);

    const auto nObservationsBlockHost = nObservationsBlock.getBuffer().toHost(data_management::readOnly, status);
    DAAL_CHECK_STATUS_VAR(status);

    status |= finalizeCovariance<algorithmFPType, method>(nFeatures, *nObservationsBlockHost, crossProductBlock.getBuffer(), sumBlock.getBuffer(),
                                                          covBlock.getBuffer(), meanBlock.getBuffer(), parameter);
    DAAL_CHECK_STATUS_VAR(status);

    status |= sumTable->releaseBlockOfRows(sumBlock);
    DAAL_CHECK_STATUS_VAR(status);
    status |= crossProductTable->releaseBlockOfRows(crossProductBlock);
    DAAL_CHECK_STATUS_VAR(status);
    status |= nObservationsTable->releaseBlockOfRows(nObservationsBlock);
    DAAL_CHECK_STATUS_VAR(status);
    status |= meanTable->releaseBlockOfRows(meanBlock);
    DAAL_CHECK_STATUS_VAR(status);
    status |= covTable->releaseBlockOfRows(covBlock);
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

} // namespace internal
} // namespace oneapi
} // namespace covariance
} // namespace algorithms
} // namespace daal

#endif
