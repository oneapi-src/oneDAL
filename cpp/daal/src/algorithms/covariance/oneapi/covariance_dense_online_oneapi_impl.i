/* file: covariance_dense_online_oneapi_impl.i */
/*******************************************************************************
* Copyright 2019 Intel Corporation
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
//  Covariance matrix computation algorithm implementation in batch mode
//--
*/

#ifndef __COVARIANCE_DENSE_ONLINE_ONEAPI_IMPL_I__
#define __COVARIANCE_DENSE_ONLINE_ONEAPI_IMPL_I__

#include "src/algorithms/covariance/oneapi/covariance_kernel_oneapi.h"
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
template <typename algorithmFPType>
inline bool isFirstDataBlock(algorithmFPType nObservations)
{
    return (nObservations < static_cast<algorithmFPType>(0.5));
}

template <typename algorithmFPType, Method method>
services::Status CovarianceDenseOnlineKernelOneAPI<algorithmFPType, method>::compute(NumericTable * dataTable, NumericTable * nObsTable,
                                                                                     NumericTable * crossProductTable, NumericTable * sumTable,
                                                                                     const Parameter * parameter)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(computeDenseOnline);
    services::Status status;

    auto & context                       = Environment::getInstance()->getDefaultExecutionContext();
    algorithmFPType * nObservations      = nullptr;
    algorithmFPType partialNObservations = 0.0;

    if (dataTable->getNumberOfColumns() > static_cast<size_t>(services::internal::MaxVal<uint32_t>::get()))
    {
        return services::Status(daal::services::ErrorCovarianceInternal);
    }
    const uint32_t nFeatures = static_cast<uint32_t>(dataTable->getNumberOfColumns());

    if (dataTable->getNumberOfRows() > static_cast<size_t>(services::internal::MaxVal<uint32_t>::get()))
    {
        return services::Status(daal::services::ErrorCovarianceInternal);
    }
    const uint32_t nVectors = static_cast<uint32_t>(dataTable->getNumberOfRows());

    partialNObservations = static_cast<algorithmFPType>(nVectors);

    BlockDescriptor<algorithmFPType> dataBlock;
    BlockDescriptor<algorithmFPType> sumBlock;
    BlockDescriptor<algorithmFPType> crossProductBlock;
    BlockDescriptor<algorithmFPType> nObsBlock;

    {
        status |= dataTable->getBlockOfRows(0, nVectors, readWrite, dataBlock);
        DAAL_CHECK_STATUS_VAR(status);

        status |= sumTable->getBlockOfRows(0, sumTable->getNumberOfRows(), readWrite, sumBlock);
        DAAL_CHECK_STATUS_VAR(status);

        status |= crossProductTable->getBlockOfRows(0, crossProductTable->getNumberOfRows(), readWrite, crossProductBlock);
        DAAL_CHECK_STATUS_VAR(status);

        status |= nObsTable->getBlockOfRows(0, crossProductTable->getNumberOfRows(), readWrite, nObsBlock);
        DAAL_CHECK_STATUS_VAR(status);
        nObservations = nObsBlock.getBlockPtr();
        DAAL_ASSERT(nObservations != nullptr);
    }

    if (isFirstDataBlock(*nObservations))
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(crossProductBlock.getBuffer()), algorithmFPType, nFeatures * nFeatures);
        DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(sumBlock.getBuffer()), algorithmFPType, nFeatures);
        status |= calculateCrossProductAndSums<algorithmFPType, method>(dataTable, crossProductBlock.getBuffer(), sumBlock.getBuffer());
        DAAL_CHECK_STATUS_VAR(status);
    }
    else
    {
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, nFeatures, nFeatures);
        auto partialCrossProductBlock = context.allocate(TypeIds::id<algorithmFPType>(), nFeatures * nFeatures, status);
        DAAL_CHECK_STATUS_VAR(status);

        auto partialSumBlock = context.allocate(TypeIds::id<algorithmFPType>(), nFeatures, status);
        DAAL_CHECK_STATUS_VAR(status);

        status |= calculateCrossProductAndSums<algorithmFPType, method>(dataTable, partialCrossProductBlock.template get<algorithmFPType>(),
                                                                        partialSumBlock.template get<algorithmFPType>());
        DAAL_CHECK_STATUS_VAR(status);

        status |= mergeCrossProduct<algorithmFPType>(nFeatures, partialCrossProductBlock.template get<algorithmFPType>(),
                                                     partialSumBlock.template get<algorithmFPType>(), partialNObservations,
                                                     crossProductBlock.getBuffer(), sumBlock.getBuffer(), *nObservations);
        DAAL_CHECK_STATUS_VAR(status);

        status |= mergeSums<algorithmFPType, method>(nFeatures, partialSumBlock.template get<algorithmFPType>(), sumBlock.getBuffer());
        DAAL_CHECK_STATUS_VAR(status);
    }

    *nObservations += partialNObservations;

    DAAL_CHECK_STATUS_VAR(dataTable->releaseBlockOfRows(dataBlock));
    DAAL_CHECK_STATUS_VAR(sumTable->releaseBlockOfRows(sumBlock));
    DAAL_CHECK_STATUS_VAR(crossProductTable->releaseBlockOfRows(crossProductBlock));
    DAAL_CHECK_STATUS_VAR(nObsTable->releaseBlockOfRows(nObsBlock));

    return status;
}

template <typename algorithmFPType, Method method>
services::Status CovarianceDenseOnlineKernelOneAPI<algorithmFPType, method>::finalizeCompute(NumericTable * nObservationsTable,
                                                                                             NumericTable * crossProductTable,
                                                                                             NumericTable * sumTable, NumericTable * covTable,
                                                                                             NumericTable * meanTable, const Parameter * parameter)
{
    services::Status status;

    if (crossProductTable->getNumberOfColumns() > static_cast<size_t>(services::internal::MaxVal<uint32_t>::get()))
    {
        return services::Status(daal::services::ErrorCovarianceInternal);
    }

    const uint32_t nFeatures        = static_cast<uint32_t>(crossProductTable->getNumberOfColumns());
    algorithmFPType * nObservations = nullptr;

    BlockDescriptor<algorithmFPType> dataBlock;
    BlockDescriptor<algorithmFPType> sumBlock;
    BlockDescriptor<algorithmFPType> meanBlock;
    BlockDescriptor<algorithmFPType> covBlock;
    BlockDescriptor<algorithmFPType> crossProductBlock;
    BlockDescriptor<algorithmFPType> nObservationsBlock;

    {
        status |= sumTable->getBlockOfRows(0, sumTable->getNumberOfRows(), readWrite, sumBlock);
        DAAL_CHECK_STATUS_VAR(status);

        status |= crossProductTable->getBlockOfRows(0, crossProductTable->getNumberOfRows(), readWrite, crossProductBlock);
        DAAL_CHECK_STATUS_VAR(status);

        status |= covTable->getBlockOfRows(0, covTable->getNumberOfRows(), readWrite, covBlock);
        DAAL_CHECK_STATUS_VAR(status);

        status |= meanTable->getBlockOfRows(0, meanTable->getNumberOfRows(), readWrite, meanBlock);
        DAAL_CHECK_STATUS_VAR(status);

        status |= nObservationsTable->getBlockOfRows(0, crossProductTable->getNumberOfRows(), readWrite, nObservationsBlock);
        DAAL_CHECK_STATUS_VAR(status);
        nObservations = nObservationsBlock.getBlockPtr();
        DAAL_ASSERT(nObservations != nullptr);
    }

    status |= finalizeCovariance<algorithmFPType, method>(nFeatures, *nObservations, crossProductBlock.getBuffer(), sumBlock.getBuffer(),
                                                          covBlock.getBuffer(), meanBlock.getBuffer(), parameter);
    DAAL_CHECK_STATUS_VAR(status);

    DAAL_CHECK_STATUS_VAR(sumTable->releaseBlockOfRows(sumBlock));
    DAAL_CHECK_STATUS_VAR(crossProductTable->releaseBlockOfRows(crossProductBlock));
    DAAL_CHECK_STATUS_VAR(nObservationsTable->releaseBlockOfRows(nObservationsBlock));
    DAAL_CHECK_STATUS_VAR(meanTable->releaseBlockOfRows(meanBlock));
    DAAL_CHECK_STATUS_VAR(covTable->releaseBlockOfRows(covBlock));

    return status;
}

} // namespace internal
} // namespace oneapi
} // namespace covariance
} // namespace algorithms
} // namespace daal

#endif
