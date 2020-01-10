/* file: covariance_dense_batch_oneapi_impl.i */
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

#ifndef __COVARIANCE_DENSE_BATCH_ONEAPI_IMPL_I__
#define __COVARIANCE_DENSE_BATCH_ONEAPI_IMPL_I__

#include "covariance_kernel_oneapi.h"
#include "covariance_oneapi_impl.i"

using namespace daal::oneapi::internal;

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
services::Status CovarianceDenseBatchKernelOneAPI<algorithmFPType, method>::compute(NumericTable * dataTable, NumericTable * covTable,
                                                                                    NumericTable * meanTable, const Parameter * parameter)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(computeDenseBatch);

    services::Status status;

    const size_t nFeatures              = dataTable->getNumberOfColumns();
    const size_t nVectors               = dataTable->getNumberOfRows();
    const algorithmFPType nObservations = static_cast<algorithmFPType>(nVectors);

    BlockDescriptor<algorithmFPType> dataBlock;
    BlockDescriptor<algorithmFPType> sumBlock;
    BlockDescriptor<algorithmFPType> crossProductBlock;

    {
        status |= dataTable->getBlockOfRows(0, nVectors, readOnly, dataBlock);
        DAAL_CHECK_STATUS_VAR(status);

        status |= meanTable->getBlockOfRows(0, meanTable->getNumberOfRows(), writeOnly, sumBlock);
        DAAL_CHECK_STATUS_VAR(status);

        status |= covTable->getBlockOfRows(0, covTable->getNumberOfRows(), writeOnly, crossProductBlock);
        DAAL_CHECK_STATUS_VAR(status);
    }

    status |= calculateCrossProductAndSums<algorithmFPType, method>(dataTable, crossProductBlock.getBuffer(), sumBlock.getBuffer());
    DAAL_CHECK_STATUS_VAR(status);

    status |= finalizeCovariance<algorithmFPType, method>(nFeatures, nObservations, crossProductBlock.getBuffer(), sumBlock.getBuffer(),
                                                          crossProductBlock.getBuffer(), sumBlock.getBuffer(), parameter);

    {
        status |= dataTable->releaseBlockOfRows(dataBlock);
        DAAL_CHECK_STATUS_VAR(status);

        status |= meanTable->releaseBlockOfRows(sumBlock);
        DAAL_CHECK_STATUS_VAR(status);

        status |= covTable->releaseBlockOfRows(crossProductBlock);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

} // namespace internal
} // namespace oneapi
} // namespace covariance
} // namespace algorithms
} // namespace daal

#endif
