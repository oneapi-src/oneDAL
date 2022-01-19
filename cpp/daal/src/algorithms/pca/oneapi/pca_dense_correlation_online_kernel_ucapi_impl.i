/* file: pca_dense_correlation_online_kernel_ucapi_impl.i */
/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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
//  Implementation of PCA Online Kernel for GPU.
//--
*/

#ifndef __PCA_DENSE_CORRELATION_ONLINE_KERNEL_UCAPI_IMPL__
#define __PCA_DENSE_CORRELATION_ONLINE_KERNEL_UCAPI_IMPL__

#include "src/externals/service_profiler.h"

#include "include/services/env_detect.h"
#include "include/services/internal/sycl/types.h"
#include "src/algorithms/covariance/oneapi/covariance_oneapi_impl.i"
#include "pca_dense_correlation_online_kernel_ucapi.h"

using namespace daal::services;
using namespace daal::internal;
using namespace daal::services::internal::sycl;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{
template <typename algorithmFPType>
PCACorrelationKernelOnlineUCAPI<algorithmFPType>::PCACorrelationKernelOnlineUCAPI(const PCACorrelationBaseIfacePtr & host_impl)
{
    _host_impl = host_impl;
}

template <typename algorithmFPType>
Status PCACorrelationKernelOnlineUCAPI<algorithmFPType>::compute(const data_management::NumericTablePtr & pData,
                                                                 PartialResult<correlationDense> * partialResult,
                                                                 const OnlineParameter<algorithmFPType, correlationDense> * parameter)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute);
    parameter->covariance->input.set(covariance::data, pData);
    parameter->covariance->parameter.outputMatrixType = covariance::correlationMatrix;

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.covariance);
        DAAL_CHECK_STATUS_VAR(parameter->covariance->computeNoThrow());
    }
    DAAL_CHECK_STATUS_VAR(copyCovarianceResultToPartialResult(parameter->covariance->getPartialResult().get(), partialResult));

    return Status();
}

template <typename algorithmFPType>
services::Status PCACorrelationKernelOnlineUCAPI<algorithmFPType>::copyCovarianceResultToPartialResult(
    const covariance::PartialResult * covariancePres, PartialResult<correlationDense> * partialResult)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.copy.to.partial.result);
    DAAL_CHECK_STATUS_VAR(copyIfNeeded(covariancePres->get(covariance::sum).get(), partialResult->get(sumCorrelation).get()));
    DAAL_CHECK_STATUS_VAR(copyIfNeeded(covariancePres->get(covariance::nObservations).get(), partialResult->get(nObservationsCorrelation).get()));
    DAAL_CHECK_STATUS_VAR(copyIfNeeded(covariancePres->get(covariance::crossProduct).get(), partialResult->get(crossProductCorrelation).get()));
    return Status();
}

template <typename algorithmFPType>
services::Status PCACorrelationKernelOnlineUCAPI<algorithmFPType>::copyIfNeeded(const data_management::NumericTable * src,
                                                                                data_management::NumericTable * dst)
{
    if (src == dst) return services::Status();

    DAAL_ASSERT(dst->getNumberOfRows() == src->getNumberOfRows());
    DAAL_ASSERT(dst->getNumberOfColumns() == src->getNumberOfColumns());

    BlockDescriptor<algorithmFPType> srcBlock;
    BlockDescriptor<algorithmFPType> dstBlock;

    {
        DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(src)->getBlockOfRows(0, src->getNumberOfRows(), readOnly, srcBlock));
        DAAL_CHECK_STATUS_VAR(dst->getBlockOfRows(0, dst->getNumberOfRows(), writeOnly, dstBlock));
    }

    const size_t nRows         = dst->getNumberOfRows();
    const size_t nCols         = dst->getNumberOfColumns();
    const size_t nDataElements = nRows * nCols;

    auto & context = services::internal::getDefaultContext();
    services::Status status;
    DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(dstBlock.getBuffer()), algorithmFPType, nDataElements);
    DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(srcBlock.getBuffer()), algorithmFPType, nDataElements);
    context.copy(dstBlock.getBuffer(), 0, srcBlock.getBuffer(), 0, nDataElements, status);
    DAAL_CHECK_STATUS_VAR(status);

    DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(src)->releaseBlockOfRows(srcBlock));
    DAAL_CHECK_STATUS_VAR(dst->releaseBlockOfRows(dstBlock));

    return status;
}

template <typename algorithmFPType>
Status PCACorrelationKernelOnlineUCAPI<algorithmFPType>::finalize(PartialResult<correlationDense> * partialResult,
                                                                  const OnlineParameter<algorithmFPType, correlationDense> * parameter,
                                                                  data_management::NumericTable & eigenvectors,
                                                                  data_management::NumericTable & eigenvalues)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(finalize);
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.covariance.finalize);
        DAAL_CHECK_STATUS_VAR(parameter->covariance->finalizeCompute());
    }

    data_management::NumericTablePtr correlation = parameter->covariance->getResult()->get(covariance::covariance);
    DAAL_ASSERT(correlation);
    DAAL_CHECK_STATUS_VAR(_host_impl->computeCorrelationEigenvalues(*correlation, eigenvectors, eigenvalues));

    return services::Status();
}

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
