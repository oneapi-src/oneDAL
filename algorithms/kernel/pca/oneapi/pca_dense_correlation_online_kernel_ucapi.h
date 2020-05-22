/* file: pca_dense_correlation_online_kernel_ucapi.h */
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
//  Implementation of PCA Online Kernel for GPU.
//--
*/

#ifndef __PCA_DENSE_CORRELATION_ONLINE_KERNEL_UCAPI_H__
#define __PCA_DENSE_CORRELATION_ONLINE_KERNEL_UCAPI_H__

#include "algorithms/kernel/pca/pca_dense_correlation_base_iface.h"
#include "oneapi/internal/types.h"
#include "oneapi/internal/execution_context.h"
#include "algorithms/pca/pca_types.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{
template <typename algorithmFPType>
class PCACorrelationKernelOnlineUCAPI : public Kernel
{
public:
    using PCACorrelationBaseIfacePtr = services::SharedPtr<PCACorrelationBaseIface<algorithmFPType> >;

public:
    PCACorrelationKernelOnlineUCAPI(const PCACorrelationBaseIfacePtr & host_impl);

    services::Status compute(const data_management::NumericTablePtr & pData, PartialResult<correlationDense> * partialResult,
                             const OnlineParameter<algorithmFPType, correlationDense> * parameter);
    services::Status finalize(PartialResult<correlationDense> * partialResult, const OnlineParameter<algorithmFPType, correlationDense> * parameter,
                              data_management::NumericTable & eigenvectors, data_management::NumericTable & eigenvalues);

private:
    PCACorrelationBaseIfacePtr _host_impl;

private:
    services::Status copyIfNeeded(const data_management::NumericTable * src, data_management::NumericTable * dst);
    services::Status copyCovarianceResultToPartialResult(const covariance::PartialResult * covariancePres,
                                                         PartialResult<correlationDense> * partialResult);
};

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
