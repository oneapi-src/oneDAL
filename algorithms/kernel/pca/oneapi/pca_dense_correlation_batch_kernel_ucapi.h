/* file: pca_dense_correlation_batch_kernel_ucapi.h */
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
//  Implementation of PCA Batch Kernel for GPU.
//--
*/

#ifndef __PCA_DENSE_CORRELATION_BATCH_KERNEL_UCAPI_H__
#define __PCA_DENSE_CORRELATION_BATCH_KERNEL_UCAPI_H__

#include "pca_dense_correlation_base_iface.h"
#include "oneapi/internal/types.h"
#include "oneapi/internal/execution_context.h"
#include "pca_types.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{
template <typename algorithmFPType>
class PCACorrelationKernelUCAPI : public Kernel
{
public:
    using PCACorrelationBaseIfacePtr = services::SharedPtr<PCACorrelationBaseIface<algorithmFPType> >;

public:
    PCACorrelationKernelUCAPI(const PCACorrelationBaseIfacePtr & host_impl);

    services::Status compute(bool isCorrelation, bool isDeterministic, data_management::NumericTable & dataTable,
                             covariance::BatchImpl * covarianceAlg, DAAL_UINT64 resultsToCompute, data_management::NumericTable & eigenvectors,
                             data_management::NumericTable & eigenvalues, data_management::NumericTable & means,
                             data_management::NumericTable & variances);

private:
    services::Status calculateVariances(oneapi::internal::ExecutionContextIface & context,
                                        const oneapi::internal::KernelPtr & calculateVariancesKernel, data_management::NumericTable & covariance,
                                        const services::Buffer<algorithmFPType> & variances);

    services::Status correlationFromCovarianceTable(uint32_t nObservations, data_management::NumericTable & covariance,
                                                    const services::Buffer<algorithmFPType> & variances);

private:
    PCACorrelationBaseIfacePtr _host_impl;
};

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
