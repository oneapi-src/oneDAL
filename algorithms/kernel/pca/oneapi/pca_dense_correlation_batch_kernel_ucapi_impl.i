/* file: pca_dense_correlation_batch_kernel_ucapi_impl.i */
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

#ifndef __PCA_DENSE_CORRELATION_BATCH_KERNEL_UCAPI_IMPL__
#define __PCA_DENSE_CORRELATION_BATCH_KERNEL_UCAPI_IMPL__

#include "service_ittnotify.h"
DAAL_ITTNOTIFY_DOMAIN(pca.dense.correlation.batch.oneapi);

#include "env_detect.h"
#include "cl_kernels/pca_cl_kernels.cl"
#include "numeric_table_sycl_homogen.h"
#include "oneapi/blas_gpu.h"
#include "oneapi/sum_reducer.h"
#include "covariance/oneapi/covariance_oneapi_impl.i"

using namespace daal::services;
using namespace daal::internal;
using namespace daal::oneapi::internal;
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
PCACorrelationKernelUCAPI<algorithmFPType>::PCACorrelationKernelUCAPI(const PCACorrelationBaseIfacePtr & host_impl)
{
    _host_impl = host_impl;
}

template <typename algorithmFPType>
Status PCACorrelationKernelUCAPI<algorithmFPType>::compute(bool isCorrelation, bool isDeterministic, NumericTable & dataTable,
                                                           covariance::BatchImpl * covarianceAlg, DAAL_UINT64 resultsToCompute,
                                                           NumericTable & eigenvectors, NumericTable & eigenvalues, NumericTable & means,
                                                           NumericTable & variances)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute);
    Status st;

    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();

    auto fptype_name   = oneapi::internal::getKeyFPType<algorithmFPType>();
    auto build_options = fptype_name;
    build_options.add("-cl-std=CL1.2");

    services::String cachekey("__daal_algorithms_pca_cor_dense_batch_");
    cachekey.add(fptype_name);

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);
        kernel_factory.build(ExecutionTargetIds::device, cachekey.c_str(), pca_cl_kernels, build_options.c_str());
    }

    auto calculateVariancesKernel = kernel_factory.getKernel("calculateVariances");

    const uint32_t N = dataTable.getNumberOfRows();
    const uint32_t p = dataTable.getNumberOfColumns();

    if (isCorrelation)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.correlation);

        if (resultsToCompute & mean)
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(compute.correlation.fillTable(means));

            BlockDescriptor<algorithmFPType> meansBlock;
            means.getBlockOfRows(0, 1, readWrite, meansBlock);
            context.fill(meansBlock.getBuffer(), (algorithmFPType)0, &st);
            means.releaseBlockOfRows(meansBlock);

            DAAL_CHECK_STATUS_VAR(st);
        }

        if (resultsToCompute & variance)
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(compute.correlation.fillTable(variances));

            BlockDescriptor<algorithmFPType> varBlock;
            variances.getBlockOfRows(0, 1, readWrite, varBlock);
            context.fill(varBlock.getBuffer(), (algorithmFPType)1, &st);
            variances.releaseBlockOfRows(varBlock);

            DAAL_CHECK_STATUS_VAR(st);
        }

        {
            DAAL_ITTNOTIFY_SCOPED_TASK(compute.correlation.eigenvalues);
            DAAL_CHECK_STATUS(st, _host_impl->computeCorrelationEigenvalues(dataTable, eigenvectors, eigenvalues));
        }
    }
    else
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.full);
        DAAL_CHECK(covarianceAlg, services::ErrorNullPtr);
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(compute.full.covariance);
            DAAL_CHECK_STATUS(st, covarianceAlg->computeNoThrow());
        }

        auto pCovarianceTable          = covarianceAlg->getResult()->get(covariance::covariance);
        NumericTable & covarianceTable = *pCovarianceTable;

        // copying variances. Means are computed inplace
        // with help of setResult in BatchContainer

        if (resultsToCompute & variance)
        {
            BlockDescriptor<algorithmFPType> varBlock;
            variances.getBlockOfRows(0, 1, readWrite, varBlock);

            DAAL_CHECK_STATUS(st, calculateVariances(context, calculateVariancesKernel, covarianceTable, varBlock.getBuffer()));

            DAAL_CHECK_STATUS(st, correlationFromCovarianceTable(N, covarianceTable, varBlock.getBuffer()));

            variances.releaseBlockOfRows(varBlock);
        }
        else
        {
            auto variancesBuffer = context.allocate(TypeIds::id<algorithmFPType>(), p, &st);
            DAAL_CHECK_STATUS_VAR(st);

            DAAL_CHECK_STATUS(
                st, calculateVariances(context, calculateVariancesKernel, covarianceTable, variancesBuffer.template get<algorithmFPType>()));

            DAAL_CHECK_STATUS(st, correlationFromCovarianceTable(N, covarianceTable, variancesBuffer.template get<algorithmFPType>()));
        }

        {
            DAAL_ITTNOTIFY_SCOPED_TASK(compute.full.computeEigenvalues);
            DAAL_CHECK_STATUS(st, _host_impl->computeCorrelationEigenvalues(covarianceTable, eigenvectors, eigenvalues));
        }
    }

    if (isDeterministic)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.signFlipEigenvectors);
        DAAL_CHECK_STATUS(st, _host_impl->signFlipEigenvectors(eigenvectors));
    }

    return st;
}

template <typename algorithmFPType>
services::Status PCACorrelationKernelUCAPI<algorithmFPType>::correlationFromCovarianceTable(uint32_t nObservations, NumericTable & covariance,
                                                                                            const services::Buffer<algorithmFPType> & variances)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.correlationFromCovarianceTable);
    services::Status status;

    uint32_t nFeatures = covariance.getNumberOfRows();

    BlockDescriptor<algorithmFPType> covBlock;

    covariance.getBlockOfRows(0, nFeatures, writeOnly, covBlock);

    covariance::Parameter parameter;
    parameter.outputMatrixType = covariance::correlationMatrix;

    status |= covariance::oneapi::internal::finalize<algorithmFPType>(nFeatures, nObservations, covBlock.getBuffer(), covBlock.getBuffer(), variances,
                                                                      &parameter);

    covariance.releaseBlockOfRows(covBlock);

    return status;
}

template <typename algorithmFPType>
services::Status PCACorrelationKernelUCAPI<algorithmFPType>::calculateVariances(ExecutionContextIface & context,
                                                                                const KernelPtr & calculateVariancesKernel, NumericTable & covariance,
                                                                                const services::Buffer<algorithmFPType> & variances)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.calculateVariances);
    services::Status status;

    uint32_t nFeatures = covariance.getNumberOfRows();

    BlockDescriptor<algorithmFPType> covBlock;

    covariance.getBlockOfRows(0, nFeatures, readOnly, covBlock);

    KernelArguments args(2);
    args.set(0, covBlock.getBuffer(), AccessModeIds::read);
    args.set(1, variances, AccessModeIds::write);

    KernelRange range(nFeatures);
    context.run(range, calculateVariancesKernel, args, &status);

    covariance.releaseBlockOfRows(covBlock);

    return status;
}

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
