/* file: pca_dense_correlation_batch_kernel_ucapi_impl.i */
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
//  Implementation of PCA Batch Kernel for GPU.
//--
*/

#ifndef __PCA_DENSE_CORRELATION_BATCH_KERNEL_UCAPI_IMPL__
#define __PCA_DENSE_CORRELATION_BATCH_KERNEL_UCAPI_IMPL__

#include "src/externals/service_profiler.h"

#include "services/env_detect.h"
#include "include/services/internal/sycl/types.h"
#include "src/algorithms/pca/oneapi/cl_kernels/pca_cl_kernels.cl"
#include "data_management/data/internal/numeric_table_sycl_homogen.h"
#include "src/sycl/blas_gpu.h"
#include "src/sycl/reducer.h"
#include "src/algorithms/covariance/oneapi/covariance_oneapi_impl.i"

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
PCACorrelationKernelBatchUCAPI<algorithmFPType>::PCACorrelationKernelBatchUCAPI(const PCACorrelationBaseIfacePtr & host_impl)
{
    _host_impl = host_impl;
}

template <typename algorithmFPType>
Status PCACorrelationKernelBatchUCAPI<algorithmFPType>::compute(bool isCorrelation, bool isDeterministic, NumericTable & dataTable,
                                                                covariance::BatchImpl * covarianceAlg, DAAL_UINT64 resultsToCompute,
                                                                NumericTable & eigenvectors, NumericTable & eigenvalues, NumericTable & means,
                                                                NumericTable & variances)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute);
    Status st;

    auto & context        = Environment::getInstance().getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();

    auto fptype_name   = services::internal::sycl::getKeyFPType<algorithmFPType>();
    auto build_options = fptype_name;
    build_options.add("-cl-std=CL1.2");

    services::String cachekey("__daal_algorithms_pca_cor_dense_batch_");
    cachekey.add(fptype_name);

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);
        kernel_factory.build(ExecutionTargetIds::device, cachekey.c_str(), pca_cl_kernels, build_options.c_str(), st);
        DAAL_CHECK_STATUS_VAR(st);
    }

    auto calculateVariancesKernel = kernel_factory.getKernel("calculateVariances", st);
    DAAL_CHECK_STATUS_VAR(st);

    if (dataTable.getNumberOfColumns() > static_cast<size_t>(services::internal::MaxVal<uint32_t>::get()))
    {
        return services::Status(daal::services::ErrorCovarianceInternal);
    }
    if (dataTable.getNumberOfRows() > static_cast<size_t>(services::internal::MaxVal<uint32_t>::get()))
    {
        return services::Status(daal::services::ErrorCovarianceInternal);
    }

    const uint32_t N = static_cast<uint32_t>(dataTable.getNumberOfRows());
    const uint32_t p = static_cast<uint32_t>(dataTable.getNumberOfColumns());

    if (isCorrelation)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.correlation);

        if (resultsToCompute & mean)
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(compute.correlation.fillTable(means));

            BlockDescriptor<algorithmFPType> meansBlock;
            DAAL_CHECK_STATUS_VAR(means.getBlockOfRows(0, 1, readWrite, meansBlock));
            DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(meansBlock.getBuffer()), algorithmFPType, p);
            context.fill(meansBlock.getBuffer(), (algorithmFPType)0, st);
            DAAL_CHECK_STATUS_VAR(st);
            DAAL_CHECK_STATUS_VAR(means.releaseBlockOfRows(meansBlock));
        }

        if (resultsToCompute & variance)
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(compute.correlation.fillTable(variances));

            BlockDescriptor<algorithmFPType> varBlock;
            DAAL_CHECK_STATUS_VAR(variances.getBlockOfRows(0, 1, readWrite, varBlock));
            DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(varBlock.getBuffer()), algorithmFPType, p);
            context.fill(varBlock.getBuffer(), (algorithmFPType)1, st);
            DAAL_CHECK_STATUS_VAR(st);
            DAAL_CHECK_STATUS_VAR(variances.releaseBlockOfRows(varBlock));
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

        auto pCovarianceTable = covarianceAlg->getResult()->get(covariance::covariance);
        DAAL_ASSERT(pCovarianceTable);
        NumericTable & covarianceTable = *pCovarianceTable;

        // copying variances. Means are computed inplace
        // with help of setResult in BatchContainer

        if (resultsToCompute & mean)
        {
            auto mean_cov = covarianceAlg->getResult()->get(covariance::mean);
            DAAL_ASSERT(mean_cov);

            BlockDescriptor<algorithmFPType> meansBlock, covMeanBlock;
            DAAL_CHECK_STATUS_VAR(means.getBlockOfRows(0, 1, readWrite, meansBlock));
            DAAL_CHECK_STATUS_VAR(mean_cov->getBlockOfRows(0, 1, readOnly, covMeanBlock));

            DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(meansBlock.getBuffer()), algorithmFPType, p);
            DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(covMeanBlock.getBuffer()), algorithmFPType, p);
            context.copy(meansBlock.getBuffer(), 0, covMeanBlock.getBuffer(), 0, p, st);
            DAAL_CHECK_STATUS_VAR(st);

            DAAL_CHECK_STATUS_VAR(means.releaseBlockOfRows(meansBlock));
            DAAL_CHECK_STATUS_VAR(mean_cov->releaseBlockOfRows(covMeanBlock));
        }

        if (resultsToCompute & variance)
        {
            BlockDescriptor<algorithmFPType> varBlock;
            DAAL_CHECK_STATUS_VAR(variances.getBlockOfRows(0, 1, readWrite, varBlock));
            DAAL_CHECK_STATUS(st, calculateVariances(context, calculateVariancesKernel, covarianceTable, varBlock.getBuffer()));
            DAAL_CHECK_STATUS(st, correlationFromCovarianceTable(N, covarianceTable, varBlock.getBuffer()));
            DAAL_CHECK_STATUS_VAR(variances.releaseBlockOfRows(varBlock));
        }
        else
        {
            auto variancesBuffer = context.allocate(TypeIds::id<algorithmFPType>(), p, st);
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
services::Status PCACorrelationKernelBatchUCAPI<algorithmFPType>::correlationFromCovarianceTable(
    uint32_t nObservations, NumericTable & covariance, const services::internal::Buffer<algorithmFPType> & variances)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.correlationFromCovarianceTable);

    if (covariance.getNumberOfRows() > static_cast<size_t>(services::internal::MaxVal<uint32_t>::get()))
    {
        return services::Status(daal::services::ErrorCovarianceInternal);
    }

    const uint32_t nFeatures = static_cast<uint32_t>(covariance.getNumberOfRows());

    BlockDescriptor<algorithmFPType> covBlock;
    DAAL_CHECK_STATUS_VAR(covariance.getBlockOfRows(0, nFeatures, writeOnly, covBlock));

    covariance::Parameter parameter;
    parameter.outputMatrixType = covariance::correlationMatrix;

    DAAL_CHECK_STATUS_VAR(covariance::oneapi::internal::finalize<algorithmFPType>(nFeatures, nObservations, covBlock.getBuffer(),
                                                                                  covBlock.getBuffer(), variances, &parameter));

    DAAL_CHECK_STATUS_VAR(covariance.releaseBlockOfRows(covBlock));

    return services::Status();
}

template <typename algorithmFPType>
services::Status PCACorrelationKernelBatchUCAPI<algorithmFPType>::calculateVariances(ExecutionContextIface & context,
                                                                                     const KernelPtr & calculateVariancesKernel,
                                                                                     NumericTable & covariance,
                                                                                     const services::internal::Buffer<algorithmFPType> & variances)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.calculateVariances);
    services::Status status;

    if (covariance.getNumberOfRows() > static_cast<size_t>(services::internal::MaxVal<uint32_t>::get()))
    {
        return services::Status(daal::services::ErrorCovarianceInternal);
    }

    uint32_t nFeatures = static_cast<uint32_t>(covariance.getNumberOfRows());

    BlockDescriptor<algorithmFPType> covBlock;
    DAAL_CHECK_STATUS_VAR(covariance.getBlockOfRows(0, nFeatures, readOnly, covBlock));

    DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(covBlock.getBuffer()), algorithmFPType, nFeatures);

    KernelArguments args(2, status);
    DAAL_CHECK_STATUS_VAR(status);
    args.set(0, covBlock.getBuffer(), AccessModeIds::read);
    args.set(1, variances, AccessModeIds::write);

    KernelRange range(nFeatures);
    context.run(range, calculateVariancesKernel, args, status);
    DAAL_CHECK_STATUS_VAR(status);

    DAAL_CHECK_STATUS_VAR(covariance.releaseBlockOfRows(covBlock));

    return status;
}

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
