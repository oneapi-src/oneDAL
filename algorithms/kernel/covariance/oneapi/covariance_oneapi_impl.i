/* file: covariance_oneapi_impl.i */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
//  Covariance matrix computation algorithm implementation
//--
*/

#ifndef __COVARIANCE_ONEAPI_IMPL_I__
#define __COVARIANCE_ONEAPI_IMPL_I__

#include "services/buffer.h"
#include "numeric_table.h"
#include "env_detect.h"
#include "error_indexes.h"
#include "oneapi/blas_gpu.h"
#include "oneapi/sum_reducer.h"
#include "cl_kernels/covariance_kernels.cl"
#include "service_ittnotify.h"
#include "service_data_utils.h"

using namespace daal::services::internal;
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
template <typename algorithmFPType>
static void __buildProgram(ClKernelFactoryIface & factory)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);

    auto fptype_name   = getKeyFPType<algorithmFPType>();
    auto build_options = fptype_name;
    build_options.add("-cl-std=CL1.2");

    services::String cachekey("__daal_algorithms_covariance_dense_batch_finalizeCovariance_");
    cachekey.add(fptype_name);
    factory.build(ExecutionTargetIds::device, cachekey.c_str(), covariance_kernels, build_options.c_str());
}

static size_t getGlobalRangeSize(size_t localRangeSize, size_t N)
{
    size_t factor = N / localRangeSize;

    if (factor * localRangeSize != N)
    {
        factor++;
    }
    return factor * localRangeSize;
}

static KernelNDRange getKernelNDRange(size_t localRangeSize, size_t globalRangeSize, services::Status & status)
{
    KernelNDRange ndrange(2);
    KernelRange local_range(localRangeSize, localRangeSize);
    KernelRange global_range(globalRangeSize, globalRangeSize);

    ndrange.global(global_range, &status);
    ndrange.local(local_range, &status);

    return ndrange;
}

template <typename algorithmFPType, Method method>
services::Status prepareSums(NumericTable * dataTable, const services::Buffer<algorithmFPType> & sumsBuffer)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.prepareSums);

    const size_t nFeatures = dataTable->getNumberOfColumns();
    auto & context         = services::Environment::getInstance()->getDefaultExecutionContext();
    services::Status status;

    if (method == sumDense || method == sumCSR)
    {
        NumericTable * dataSumsTable = dataTable->basicStatistics.get(NumericTable::sum).get();

        BlockDescriptor<algorithmFPType> userSums;
        dataSumsTable->getBlockOfRows(0, dataSumsTable->getNumberOfRows(), readOnly, userSums);

        context.copy(sumsBuffer, 0, userSums.getBuffer(), 0, nFeatures, &status);

        dataSumsTable->releaseBlockOfRows(userSums);
    }
    else
    {
        const algorithmFPType zero = 0.0;
        context.fill(sumsBuffer, zero, &status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status prepareCrossProduct(size_t nFeatures, const services::Buffer<algorithmFPType> & crossProductBuffer)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.prepareCrossProduct);

    const algorithmFPType zero = 0.0;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    services::Status status;

    context.fill(crossProductBuffer, zero, &status);
    return status;
}

template <typename algorithmFPType, Method method>
services::Status updateDenseCrossProductAndSums(bool isNormalized, size_t nFeatures, size_t nVectors,
                                                const services::Buffer<algorithmFPType> & dataBlock,
                                                const services::Buffer<algorithmFPType> & crossProduct,
                                                const services::Buffer<algorithmFPType> & sums)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.updateCrossProductAndSums);
    auto & context              = services::Environment::getInstance()->getDefaultExecutionContext();
    bool nonNormalizedFastDense = ((!isNormalized) && (method == defaultDense || method == sumDense));

    if (isNormalized || nonNormalizedFastDense)
    {
        services::Status status;

        algorithmFPType nVectorsInv = algorithmFPType(1.0) / algorithmFPType(nVectors);
        algorithmFPType beta        = (isNormalized == true) ? algorithmFPType(0.0) : -nVectorsInv;

        if (!isNormalized)
        {
            auto sumResult = math::SumReducer::sum(math::Layout::ColMajor, dataBlock, nFeatures, nVectors, &status);
            DAAL_CHECK_STATUS_VAR(status);

            context.copy(sums, 0, sumResult.sum, 0, sums.size(), &status);
            DAAL_CHECK_STATUS_VAR(status);

            {
                DAAL_ITTNOTIFY_SCOPED_TASK(compute.updateCrossProductAndSums.gemmSums);

                status |= BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::Trans, math::Transpose::NoTrans, nFeatures,
                                                          nFeatures, 1, algorithmFPType(1.0), sums, nFeatures, 0, sums, nFeatures, 0,
                                                          algorithmFPType(0.0), crossProduct, nFeatures, 0);
            }
            DAAL_CHECK_STATUS_VAR(status);
        }

        {
            DAAL_ITTNOTIFY_SCOPED_TASK(compute.updateCrossProductAndSums.gemmData);

            status |= BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::Trans, math::Transpose::NoTrans, nFeatures, nFeatures,
                                                      nVectors, algorithmFPType(1.0), dataBlock, nFeatures, 0, dataBlock, nFeatures, 0, beta,
                                                      crossProduct, nFeatures, 0);
        }
        DAAL_CHECK_STATUS_VAR(status);
    }
    else
    {
        return services::ErrorMethodNotImplemented;
    }

    return services::Status();
}

template <typename algorithmFPType>
services::Status mergeCrossProduct(size_t nFeatures, const services::Buffer<algorithmFPType> & partialCrossProduct,
                                   const services::Buffer<algorithmFPType> & partialSums, algorithmFPType partialNObservations,
                                   const services::Buffer<algorithmFPType> & crossProduct, const services::Buffer<algorithmFPType> & sums,
                                   algorithmFPType nObservations)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.mergeCrossProduct);

    if (nFeatures > services::internal::MaxVal<uint32_t>::get())
    {
        return services::Status(daal::services::ErrorCovarianceInternal);
    }

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & factory = context.getClKernelFactory();
    __buildProgram<algorithmFPType>(factory);

    auto kernel = factory.getKernel("mergeCrossProduct");

    {
        KernelArguments args(7);
        args.set(0, static_cast<uint32_t>(nFeatures));
        args.set(1, partialCrossProduct, AccessModeIds::read);
        args.set(2, partialSums, AccessModeIds::read);
        args.set(3, partialNObservations);
        args.set(4, crossProduct, AccessModeIds::readwrite);
        args.set(5, sums, AccessModeIds::read);
        args.set(6, nObservations);

        size_t localRangeSize = 16;
        KernelNDRange ndrange = getKernelNDRange(localRangeSize, getGlobalRangeSize(localRangeSize, nFeatures), status);

        context.run(ndrange, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType, Method method>
services::Status mergeSums(size_t nFeatures, const services::Buffer<algorithmFPType> & partialSums, const services::Buffer<algorithmFPType> & sums)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.mergeSums);
    services::Status status;

    status |= BlasGpu<algorithmFPType>::xaxpy(nFeatures, 1, partialSums, 1, sums, 1);

    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

template <typename algorithmFPType>
services::Status prepareMeansAndCrossProductDiag(size_t nFeatures, algorithmFPType nObservations,
                                                 const services::Buffer<algorithmFPType> & crossProduct,
                                                 const services::Buffer<algorithmFPType> & diagCrossProduct,
                                                 const services::Buffer<algorithmFPType> & sums, const services::Buffer<algorithmFPType> & mean)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.prepareMeansAndCrossProductDiag);

    if (nFeatures > services::internal::MaxVal<uint32_t>::get())
    {
        return services::Status(daal::services::ErrorCovarianceInternal);
    }

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & factory = context.getClKernelFactory();
    __buildProgram<algorithmFPType>(factory);

    auto kernel = factory.getKernel("prepareMeansAndCrossProductDiag");
    {
        KernelArguments args(6);
        args.set(0, static_cast<uint32_t>(nFeatures));
        args.set(1, nObservations);
        args.set(2, crossProduct, AccessModeIds::read);
        args.set(3, diagCrossProduct, AccessModeIds::write);
        args.set(4, sums, AccessModeIds::readwrite);
        args.set(5, mean, AccessModeIds::readwrite);

        KernelRange range(nFeatures);
        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status finalize(size_t nFeatures, algorithmFPType nObservations, const services::Buffer<algorithmFPType> & crossProduct,
                          const services::Buffer<algorithmFPType> & cov, const services::Buffer<algorithmFPType> & diagCrossProduct,
                          const Parameter * parameter)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.finalize);

    if (nFeatures > services::internal::MaxVal<uint32_t>::get())
    {
        return services::Status(daal::services::ErrorCovarianceInternal);
    }

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & factory = context.getClKernelFactory();
    __buildProgram<algorithmFPType>(factory);

    auto kernel = factory.getKernel("finalize");

    {
        uint32_t isOutputCorrelationMatrix = parameter->outputMatrixType == covariance::correlationMatrix;

        KernelArguments args(6);
        args.set(0, static_cast<uint32_t>(nFeatures));
        args.set(1, nObservations);
        args.set(2, crossProduct, AccessModeIds::read);
        args.set(3, cov, AccessModeIds::readwrite);
        args.set(4, diagCrossProduct, AccessModeIds::read);
        args.set(5, isOutputCorrelationMatrix);

        size_t localRangeSize = 4;
        KernelNDRange ndrange = getKernelNDRange(localRangeSize, getGlobalRangeSize(localRangeSize, nFeatures), status);

        context.run(ndrange, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType, Method method>
services::Status calculateCrossProductAndSums(NumericTable * dataTable, const services::Buffer<algorithmFPType> & crossProduct,
                                              const services::Buffer<algorithmFPType> & sums)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.calculateCrossProductAndSums);
    services::Status status;

    const size_t nFeatures  = dataTable->getNumberOfColumns();
    const size_t nVectors   = dataTable->getNumberOfRows();
    const bool isNormalized = dataTable->isNormalized(NumericTableIface::standardScoreNormalized);

    BlockDescriptor<algorithmFPType> dataBlock;
    status |= dataTable->getBlockOfRows(0, nVectors, readWrite, dataBlock);
    DAAL_CHECK_STATUS_VAR(status);

    status |= prepareSums<algorithmFPType, method>(dataTable, sums);
    DAAL_CHECK_STATUS_VAR(status);

    status |= prepareCrossProduct<algorithmFPType>(nFeatures, crossProduct);
    DAAL_CHECK_STATUS_VAR(status);

    status |= updateDenseCrossProductAndSums<algorithmFPType, method>(isNormalized, nFeatures, nVectors, dataBlock.getBuffer(), crossProduct, sums);
    DAAL_CHECK_STATUS_VAR(status);

    status |= dataTable->releaseBlockOfRows(dataBlock);
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

template <typename algorithmFPType, Method method>
services::Status finalizeCovariance(size_t nFeatures, algorithmFPType nObservations, const services::Buffer<algorithmFPType> & crossProduct,
                                    const services::Buffer<algorithmFPType> & sums, const services::Buffer<algorithmFPType> & cov,
                                    const services::Buffer<algorithmFPType> & mean, const Parameter * parameter)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.finalizeCovariance);
    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto diagCrossProduct = context.allocate(TypeIds::id<algorithmFPType>(), nFeatures, &status);
    DAAL_CHECK_STATUS_VAR(status);

    status |= prepareMeansAndCrossProductDiag<algorithmFPType>(nFeatures, nObservations, crossProduct,
                                                               diagCrossProduct.template get<algorithmFPType>(), sums, mean);
    DAAL_CHECK_STATUS_VAR(status);

    status |= finalize<algorithmFPType>(nFeatures, nObservations, crossProduct, cov, diagCrossProduct.template get<algorithmFPType>(), parameter);
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

} // namespace internal
} // namespace oneapi
} // namespace covariance
} // namespace algorithms
} // namespace daal

#endif
