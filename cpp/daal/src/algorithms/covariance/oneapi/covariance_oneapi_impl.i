/* file: covariance_oneapi_impl.i */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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

#include "services/internal/buffer.h"
#include "data_management/data/numeric_table.h"
#include "services/env_detect.h"
#include "services/error_indexes.h"
#include "src/sycl/blas_gpu.h"
#include "src/sycl/reducer.h"
#include "src/algorithms/covariance/oneapi/cl_kernels/covariance_kernels.cl"
#include "src/externals/service_profiler.h"
#include "src/services/service_data_utils.h"

using namespace daal::services::internal;
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
static services::Status buildProgram(ClKernelFactoryIface & factory)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);

    services::Status status;

    auto fptype_name   = getKeyFPType<algorithmFPType>();
    auto build_options = fptype_name;
    build_options.add("-cl-std=CL1.2");

    services::String cachekey("__daal_algorithms_covariance_dense_batch_finalizeCovariance_");
    cachekey.add(fptype_name);
    factory.build(ExecutionTargetIds::device, cachekey.c_str(), covariance_kernels, build_options.c_str(), status);
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

static uint32_t getGlobalRangeSize(uint32_t localRangeSize, uint32_t N)
{
    DAAL_ASSERT(localRangeSize != 0);
    uint32_t factor = N / localRangeSize;

    if (factor * localRangeSize != N)
    {
        factor++;
    }
    return factor * localRangeSize;
}

static KernelNDRange getKernelNDRange(uint32_t localRangeSize, uint32_t globalRangeSize, services::Status & status)
{
    KernelNDRange ndrange(2);
    KernelRange local_range(localRangeSize, localRangeSize);
    KernelRange global_range(globalRangeSize, globalRangeSize);

    ndrange.global(global_range, status);
    DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, ndrange);

    ndrange.local(local_range, status);
    DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, ndrange);

    return ndrange;
}

template <typename algorithmFPType, Method method>
services::Status prepareSums(NumericTable * dataTable, const services::internal::Buffer<algorithmFPType> & sumsBuffer)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.prepareSums);

    const uint32_t nFeatures = dataTable->getNumberOfColumns();
    auto & context           = services::internal::getDefaultContext();
    services::Status status;

    if (method == sumDense || method == sumCSR)
    {
        NumericTable * dataSumsTable = dataTable->basicStatistics.get(NumericTable::sum).get();

        BlockDescriptor<algorithmFPType> userSums;
        DAAL_CHECK_STATUS_VAR(dataSumsTable->getBlockOfRows(0, dataSumsTable->getNumberOfRows(), readOnly, userSums));

        DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(sumsBuffer), algorithmFPType, nFeatures);
        DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(userSums.getBuffer()), algorithmFPType, nFeatures);
        context.copy(sumsBuffer, 0, userSums.getBuffer(), 0, nFeatures, status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_CHECK_STATUS_VAR(dataSumsTable->releaseBlockOfRows(userSums));
    }
    else
    {
        const algorithmFPType zero = 0.0;
        DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(sumsBuffer), algorithmFPType, nFeatures);
        context.fill(sumsBuffer, zero, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status prepareCrossProduct(const services::internal::Buffer<algorithmFPType> & crossProductBuffer, uint32_t nFeatures)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.prepareCrossProduct);

    const algorithmFPType zero = 0.0;

    auto & context = services::internal::getDefaultContext();
    services::Status status;

    DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(crossProductBuffer), algorithmFPType, nFeatures * nFeatures);
    context.fill(crossProductBuffer, zero, status);
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

template <typename algorithmFPType, Method method>
services::Status updateDenseCrossProductAndSums(bool isNormalized, uint32_t nFeatures, uint32_t nVectors,
                                                const services::internal::Buffer<algorithmFPType> & dataBlock,
                                                const services::internal::Buffer<algorithmFPType> & crossProduct,
                                                const services::internal::Buffer<algorithmFPType> & sums)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.updateCrossProductAndSums);
    auto & context              = services::internal::getDefaultContext();
    bool nonNormalizedFastDense = ((!isNormalized) && (method == defaultDense || method == sumDense));

    if (isNormalized || nonNormalizedFastDense)
    {
        services::Status status;

        DAAL_ASSERT(nVectors != 0);
        algorithmFPType nVectorsInv = algorithmFPType(1.0) / algorithmFPType(nVectors);
        algorithmFPType beta        = (isNormalized == true) ? algorithmFPType(0.0) : -nVectorsInv;

        if (!isNormalized)
        {
            DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(dataBlock), algorithmFPType, nFeatures * nVectors);
            auto sumResult = math::SumReducer::sum(math::Layout::ColMajor, dataBlock, nFeatures, nVectors, status);
            DAAL_CHECK_STATUS_VAR(status);

            DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(sums), algorithmFPType, sums.size());
            DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(sumResult.sum), algorithmFPType, sums.size());
            context.copy(sums, 0, sumResult.sum, 0, sums.size(), status);
            DAAL_CHECK_STATUS_VAR(status);

            {
                DAAL_ITTNOTIFY_SCOPED_TASK(compute.updateCrossProductAndSums.gemmSums);

                DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(sums), algorithmFPType, nFeatures);
                DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(crossProduct), algorithmFPType, nFeatures * nFeatures);
                status |= BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::Trans, math::Transpose::NoTrans, nFeatures,
                                                          nFeatures, 1, algorithmFPType(1.0), sums, nFeatures, 0, sums, nFeatures, 0,
                                                          algorithmFPType(0.0), crossProduct, nFeatures, 0);
            }
            DAAL_CHECK_STATUS_VAR(status);
        }

        {
            DAAL_ITTNOTIFY_SCOPED_TASK(compute.updateCrossProductAndSums.gemmData);

            DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(dataBlock), algorithmFPType, nFeatures * nVectors);
            DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(crossProduct), algorithmFPType, nFeatures * nFeatures);
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
services::Status mergeCrossProduct(uint32_t nFeatures, const services::internal::Buffer<algorithmFPType> & partialCrossProduct,
                                   const services::internal::Buffer<algorithmFPType> & partialSums, algorithmFPType partialNObservations,
                                   const services::internal::Buffer<algorithmFPType> & crossProduct,
                                   const services::internal::Buffer<algorithmFPType> & sums, algorithmFPType nObservations)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.mergeCrossProduct);
    services::Status status;

    auto & context = services::internal::getDefaultContext();
    auto & factory = context.getClKernelFactory();
    status |= buildProgram<algorithmFPType>(factory);
    DAAL_CHECK_STATUS_VAR(status);

    auto kernel = factory.getKernel("mergeCrossProduct", status);
    DAAL_CHECK_STATUS_VAR(status);

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(crossProduct), algorithmFPType, nFeatures * nFeatures);
        DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(partialCrossProduct), algorithmFPType, nFeatures * nFeatures);
        DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(sums), algorithmFPType, nFeatures);
        DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(partialSums), algorithmFPType, nFeatures);

        DAAL_ASSERT(partialNObservations != 0);
        DAAL_ASSERT(nObservations != 0);
        DAAL_ASSERT(nObservations + partialNObservations != 0);

        const algorithmFPType invPartialNObs = (algorithmFPType)(1.0) / partialNObservations;
        const algorithmFPType invNObs        = (algorithmFPType)(1.0) / nObservations;
        const algorithmFPType invNewNObs     = (algorithmFPType)(1.0) / (nObservations + partialNObservations);

        KernelArguments args(8, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, nFeatures);
        args.set(1, partialCrossProduct, AccessModeIds::read);
        args.set(2, partialSums, AccessModeIds::read);
        args.set(3, crossProduct, AccessModeIds::readwrite);
        args.set(4, sums, AccessModeIds::read);
        args.set(5, invPartialNObs);
        args.set(6, invNObs);
        args.set(7, invNewNObs);

        const uint32_t localRangeSize = 16;
        KernelNDRange ndrange         = getKernelNDRange(localRangeSize, getGlobalRangeSize(localRangeSize, nFeatures), status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(ndrange, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType, Method method>
services::Status mergeSums(uint32_t nFeatures, const services::internal::Buffer<algorithmFPType> & partialSums,
                           const services::internal::Buffer<algorithmFPType> & sums)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.mergeSums);
    services::Status status;

    DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(partialSums), algorithmFPType, nFeatures);
    DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(sums), algorithmFPType, nFeatures);
    status |= BlasGpu<algorithmFPType>::xaxpy(nFeatures, 1, partialSums, 1, sums, 1);

    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

template <typename algorithmFPType>
services::Status prepareMeansAndCrossProductDiag(uint32_t nFeatures, algorithmFPType nObservations,
                                                 const services::internal::Buffer<algorithmFPType> & crossProduct,
                                                 const services::internal::Buffer<algorithmFPType> & diagCrossProduct,
                                                 const services::internal::Buffer<algorithmFPType> & sums,
                                                 const services::internal::Buffer<algorithmFPType> & means)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.prepareMeansAndCrossProductDiag);

    services::Status status;

    auto & context = services::internal::getDefaultContext();
    auto & factory = context.getClKernelFactory();
    status |= buildProgram<algorithmFPType>(factory);
    DAAL_CHECK_STATUS_VAR(status);

    auto kernel = factory.getKernel("prepareMeansAndCrossProductDiag", status);
    DAAL_CHECK_STATUS_VAR(status);
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(crossProduct), algorithmFPType, nFeatures * nFeatures);
        DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(diagCrossProduct), algorithmFPType, nFeatures);
        DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(sums), algorithmFPType, nFeatures);
        DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(means), algorithmFPType, nFeatures);

        KernelArguments args(6, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, nFeatures);
        args.set(1, nObservations);
        args.set(2, crossProduct, AccessModeIds::read);
        args.set(3, diagCrossProduct, AccessModeIds::write);
        args.set(4, sums, AccessModeIds::readwrite);
        args.set(5, means, AccessModeIds::readwrite);

        KernelRange range(nFeatures);
        context.run(range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status finalize(uint32_t nFeatures, algorithmFPType nObservations, const services::internal::Buffer<algorithmFPType> & crossProduct,
                          const services::internal::Buffer<algorithmFPType> & cov,
                          const services::internal::Buffer<algorithmFPType> & diagCrossProduct, const Parameter * parameter)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.finalize);

    services::Status status;

    auto & context = services::internal::getDefaultContext();
    auto & factory = context.getClKernelFactory();
    status |= buildProgram<algorithmFPType>(factory);
    DAAL_CHECK_STATUS_VAR(status);

    auto kernel = factory.getKernel("finalize", status);
    DAAL_CHECK_STATUS_VAR(status);

    {
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, nFeatures, nFeatures);
        DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(crossProduct), algorithmFPType, nFeatures * nFeatures);
        DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(cov), algorithmFPType, nFeatures * nFeatures);
        DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(diagCrossProduct), algorithmFPType, nFeatures);

        uint32_t isOutputCorrelationMatrix = static_cast<uint32_t>(parameter->outputMatrixType == covariance::correlationMatrix);

        KernelArguments args(6, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, nFeatures);
        args.set(1, nObservations);
        args.set(2, crossProduct, AccessModeIds::read);
        args.set(3, cov, AccessModeIds::readwrite);
        args.set(4, diagCrossProduct, AccessModeIds::read);
        args.set(5, isOutputCorrelationMatrix);

        const uint32_t localRangeSize = 4;
        KernelNDRange ndrange         = getKernelNDRange(localRangeSize, getGlobalRangeSize(localRangeSize, nFeatures), status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(ndrange, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType, Method method>
services::Status calculateCrossProductAndSums(NumericTable * dataTable, const services::internal::Buffer<algorithmFPType> & crossProduct,
                                              const services::internal::Buffer<algorithmFPType> & sums)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.calculateCrossProductAndSums);
    services::Status status;

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

    const bool isNormalized = dataTable->isNormalized(NumericTableIface::standardScoreNormalized);

    BlockDescriptor<algorithmFPType> dataBlock;
    status |= dataTable->getBlockOfRows(0, nVectors, readOnly, dataBlock);
    DAAL_CHECK_STATUS_VAR(status);

    status |= prepareSums<algorithmFPType, method>(dataTable, sums);
    DAAL_CHECK_STATUS_VAR(status);

    status |= prepareCrossProduct<algorithmFPType>(crossProduct, nFeatures);
    DAAL_CHECK_STATUS_VAR(status);

    status |= updateDenseCrossProductAndSums<algorithmFPType, method>(isNormalized, nFeatures, nVectors, dataBlock.getBuffer(), crossProduct, sums);
    DAAL_CHECK_STATUS_VAR(status);

    status |= dataTable->releaseBlockOfRows(dataBlock);
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

template <typename algorithmFPType, Method method>
services::Status finalizeCovariance(uint32_t nFeatures, algorithmFPType nObservations,
                                    const services::internal::Buffer<algorithmFPType> & crossProduct,
                                    const services::internal::Buffer<algorithmFPType> & sums, const services::internal::Buffer<algorithmFPType> & cov,
                                    const services::internal::Buffer<algorithmFPType> & mean, const Parameter * parameter)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.finalizeCovariance);
    services::Status status;

    auto & context = services::internal::getDefaultContext();

    auto diagCrossProduct = context.allocate(TypeIds::id<algorithmFPType>(), nFeatures, status);
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
