/* file: kernel_function_rbf_dense_default_oneapi_impl.i */
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
//  RBF kernel functions implementation
//--
*/

#ifndef __KERNEL_FUNCTION_RBF_DENSE_DEFAULT_IMPL_ONEAPI_I__
#define __KERNEL_FUNCTION_RBF_DENSE_DEFAULT_IMPL_ONEAPI_I__

#include "algorithms/kernel_function/kernel_function_types_rbf.h"
#include "service/kernel/data_management/service_numeric_table.h"
#include "externals/service_math.h"
#include "externals/service_ittnotify.h"
#include "service/kernel/oneapi/blas_gpu.h"
#include "service/kernel/oneapi/reducer.h"
#include "algorithms/kernel/kernel_function/oneapi/cl_kernels/kernel_function.cl"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace rbf
{
namespace internal
{
using namespace daal::oneapi::internal;
using namespace daal::oneapi::internal::math;

template <typename algorithmFPType>
services::Status KernelImplRBFOneAPI<defaultDense, algorithmFPType>::buildProgram(ClKernelFactoryIface & factory)
{
    services::String options = getKeyFPType<algorithmFPType>();

    services::String cachekey("__daal_algorithms_kernel_function_rbf");
    cachekey.add(options);

    services::Status status;
    factory.build(ExecutionTargetIds::device, cachekey.c_str(), clKernelKF, options.c_str(), &status);
    return status;
}

template <typename algorithmFPType>
services::Status KernelImplRBFOneAPI<defaultDense, algorithmFPType>::lazyAllocate(UniversalBuffer & x, const size_t n)
{
    services::Status status;
    ExecutionContextIface & ctx = services::Environment::getInstance()->getDefaultExecutionContext();
    const TypeIds::Id idType    = TypeIds::id<algorithmFPType>();
    if (x.empty() || x.get<algorithmFPType>().size() < n)
    {
        x = ctx.allocate(idType, n, &status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status KernelImplRBFOneAPI<defaultDense, algorithmFPType>::computeRBF(const UniversalBuffer & sqrMatLeft,
                                                                                const UniversalBuffer & sqrMatRight, const uint32_t ld,
                                                                                const algorithmFPType coeff, services::Buffer<algorithmFPType> & rbf,
                                                                                const size_t n, const size_t m)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(KernelRBF.computeRBF);

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & factory = context.getClKernelFactory();

    services::Status status = buildProgram(factory);
    DAAL_CHECK_STATUS_VAR(status);

    auto kernel = factory.getKernel("computeRBF");

    const algorithmFPType threshold = math::expThreshold<algorithmFPType>();

    KernelArguments args(6);
    args.set(0, sqrMatLeft, AccessModeIds::read);
    args.set(1, sqrMatRight, AccessModeIds::read);
    args.set(2, ld);
    args.set(3, threshold);
    args.set(4, coeff);
    args.set(5, rbf, AccessModeIds::readwrite);

    KernelRange range(n, m);

    context.run(range, kernel, args, &status);
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

template <typename algorithmFPType>
services::Status KernelImplRBFOneAPI<defaultDense, algorithmFPType>::computeInternalVectorVector(NumericTable * vecLeft, NumericTable * vecRight,
                                                                                                 NumericTable * result, const ParameterBase * par)
{
    return services::ErrorMethodNotImplemented;
}

template <typename algorithmFPType>
services::Status KernelImplRBFOneAPI<defaultDense, algorithmFPType>::computeInternalMatrixVector(NumericTable * matLeft, NumericTable * vecRight,
                                                                                                 NumericTable * result, const ParameterBase * par)
{
    return services::ErrorMethodNotImplemented;
}

template <typename algorithmFPType>
services::Status KernelImplRBFOneAPI<defaultDense, algorithmFPType>::computeInternalMatrixMatrix(NumericTable * matLeft, NumericTable * matRight,
                                                                                                 NumericTable * result, const ParameterBase * par)
{
    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    const size_t nMatLeft  = matLeft->getNumberOfRows();
    const size_t nMatRight = matRight->getNumberOfRows();

    const size_t pMatLeft  = matLeft->getNumberOfColumns();
    const size_t pMatRight = matRight->getNumberOfColumns();
    DAAL_ASSERT(pMatLeft == pMatRight);

    const Parameter * rbfPar    = static_cast<const Parameter *>(par);
    const algorithmFPType coeff = algorithmFPType(-0.5 / (rbfPar->sigma * rbfPar->sigma));

    BlockDescriptor<algorithmFPType> matLeftBlock;
    BlockDescriptor<algorithmFPType> matRightBlock;
    BlockDescriptor<algorithmFPType> resultBlock;

    DAAL_CHECK_STATUS(status, matLeft->getBlockOfRows(0, nMatLeft, ReadWriteMode::readOnly, matLeftBlock));
    DAAL_CHECK_STATUS(status, matRight->getBlockOfRows(0, nMatRight, ReadWriteMode::readOnly, matRightBlock));

    DAAL_CHECK_STATUS(status, result->getBlockOfRows(0, nMatLeft, ReadWriteMode::writeOnly, resultBlock));

    const services::Buffer<algorithmFPType> matLeftBuf  = matLeftBlock.getBuffer();
    const services::Buffer<algorithmFPType> matRightBuf = matRightBlock.getBuffer();

    services::Buffer<algorithmFPType> rBuf = resultBlock.getBuffer();

    DAAL_CHECK_STATUS(status, lazyAllocate(_sqrMatLeft, nMatLeft));
    DAAL_CHECK_STATUS(status, lazyAllocate(_sqrMatRight, nMatRight));

    UniversalBuffer sqrA1U = context.allocate(TypeIds::id<algorithmFPType>(), nMatLeft, &status);
    DAAL_CHECK_STATUS_VAR(status);
    UniversalBuffer sqrA2U = context.allocate(TypeIds::id<algorithmFPType>(), nMatRight, &status);
    DAAL_CHECK_STATUS_VAR(status);

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(KernelRBF.sumOfSquares);

        Reducer::reduce(Reducer::BinaryOp::SUM_OF_SQUARES, Layout::RowMajor, matLeftBuf, _sqrMatLeft, nMatLeft, pMatLeft, &status);
        DAAL_CHECK_STATUS_VAR(status);
        Reducer::reduce(Reducer::BinaryOp::SUM_OF_SQUARES, Layout::RowMajor, matRightBuf, _sqrMatRight, nMatRight, pMatRight, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(KernelRBF.gemm);
        DAAL_CHECK_STATUS(status, BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::NoTrans, math::Transpose::Trans, nMatLeft,
                                                                  nMatRight, pMatLeft, algorithmFPType(-2.0), matLeftBuf, pMatLeft, 0, matRightBuf,
                                                                  pMatRight, 0, algorithmFPType(0.0), rBuf, nMatRight, 0));
    }

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, nMatLeft, nMatRight);
    DAAL_CHECK_STATUS(status, computeRBF(_sqrMatLeft, _sqrMatRight, nMatRight, coeff, rBuf, nMatLeft, nMatRight));

    DAAL_CHECK_STATUS(status, matLeft->releaseBlockOfRows(matLeftBlock));
    DAAL_CHECK_STATUS(status, matRight->releaseBlockOfRows(matRightBlock));
    DAAL_CHECK_STATUS(status, result->releaseBlockOfRows(resultBlock));

    return status;
}

} // namespace internal
} // namespace rbf
} // namespace kernel_function
} // namespace algorithms
} // namespace daal

#endif
