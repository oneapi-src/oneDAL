/* file: kernel_function_rbf_dense_default_oneapi_impl.i */
/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_math.h"
#include "src/externals/service_profiler.h"
#include "src/sycl/blas_gpu.h"
#include "src/sycl/reducer.h"

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
using namespace daal::services::internal::sycl;
using namespace daal::services::internal::sycl::math;

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

    auto & context = services::internal::getDefaultContext();

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

    const services::internal::Buffer<algorithmFPType> matLeftBuf  = matLeftBlock.getBuffer();
    const services::internal::Buffer<algorithmFPType> matRightBuf = matRightBlock.getBuffer();

    services::internal::Buffer<algorithmFPType> rBuf = resultBlock.getBuffer();

    DAAL_CHECK_STATUS(status, Helper::lazyAllocate(_sqrMatLeft, nMatLeft));
    DAAL_CHECK_STATUS(status, Helper::lazyAllocate(_sqrMatRight, nMatRight));

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(KernelRBF.sumOfSquares);

        Reducer::reduce(Reducer::BinaryOp::SUM_OF_SQUARES, Layout::RowMajor, matLeftBuf, _sqrMatLeft, nMatLeft, pMatLeft, status);
        DAAL_CHECK_STATUS_VAR(status);
        Reducer::reduce(Reducer::BinaryOp::SUM_OF_SQUARES, Layout::RowMajor, matRightBuf, _sqrMatRight, nMatRight, pMatRight, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(KernelRBF.gemm);
        DAAL_CHECK_STATUS(status, BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::NoTrans, math::Transpose::Trans, nMatLeft,
                                                                  nMatRight, pMatLeft, algorithmFPType(-2.0), matLeftBuf, pMatLeft, 0, matRightBuf,
                                                                  pMatRight, 0, algorithmFPType(0.0), rBuf, nMatRight, 0));
    }

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, nMatLeft, nMatRight);
    DAAL_CHECK_STATUS(status, Helper::computeRBF(_sqrMatLeft, _sqrMatRight, nMatRight, coeff, rBuf, nMatLeft, nMatRight));

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
