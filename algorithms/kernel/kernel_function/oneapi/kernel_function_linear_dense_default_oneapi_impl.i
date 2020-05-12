/* file: kernel_function_linear_dense_default_oneapi_impl.i */
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
//  Linear kernel functions implementation
//--
*/

#ifndef __KERNEL_FUNCTION_LINEAR_DENSE_DEFAULT_ONEAPI_IMPL_I__
#define __KERNEL_FUNCTION_LINEAR_DENSE_DEFAULT_ONEAPI_IMPL_I__

#include "algorithms/kernel_function/kernel_function_types_linear.h"

#include "externals/service_stat.h"
#include "algorithms/kernel/service_error_handling.h"
#include "algorithms/kernel/kernel_function/oneapi/cl_kernels/kernel_function.cl"
#include "externals/service_ittnotify.h"
#include "service/kernel/oneapi/blas_gpu.h"
#include "service/kernel/oneapi/reducer.h"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace linear
{
namespace internal
{
using namespace daal::oneapi::internal;

template <typename algorithmFPType>
services::Status KernelImplLinearOneAPI<defaultDense, algorithmFPType>::computeInternalVectorVector(NumericTable * vecLeft, NumericTable * vecRight,
                                                                                                    NumericTable * result, const ParameterBase * par)
{
    return services::ErrorMethodNotImplemented;
}

template <typename algorithmFPType>
services::Status KernelImplLinearOneAPI<defaultDense, algorithmFPType>::computeInternalMatrixVector(NumericTable * matLeft, NumericTable * vecRight,
                                                                                                    NumericTable * result, const ParameterBase * par)
{
    return services::ErrorMethodNotImplemented;
}

template <typename algorithmFPType>
services::Status KernelImplLinearOneAPI<defaultDense, algorithmFPType>::computeInternalMatrixMatrix(NumericTable * matLeft, NumericTable * matRight,
                                                                                                    NumericTable * result, const ParameterBase * par)
{
    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    const size_t nMatLeft  = matLeft->getNumberOfRows();
    const size_t nMatRight = matRight->getNumberOfRows();

    const size_t pMatLeft  = matLeft->getNumberOfColumns();
    const size_t pMatRight = matRight->getNumberOfColumns();
    DAAL_ASSERT(pMatLeft == pMatRight);

    const Parameter * linPar    = static_cast<const Parameter *>(par);
    const algorithmFPType alpha = algorithmFPType(linPar->k);
    const algorithmFPType beta  = algorithmFPType(linPar->b);

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(KernelLinearOneAPI.gemm);

        BlockDescriptor<algorithmFPType> matLeftBlock;
        BlockDescriptor<algorithmFPType> matRightBlock;
        BlockDescriptor<algorithmFPType> resultBlock;

        DAAL_CHECK_STATUS(status, matLeft->getBlockOfRows(0, nMatLeft, ReadWriteMode::readOnly, matLeftBlock));
        DAAL_CHECK_STATUS(status, matRight->getBlockOfRows(0, nMatRight, ReadWriteMode::readOnly, matRightBlock));

        DAAL_CHECK_STATUS(status, result->getBlockOfRows(0, nMatLeft, ReadWriteMode::writeOnly, resultBlock));

        const services::Buffer<algorithmFPType> matLeftBuff  = matLeftBlock.getBuffer();
        const services::Buffer<algorithmFPType> matRightBuff = matRightBlock.getBuffer();

        services::Buffer<algorithmFPType> resultBuff = resultBlock.getBuffer();

        if (beta != 0.0)
        {
            context.fill(resultBuff, 1.0, &status);
            DAAL_CHECK_STATUS_VAR(status);
        }

        DAAL_CHECK_STATUS(status, BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::NoTrans, math::Transpose::Trans, nMatLeft,
                                                                  nMatRight, pMatLeft, alpha, matLeftBuff, pMatLeft, 0, matRightBuff, pMatRight, 0,
                                                                  beta, resultBuff, nMatRight, 0));

        DAAL_CHECK_STATUS(status, matLeft->releaseBlockOfRows(matLeftBlock));
        DAAL_CHECK_STATUS(status, matRight->releaseBlockOfRows(matRightBlock));
        DAAL_CHECK_STATUS(status, result->releaseBlockOfRows(resultBlock));
    }

    return status;
}

} // namespace internal
} // namespace linear
} // namespace kernel_function
} // namespace algorithms
} // namespace daal

#endif
