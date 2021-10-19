/* file: kernel_function_linear_csr_fast_oneapi_impl.i */
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
//  Linear kernel functions implementation
//--
*/

#ifndef __KERNEL_FUNCTION_LINEAR_CSR_FAST_ONEAPI_IMPL_I__
#define __KERNEL_FUNCTION_LINEAR_CSR_FAST_ONEAPI_IMPL_I__

#include "algorithms/kernel_function/kernel_function_types_linear.h"

#include "src/externals/service_stat.h"
#include "src/algorithms/service_error_handling.h"
#include "src/externals/service_profiler.h"
#include "src/sycl/spblas_gpu.h"

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
using namespace daal::services::internal::sycl;

template <typename algorithmFPType>
services::Status KernelImplLinearOneAPI<fastCSR, algorithmFPType>::computeInternalVectorVector(NumericTable * vecLeft, NumericTable * vecRight,
                                                                                               NumericTable * result, const ParameterBase * par)
{
    return services::ErrorMethodNotImplemented;
}

template <typename algorithmFPType>
services::Status KernelImplLinearOneAPI<fastCSR, algorithmFPType>::computeInternalMatrixVector(NumericTable * matLeft, NumericTable * vecRight,
                                                                                               NumericTable * result, const ParameterBase * par)
{
    return services::ErrorMethodNotImplemented;
}

template <typename algorithmFPType>
services::Status KernelImplLinearOneAPI<fastCSR, algorithmFPType>::computeInternalMatrixMatrix(NumericTable * matLeft, NumericTable * matRight,
                                                                                               NumericTable * result, const ParameterBase * par)
{
    services::Status status;

    auto & context = services::internal::getDefaultContext();

    const size_t nMatLeft  = matLeft->getNumberOfRows();
    const size_t nMatRight = matRight->getNumberOfRows();

    const size_t pMatLeft  = matLeft->getNumberOfColumns();
    const size_t pMatRight = matRight->getNumberOfColumns();
    DAAL_ASSERT(pMatLeft == pMatRight);

    const Parameter * linPar    = static_cast<const Parameter *>(par);
    const algorithmFPType alpha = algorithmFPType(linPar->k);
    const algorithmFPType beta  = algorithmFPType(linPar->b);

    CSRBlockDescriptor<algorithmFPType> matLeftBD, matRightBD;
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(KernelLinearCSROneAPI.gemm);

        CSRNumericTableIface * matLeftCSR = dynamic_cast<CSRNumericTableIface *>(matLeft);
        DAAL_CHECK(matLeftCSR, services::ErrorIncorrectTypeOfInputNumericTable);
        CSRNumericTableIface * matRightCSR = dynamic_cast<CSRNumericTableIface *>(matRight);
        DAAL_CHECK(matRightCSR, services::ErrorIncorrectTypeOfInputNumericTable);
        DAAL_CHECK_STATUS(status, matLeftCSR->getSparseBlock(0, nMatLeft, readOnly, matLeftBD));
        DAAL_CHECK_STATUS(status, matRightCSR->getSparseBlock(0, nMatRight, readOnly, matRightBD));

        const auto matLeftValuesBuff        = matLeftBD.getBlockValuesBuffer();
        const auto matLeftColumnIndicesBuff = matLeftBD.getBlockColumnIndicesBuffer();
        const auto matLeftRowIndicesBuff    = matLeftBD.getBlockRowIndicesBuffer();

        const auto matRightValuesBuff        = matRightBD.getBlockValuesBuffer();
        const auto matRightColumnIndicesBuff = matRightBD.getBlockColumnIndicesBuffer();
        const auto matRightRowIndicesBuff    = matRightBD.getBlockRowIndicesBuffer();

        BlockDescriptor<algorithmFPType> resultBlock;
        DAAL_CHECK_STATUS(status, result->getBlockOfRows(0, nMatLeft, ReadWriteMode::writeOnly, resultBlock));

        auto resultBuff = resultBlock.getBuffer();

        if (beta != 0.0)
        {
            context.fill(resultBuff, 1.0, status);
            DAAL_CHECK_STATUS_VAR(status);
        }

        DAAL_CHECK_STATUS(
            status, math::SpBlasGpu<algorithmFPType>::xgemm(math::Transpose::Trans, math::Transpose::NoTrans, nMatLeft, nMatRight, pMatLeft, alpha,
                                                            matLeftValuesBuff, matLeftColumnIndicesBuff, matLeftRowIndicesBuff, matRightValuesBuff,
                                                            matRightColumnIndicesBuff, matRightRowIndicesBuff, beta, resultBuff, nMatRight, 0));

        DAAL_CHECK_STATUS(status, matLeftCSR->releaseSparseBlock(matLeftBD));
        DAAL_CHECK_STATUS(status, matRightCSR->releaseSparseBlock(matRightBD));
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
