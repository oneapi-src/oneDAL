/* file: kernel_function_rbf_csr_fast_oneapi_impl.i */
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

#ifndef __KERNEL_FUNCTION_RBF_CSR_FAST_IMPL_ONEAPI_I__
#define __KERNEL_FUNCTION_RBF_CSR_FAST_IMPL_ONEAPI_I__

#include "algorithms/kernel_function/kernel_function_types_rbf.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_math.h"
#include "src/externals/service_profiler.h"
#include "src/sycl/spblas_gpu.h"
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
services::Status KernelImplRBFOneAPI<fastCSR, algorithmFPType>::computeInternalVectorVector(NumericTable * vecLeft, NumericTable * vecRight,
                                                                                            NumericTable * result, const ParameterBase * par)
{
    return services::ErrorMethodNotImplemented;
}

template <typename algorithmFPType>
services::Status KernelImplRBFOneAPI<fastCSR, algorithmFPType>::computeInternalMatrixVector(NumericTable * matLeft, NumericTable * vecRight,
                                                                                            NumericTable * result, const ParameterBase * par)
{
    return services::ErrorMethodNotImplemented;
}

template <typename algorithmFPType>
services::Status KernelImplRBFOneAPI<fastCSR, algorithmFPType>::computeInternalMatrixMatrix(NumericTable * matLeft, NumericTable * matRight,
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

    DAAL_CHECK_STATUS(status, Helper::lazyAllocate(_sqrMatLeft, nMatLeft));
    DAAL_CHECK_STATUS(status, Helper::lazyAllocate(_sqrMatRight, nMatRight));

    CSRBlockDescriptor<algorithmFPType> matLeftBD, matRightBD;

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

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(KernelRBF.sumOfSquaresCSR);

        DAAL_CHECK_STATUS(status, Helper::sumOfSquaresCSR(matLeftValuesBuff, matLeftRowIndicesBuff, _sqrMatLeft, nMatLeft));
        DAAL_CHECK_STATUS_VAR(status);
        DAAL_CHECK_STATUS(status, Helper::sumOfSquaresCSR(matRightValuesBuff, matRightRowIndicesBuff, _sqrMatRight, nMatRight));
        DAAL_CHECK_STATUS_VAR(status);
    }
    BlockDescriptor<algorithmFPType> resultBlock;
    DAAL_CHECK_STATUS(status, result->getBlockOfRows(0, nMatLeft, ReadWriteMode::writeOnly, resultBlock));
    services::internal::Buffer<algorithmFPType> resultBuff = resultBlock.getBuffer();

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(KernelRBF.gemmCSR);

        DAAL_CHECK_STATUS(status, math::SpBlasGpu<algorithmFPType>::xgemm(
                                      math::Transpose::Trans, math::Transpose::NoTrans, nMatLeft, nMatRight, pMatLeft, algorithmFPType(-2.0),
                                      matLeftValuesBuff, matLeftColumnIndicesBuff, matLeftRowIndicesBuff, matRightValuesBuff,
                                      matRightColumnIndicesBuff, matRightRowIndicesBuff, algorithmFPType(0.0), resultBuff, nMatRight, 0));
    }

    DAAL_CHECK_STATUS(status, Helper::computeRBF(_sqrMatLeft, _sqrMatRight, nMatRight, coeff, resultBuff, nMatLeft, nMatRight));

    DAAL_CHECK_STATUS(status, matLeftCSR->releaseSparseBlock(matLeftBD));
    DAAL_CHECK_STATUS(status, matRightCSR->releaseSparseBlock(matRightBD));
    DAAL_CHECK_STATUS(status, result->releaseBlockOfRows(resultBlock));

    return status;
}

} // namespace internal
} // namespace rbf
} // namespace kernel_function
} // namespace algorithms
} // namespace daal

#endif
