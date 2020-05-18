/* file: svm_predict_oneapi_impl.i */
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
//  SVM prediction algorithm implementation
//--
*/

#ifndef __SVM_PREDICT_ONEAPI_IMPL_I__
#define __SVM_PREDICT_ONEAPI_IMPL_I__

#include "service/kernel/oneapi/blas_gpu.h"
#include "externals/service_ittnotify.h"
#include "data_management/data/numeric_table_sycl_homogen.h"
#include "algorithms/kernel/svm/oneapi/svm_helper_oneapi.h"

DAAL_ITTNOTIFY_DOMAIN(svm_predict.default.batch);

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace prediction
{
namespace internal
{
using namespace daal::internal;
using namespace daal::oneapi::internal;

template <typename algorithmFPType>
services::Status SVMPredictImplOneAPI<defaultDense, algorithmFPType>::compute(const NumericTablePtr & xTable, Model * model, NumericTable & result,
                                                                              const svm::Parameter * par)
{
    services::Status status;
    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    const size_t nVectors  = xTable->getNumberOfRows();
    const size_t nFeatures = xTable->getNumberOfColumns();

    BlockDescriptor<algorithmFPType> resultBlock;
    DAAL_CHECK_STATUS(status, result.getBlockOfRows(0, nVectors, ReadWriteMode::writeOnly, resultBlock));
    auto distanceBuff = resultBlock.getBuffer();

    kernel_function::KernelIfacePtr kernel = par->kernel->clone();

    auto svCoeffTable = model->getClassificationCoefficients();
    const size_t nSV  = svCoeffTable->getNumberOfRows();

    if (nSV == 0)
    {
        context.fill(distanceBuff, 0.0, &status);
        return status;
    }

    BlockDescriptor<algorithmFPType> svCoeffBlock;
    DAAL_CHECK_STATUS(status, svCoeffTable->getBlockOfRows(0, nSV, ReadWriteMode::readOnly, svCoeffBlock));
    auto svCoeffBuff = svCoeffBlock.getBuffer();

    const algorithmFPType bias(model->getBias());
    context.fill(distanceBuff, double(bias), &status);
    DAAL_CHECK_STATUS_VAR(status);

    auto svTable = model->getSupportVectors();

    const size_t nRowsPerBlock = 1024;
    const size_t nBlocks       = nVectors / nRowsPerBlock + !!(nVectors % nRowsPerBlock);

    auto kernelResU    = context.allocate(TypeIds::id<algorithmFPType>(), nRowsPerBlock * nSV, &status);
    auto kernelResBuff = kernelResU.template get<algorithmFPType>();

    kernel_function::ResultPtr shRes(new kernel_function::Result());
    DAAL_CHECK_MALLOC(shRes)

    for (size_t blockIdx = 0; blockIdx < nBlocks; blockIdx++)
    {
        const size_t startRow          = blockIdx * nRowsPerBlock;
        const size_t offestRow         = startRow + nRowsPerBlock;
        const size_t endRow            = utils::internal::min(offestRow, nVectors);
        const size_t nRowsPerBlockReal = endRow - startRow;

        NumericTablePtr kernelResNT = SyclHomogenNumericTable<algorithmFPType>::create(kernelResBuff, nSV, nRowsPerBlockReal, &status);
        DAAL_CHECK_STATUS_VAR(status);

        BlockDescriptor<algorithmFPType> xBlock;
        DAAL_CHECK_STATUS(status, xTable->getBlockOfRows(startRow, nRowsPerBlockReal, ReadWriteMode::readOnly, xBlock));
        const services::Buffer<algorithmFPType> xBuf = xBlock.getBuffer();

        NumericTablePtr xBlockNT = SyclHomogenNumericTable<algorithmFPType>::create(xBuf, nFeatures, nRowsPerBlockReal, &status);
        DAAL_CHECK_STATUS_VAR(status);

        shRes->set(kernel_function::values, kernelResNT);
        kernel->setResult(shRes);

        kernel->getInput()->set(kernel_function::X, xBlockNT);
        kernel->getInput()->set(kernel_function::Y, svTable);
        kernel->getParameter()->computationMode = kernel_function::matrixMatrix;
        DAAL_CHECK(kernel->computeNoThrow(), services::ErrorSVMPredictKernerFunctionCall);

        {
            DAAL_ITTNOTIFY_SCOPED_TASK(gemm);
            DAAL_CHECK_STATUS(status, BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::NoTrans, math::Transpose::NoTrans,
                                                                      nRowsPerBlockReal, 1, nSV, algorithmFPType(1.0), kernelResBuff, nSV, 0,
                                                                      svCoeffBuff, 1, 0, algorithmFPType(1.0), distanceBuff, 1, startRow));
        }

        DAAL_CHECK_STATUS(status, xTable->releaseBlockOfRows(xBlock));
    }

    DAAL_CHECK_STATUS(status, result.releaseBlockOfRows(resultBlock));
    DAAL_CHECK_STATUS(status, svCoeffTable->releaseBlockOfRows(svCoeffBlock));

    return status;
}

} // namespace internal
} // namespace prediction
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
