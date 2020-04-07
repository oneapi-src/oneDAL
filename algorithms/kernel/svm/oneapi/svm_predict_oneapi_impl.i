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

#include "externals/service_memory.h"
#include "service/kernel/data_management/service_numeric_table.h"
#include "service/kernel/oneapi/blas_gpu.h"
#include "externals/service_ittnotify.h"
#include "data_management/data/numeric_table_sycl_homogen.h"

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
using namespace daal::services;
using namespace daal::services::internal;
using namespace daal::oneapi::internal;

template <typename algorithmFPType>
services::Status SVMPredictImplOneAPI<defaultDense, algorithmFPType>::compute(const NumericTablePtr & xTable, const daal::algorithms::Model * m,
                                                                              NumericTable & r, const daal::algorithms::Parameter * par)
{
    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    const size_t nVectors  = xTable->getNumberOfRows();
    const size_t nFeatures = xTable->getNumberOfColumns();

    BlockDescriptor<algorithmFPType> rBlock;
    DAAL_CHECK_STATUS(status, r.getBlockOfRows(0, nVectors, ReadWriteMode::writeOnly, rBlock));
    auto distanceBuff = rBlock.getBuffer();

    Model * model = static_cast<Model *>(const_cast<daal::algorithms::Model *>(m));
    kernel_function::KernelIfacePtr kernel;
    {
        svm::Parameter * parameter = dynamic_cast<svm::Parameter *>(const_cast<daal::algorithms::Parameter *>(par));
        if (parameter) kernel = parameter->kernel->clone();
    }

    DAAL_CHECK(kernel, services::ErrorNullParameterNotSupported);

    auto svCoeffTable = model->getClassificationCoefficients();
    const size_t nSV  = svCoeffTable->getNumberOfRows();
    BlockDescriptor<algorithmFPType> svCoeffBlock;
    DAAL_CHECK_STATUS(status, svCoeffTable->getBlockOfRows(0, nSV, ReadWriteMode::readOnly, svCoeffBlock));
    auto svCoeffBuff = svCoeffBlock.getBuffer();

    if (nSV == 0)
    {
        context.fill(distanceBuff, 0.0, &status);
        return status;
    }

    const algorithmFPType bias(model->getBias());
    context.fill(distanceBuff, double(bias), &status);
    DAAL_CHECK_STATUS_VAR(status);

    auto svTable = model->getSupportVectors();

    const size_t nRowsPerBlock = 8192;
    const size_t nBlocks       = nVectors / nRowsPerBlock + !!(nVectors % nRowsPerBlock);

    auto shU       = context.allocate(TypeIds::id<algorithmFPType>(), nRowsPerBlock * nSV, &status);
    auto shResBuff = shU.template get<algorithmFPType>();

    for (size_t blockIdx = 0; blockIdx < nBlocks; blockIdx++)
    {
        const size_t startRow = blockIdx * nRowsPerBlock;
        size_t endRow         = startRow + nRowsPerBlock;
        if (endRow > nVectors) endRow = nVectors;
        const size_t nRowsPerBlockReal = endRow - startRow;

        NumericTablePtr shResNT = SyclHomogenNumericTable<algorithmFPType>::create(shResBuff, nSV, nRowsPerBlockReal, &status);

        BlockDescriptor<algorithmFPType> xBlock;
        DAAL_CHECK_STATUS(status, xTable->getBlockOfRows(startRow, nRowsPerBlockReal, ReadWriteMode::readOnly, xBlock));
        const services::Buffer<algorithmFPType> xBuf = xBlock.getBuffer();

        NumericTablePtr xBlockNT = SyclHomogenNumericTable<algorithmFPType>::create(xBuf, nFeatures, nRowsPerBlockReal, &status);

        auto kfResultPtr = new kernel_function::Result();
        DAAL_CHECK_MALLOC(kfResultPtr)
        kernel_function::ResultPtr shRes(kfResultPtr);
        shRes->set(kernel_function::values, shResNT);
        kernel->setResult(shRes);
        kernel->getInput()->set(kernel_function::X, xBlockNT);
        kernel->getInput()->set(kernel_function::Y, svTable);
        kernel->getParameter()->computationMode = kernel_function::matrixMatrix;
        status                                  = kernel->computeNoThrow();
        if (!status) return Status(services::ErrorSVMPredictKernerFunctionCall).add(status);

        {
            DAAL_ITTNOTIFY_SCOPED_TASK(gemm);
            DAAL_CHECK_STATUS(status, BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::NoTrans, math::Transpose::NoTrans,
                                                                      nRowsPerBlockReal, 1, nSV, algorithmFPType(1.0), shResBuff, nSV, 0, svCoeffBuff,
                                                                      1, 0, algorithmFPType(1.0), distanceBuff, 1, startRow));
        }
    }

    DAAL_CHECK_STATUS(status, r.releaseBlockOfRows(rBlock));
    DAAL_CHECK_STATUS(status, svCoeffTable->releaseBlockOfRows(svCoeffBlock));

    return status;
}

} // namespace internal
} // namespace prediction
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
