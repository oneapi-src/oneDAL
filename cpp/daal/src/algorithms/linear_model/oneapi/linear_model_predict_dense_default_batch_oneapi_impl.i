/* file: linear_model_predict_dense_default_batch_oneapi_impl.i */
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
//  Common functions for linear regression predictions calculation
//--
*/

#ifndef __LINEAR_MODEL_PREDICT_DENSE_DEFAULT_BATCH_ONEAPI_IMPL_I__
#define __LINEAR_MODEL_PREDICT_DENSE_DEFAULT_BATCH_ONEAPI_IMPL_I__

#include "src/algorithms/linear_model/oneapi/linear_model_predict_kernel_oneapi.h"
#include "src/data_management/service_numeric_table.h"
#include "src/sycl/blas_gpu.h"
#include "services/internal/execution_context.h"
#include "src/services/service_data_utils.h"
#include "src/algorithms/linear_model/oneapi/cl_kernel/linear_model_prediction.cl"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace prediction
{
namespace internal
{
using namespace daal::services::internal::sycl;

template <typename algorithmFPType>
services::Status PredictKernelOneAPI<algorithmFPType, defaultDense>::addBetaIntercept(const services::internal::Buffer<algorithmFPType> & betaTable,
                                                                                      const size_t nBetas,
                                                                                      services::internal::Buffer<algorithmFPType> & yTable,
                                                                                      const size_t yNRows, const size_t yNCols)
{
    services::Status status;

    ExecutionContextIface & ctx    = services::internal::getDefaultContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    const services::String options = getKeyFPType<algorithmFPType>();
    services::String cachekey("__daal_algorithms_linear_model_prediction_");
    cachekey.add(options);
    factory.build(ExecutionTargetIds::device, cachekey.c_str(), clKernelPrediction, options.c_str(), status);
    DAAL_CHECK_STATUS_VAR(status);

    const char * const kernelName = "addBetaIntercept";
    KernelPtr kernel              = factory.getKernel(kernelName, status);
    DAAL_CHECK_STATUS_VAR(status);

    DAAL_ASSERT(yNCols <= services::internal::MaxVal<uint32_t>::get());
    DAAL_ASSERT(nBetas <= services::internal::MaxVal<uint32_t>::get());

    DAAL_ASSERT(betaTable.size() >= nBetas * yNCols);
    DAAL_ASSERT(yTable.size() >= yNRows * yNCols);

    KernelArguments args(4, status);
    DAAL_CHECK_STATUS_VAR(status);

    args.set(0, betaTable, AccessModeIds::read);
    args.set(1, static_cast<uint32_t>(nBetas));
    args.set(2, yTable, AccessModeIds::write);
    args.set(3, static_cast<uint32_t>(yNCols));

    KernelRange range(yNRows, yNCols);

    ctx.run(range, kernel, args, status);

    return status;
}

template <typename algorithmFPType>
services::Status PredictKernelOneAPI<algorithmFPType, defaultDense>::compute_impl(const NumericTable * a, const NumericTable * b, NumericTable * r,
                                                                                  bool interceptFlag)
{
    services::Status status;

    NumericTable * xTable    = const_cast<NumericTable *>(a);
    NumericTable * yTable    = const_cast<NumericTable *>(r);
    NumericTable * betaTable = const_cast<NumericTable *>(b);

    const size_t nRows      = xTable->getNumberOfRows();
    const size_t nBetas     = betaTable->getNumberOfColumns();
    const size_t nResponses = betaTable->getNumberOfRows();

    const size_t nRowsPerBlock = 90000;

    const size_t nBlocks = (nRows / nRowsPerBlock) + (bool(nRows % nRowsPerBlock) ? 1 : 0);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nBlocks, nRowsPerBlock);

    BlockDescriptor<algorithmFPType> betaBlock;
    DAAL_CHECK_STATUS(status, betaTable->getBlockOfRows(0, nResponses, ReadWriteMode::readOnly, betaBlock));

    const services::internal::Buffer<algorithmFPType> betaBuf = betaBlock.getBuffer();

    for (size_t blockIdx = 0; blockIdx < nBlocks; ++blockIdx)
    {
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, blockIdx, nRowsPerBlock);
        const size_t startRow = blockIdx * nRowsPerBlock;
        DAAL_OVERFLOW_CHECK_BY_ADDING(size_t, startRow, nRowsPerBlock);
        const size_t endRow = ((startRow + nRowsPerBlock) > nRows) ? nRows : (startRow + nRowsPerBlock);
        DAAL_ASSERT(endRow >= startRow);

        BlockDescriptor<algorithmFPType> xBlock;
        BlockDescriptor<algorithmFPType> yBlock;

        DAAL_CHECK_STATUS(status, xTable->getBlockOfRows(startRow, endRow - startRow, ReadWriteMode::readOnly, xBlock));
        DAAL_CHECK_STATUS(status, yTable->getBlockOfRows(startRow, endRow - startRow, ReadWriteMode::readWrite, yBlock));

        const services::internal::Buffer<algorithmFPType> xBuf = xBlock.getBuffer();
        services::internal::Buffer<algorithmFPType> yBuf       = yBlock.getBuffer();

        const size_t xNRows = endRow - startRow;
        DAAL_ASSERT(nBetas >= 1);
        const size_t xNCols = nBetas - 1;
        const size_t yNCols = nResponses;

        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, xNRows, xNCols);
        DAAL_ASSERT(xBuf.size() >= xNRows * xNCols);
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, yNCols, xNCols);
        DAAL_ASSERT(betaBuf.size() >= yNCols * xNCols);
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, xNRows, yNCols);
        DAAL_ASSERT(yBuf.size() >= xNRows * yNCols);

        /* SYRK: Compute beta*xTable for each block */
        status = BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::NoTrans, math::Transpose::Trans, xNRows, yNCols, xNCols,
                                                 algorithmFPType(1.0), xBuf, xNCols, 0, betaBuf, nBetas, 1, algorithmFPType(0.0), yBuf, yNCols, 0);

        DAAL_CHECK_STATUS_VAR(status);

        if (interceptFlag)
        {
            DAAL_CHECK_STATUS(status, addBetaIntercept(betaBuf, nBetas, yBuf, xNRows, yNCols));
        }

        DAAL_CHECK_STATUS(status, xTable->releaseBlockOfRows(xBlock));
        DAAL_CHECK_STATUS(status, yTable->releaseBlockOfRows(yBlock));
    }

    DAAL_CHECK_STATUS(status, betaTable->releaseBlockOfRows(betaBlock));

    return status;
}

template <typename algorithmFPType>
services::Status PredictKernelOneAPI<algorithmFPType, defaultDense>::compute(const NumericTable * a, const linear_model::Model * m, NumericTable * r)
{
    linear_model::Model * model = const_cast<linear_model::Model *>(m);
    return compute_impl(a, model->getBeta().get(), r, model->getInterceptFlag());
}

} // namespace internal
} // namespace prediction
} // namespace linear_model
} // namespace algorithms
} // namespace daal

#endif
