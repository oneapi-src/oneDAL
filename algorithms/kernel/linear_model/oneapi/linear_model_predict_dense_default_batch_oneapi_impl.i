/* file: linear_model_predict_dense_default_batch_oneapi_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#include "linear_model_predict_kernel_oneapi.h"
#include "service_numeric_table.h"
#include "oneapi/blas_gpu.h"
#include "oneapi/internal/utils.h"
#include "cl_kernel/linear_model_prediction.cl"

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
using namespace daal::oneapi::internal;

template <typename algorithmFPType>
services::Status PredictKernelOneAPI<algorithmFPType, defaultDense>::addBetaIntercept(const services::Buffer<algorithmFPType> & betaTable,
                                                                                      const size_t nBetas, services::Buffer<algorithmFPType> & yTable,
                                                                                      const size_t yNRows, const size_t yNCols)
{
    services::Status status;

    ExecutionContextIface & ctx    = getDefaultContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    const services::String options = getKeyFPType<algorithmFPType>();
    services::String cachekey("__daal_algorithms_linear_model_prediction_");
    cachekey.add(options);
    factory.build(ExecutionTargetIds::device, cachekey.c_str(), clKernelPrediction, options.c_str());

    const char * const kernelName = "addBetaIntercept";
    KernelPtr kernel              = factory.getKernel(kernelName);

    KernelArguments args(4);
    args.set(0, betaTable, AccessModeIds::read);
    args.set(1, nBetas);
    args.set(2, yTable, AccessModeIds::write);
    args.set(3, yNCols);

    KernelRange range(yNRows, yNCols);

    ctx.run(range, kernel, args, &status);

    return status;
}

template <typename algorithmFPType>
services::Status PredictKernelOneAPI<algorithmFPType, defaultDense>::compute(const NumericTable * a, const linear_model::Model * m, NumericTable * r)
{
    services::Status status;
    linear_model::Model * const model = const_cast<linear_model::Model *>(m);

    NumericTable * xTable    = const_cast<NumericTable *>(a);
    NumericTable * yTable    = const_cast<NumericTable *>(r);
    NumericTable * betaTable = model->getBeta().get();

    const size_t nRows       = xTable->getNumberOfRows();
    const size_t nBetas      = betaTable->getNumberOfColumns();
    const size_t nResponses  = betaTable->getNumberOfRows();
    const bool interceptFlag = model->getInterceptFlag();

    const size_t nRowsPerBlock = 90000;

    size_t nBlocks = nRows / nRowsPerBlock;
    if (nBlocks * nRowsPerBlock < nRows)
    {
        ++nBlocks;
    }

    BlockDescriptor<algorithmFPType> betaBlock;
    DAAL_CHECK_STATUS(status, betaTable->getBlockOfRows(0, nResponses, ReadWriteMode::readOnly, betaBlock));
    const services::Buffer<algorithmFPType> betaBuf = betaBlock.getBuffer();

    for (size_t blockIdx = 0; blockIdx < nBlocks; ++blockIdx)
    {
        const size_t startRow = blockIdx * nRowsPerBlock;
        size_t endRow         = startRow + nRowsPerBlock;
        if (endRow > nRows)
        {
            endRow = nRows;
        };

        BlockDescriptor<algorithmFPType> xBlock;
        BlockDescriptor<algorithmFPType> yBlock;

        DAAL_CHECK_STATUS(status, xTable->getBlockOfRows(startRow, endRow - startRow, ReadWriteMode::readOnly, xBlock));
        DAAL_CHECK_STATUS(status, yTable->getBlockOfRows(startRow, endRow - startRow, ReadWriteMode::readWrite, yBlock));

        const services::Buffer<algorithmFPType> xBuf = xBlock.getBuffer();
        services::Buffer<algorithmFPType> yBuf       = yBlock.getBuffer();

        const size_t xNRows = endRow - startRow;
        const size_t xNCols = nBetas - 1;
        const size_t yNCols = nResponses;

        /* SYRK: Compute beta*xTable for each block */
        status = BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::NoTrans, math::Transpose::Trans, xNRows, yNCols, xNCols,
                                                 algorithmFPType(1.0), xBuf, xNCols, 0, betaBuf, nBetas, algorithmFPType(1), algorithmFPType(0.0),
                                                 yBuf, yNCols, 0);

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

} // namespace internal
} // namespace prediction
} // namespace linear_model
} // namespace algorithms
} // namespace daal

#endif
