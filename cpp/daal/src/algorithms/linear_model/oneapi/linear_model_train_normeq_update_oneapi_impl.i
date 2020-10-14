/* file: linear_model_train_normeq_update_oneapi_impl.i */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
//  Implementation of common base classes for normal equations model training.
//--
*/

#include "src/algorithms/linear_model/oneapi/linear_model_train_normeq_kernel_oneapi.h"
#include "src/sycl/blas_gpu.h"
#include "services/internal/execution_context.h"
#include "src/externals/service_ittnotify.h"
#include "src/algorithms/linear_model/oneapi/cl_kernel/reduce_results.cl"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace normal_equations
{
namespace training
{
namespace internal
{
using namespace daal::services::internal::sycl;

template <typename algorithmFPType>
services::Status UpdateKernelOneAPI<algorithmFPType>::compute(NumericTable & xTable, NumericTable & yTable, NumericTable & xtx, NumericTable & xty,
                                                              bool interceptFlag)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(computeUpdate);
    services::Status status;

    const size_t nRows           = xTable.getNumberOfRows();
    const size_t nCols           = xTable.getNumberOfColumns();
    const size_t nResponses      = yTable.getNumberOfColumns();
    const size_t nBetas          = nCols + 1;
    const size_t nBetasIntercept = (interceptFlag ? nBetas : (nBetas - 1));

    BlockDescriptor<algorithmFPType> xtxBlock;
    BlockDescriptor<algorithmFPType> xtyBlock;

    DAAL_CHECK_STATUS(status, xtx.getBlockOfRows(0, nBetasIntercept, ReadWriteMode::readWrite, xtxBlock));
    DAAL_CHECK_STATUS(status, xty.getBlockOfRows(0, nResponses, ReadWriteMode::readWrite, xtyBlock));

    auto & context                                      = services::internal::getDefaultContext();
    services::internal::Buffer<algorithmFPType> xtxBuff = xtxBlock.getBuffer();
    services::internal::Buffer<algorithmFPType> xtyBuff = xtyBlock.getBuffer();

    DAAL_CHECK_STATUS(status, xtx.releaseBlockOfRows(xtxBlock));
    DAAL_CHECK_STATUS(status, xty.releaseBlockOfRows(xtyBlock));

    const size_t nRowsPerBlock = 90000;

    size_t nBlocks = nRows / nRowsPerBlock;
    if (nBlocks * nRowsPerBlock < nRows)
    {
        ++nBlocks;
    }

    services::internal::Buffer<algorithmFPType> sumXBuf;
    services::internal::Buffer<algorithmFPType> sumYBuf;
    services::internal::Buffer<algorithmFPType> onesBuf;

    if (interceptFlag)
    {
        const TypeIds::Id idType = TypeIds::id<algorithmFPType>();

        sumXBuf = xtxBuff.getSubBuffer(nBetasIntercept * nCols, nBetasIntercept, status);
        DAAL_CHECK_STATUS_VAR(status);

        UniversalBuffer sumYBufTmp = context.allocate(idType, nResponses, status);
        DAAL_CHECK_STATUS_VAR(status);
        sumYBuf = sumYBufTmp.get<algorithmFPType>();
        context.fill(sumYBuf, 0.0, status);
        DAAL_CHECK_STATUS_VAR(status);

        UniversalBuffer onesBufTmp = context.allocate(idType, nRowsPerBlock, status);
        DAAL_CHECK_STATUS_VAR(status);
        onesBuf = onesBufTmp.get<algorithmFPType>();

        context.fill(onesBuf, 1.0, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

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

        DAAL_CHECK_STATUS(status, xTable.getBlockOfRows(startRow, endRow - startRow, ReadWriteMode::readOnly, xBlock));
        DAAL_CHECK_STATUS(status, yTable.getBlockOfRows(startRow, endRow - startRow, ReadWriteMode::readOnly, yBlock));

        const services::internal::Buffer<algorithmFPType> xBuf = xBlock.getBuffer();
        const services::internal::Buffer<algorithmFPType> yBuf = yBlock.getBuffer();

        const size_t xNRows   = endRow - startRow;
        const size_t xNCols   = nCols;
        const size_t xtxNCols = nBetasIntercept;
        const size_t yNCols   = nResponses;
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(computeUpdate.syrkX);
            /* Compute XTX for each block and reduce to final result*/
            status = BlasGpu<algorithmFPType>::xsyrk(math::Layout::RowMajor, math::UpLo::Upper, math::Transpose::Trans, xNCols, xNRows,
                                                     algorithmFPType(1.0), xBuf, xNCols, 0, algorithmFPType(1.0), xtxBuff, xtxNCols, 0);
        }
        DAAL_CHECK_STATUS_VAR(status);

        {
            DAAL_ITTNOTIFY_SCOPED_TASK(computeUpdate.gemmXY);
            /* Compute XTY (in real YTX) for each block and reduce to final result*/
            status =
                BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::Trans, math::Transpose::NoTrans, yNCols, xNCols, xNRows,
                                                algorithmFPType(1.0), yBuf, yNCols, 0, xBuf, xNCols, 0, algorithmFPType(1.0), xtyBuff, xtxNCols, 0);
        }
        DAAL_CHECK_STATUS_VAR(status);

        if (interceptFlag)
        {
            {
                DAAL_ITTNOTIFY_SCOPED_TASK(computeUpdate.gemm1X);
                /* Compute reduce X in columns for each block and reduce it to final result*/
                status = BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::NoTrans, math::Transpose::NoTrans, 1, xNCols,
                                                         xNRows, algorithmFPType(1.0), onesBuf, nRowsPerBlock, 0, xBuf, xNCols, 0,
                                                         algorithmFPType(1.0), sumXBuf, xNCols, 0);
            }
            DAAL_CHECK_STATUS_VAR(status);

            {
                DAAL_ITTNOTIFY_SCOPED_TASK(computeUpdate.gemm1Y);
                /* Compute reduce Y in columns for each block and reduce it to final result*/
                status = BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::NoTrans, math::Transpose::NoTrans, 1, yNCols,
                                                         xNRows, algorithmFPType(1.0), onesBuf, nRowsPerBlock, 0, yBuf, yNCols, 0,
                                                         algorithmFPType(1.0), sumYBuf, yNCols, 0);
            }
            DAAL_CHECK_STATUS_VAR(status);
        }

        DAAL_CHECK_STATUS(status, xTable.releaseBlockOfRows(xBlock));
        DAAL_CHECK_STATUS(status, yTable.releaseBlockOfRows(yBlock));
    }

    if (interceptFlag)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(computeUpdate.copyResults);

        algorithmFPType nrowsVal = static_cast<algorithmFPType>(nRows);
        const services::internal::Buffer<algorithmFPType> nrowsBuf(&nrowsVal, 1);
        DAAL_CHECK_STATUS(status, reduceResults(sumXBuf, nCols, 1, nrowsBuf, 0, 1, 1));

        DAAL_CHECK_STATUS(status, reduceResults(xtyBuff, nCols, nBetasIntercept, sumYBuf, 0, 1, nResponses));
    }

    return status;
}

template <typename algorithmFPType>
services::Status UpdateKernelOneAPI<algorithmFPType>::reduceResults(services::internal::Buffer<algorithmFPType> & dst, size_t dstOffset,
                                                                    size_t dstStride, const services::internal::Buffer<algorithmFPType> & src,
                                                                    size_t srcOffset, size_t srcStride, size_t count)
{
    services::Status status;

    ExecutionContextIface & ctx    = services::internal::getDefaultContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    const services::String options = getKeyFPType<algorithmFPType>();
    services::String cachekey("__daal_algorithms_linear_model_copy_");
    cachekey.add(options);
    factory.build(ExecutionTargetIds::device, cachekey.c_str(), clKernelCopy, options.c_str(), status);
    DAAL_CHECK_STATUS_VAR(status);

    const char * const kernelName = "reduceResults";
    KernelPtr kernel              = factory.getKernel(kernelName, status);
    DAAL_CHECK_STATUS_VAR(status);

    KernelArguments args(6);
    args.set(0, dst, AccessModeIds::write);
    args.set(1, dstOffset);
    args.set(2, dstStride);
    args.set(3, src, AccessModeIds::read);
    args.set(4, srcOffset);
    args.set(5, srcStride);

    KernelRange range(count);

    ctx.run(range, kernel, args, status);

    return status;
}

} // namespace internal
} // namespace training
} // namespace normal_equations
} // namespace linear_model
} // namespace algorithms
} // namespace daal
