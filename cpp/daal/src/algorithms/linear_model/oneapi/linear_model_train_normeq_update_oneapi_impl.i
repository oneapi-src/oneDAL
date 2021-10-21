/* file: linear_model_train_normeq_update_oneapi_impl.i */
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
//  Implementation of common base classes for normal equations model training.
//--
*/

#include "src/algorithms/linear_model/oneapi/linear_model_train_normeq_kernel_oneapi.h"
#include "src/sycl/blas_gpu.h"
#include "services/internal/execution_context.h"
#include "src/externals/service_profiler.h"
#include "src/algorithms/linear_model/oneapi/cl_kernel/reduce_results.cl"
#include "src/services/service_data_utils.h"

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

    const size_t nRows      = xTable.getNumberOfRows();
    const size_t nCols      = xTable.getNumberOfColumns();
    const size_t nResponses = yTable.getNumberOfColumns();
    const size_t nBetas     = nCols + 1;
    DAAL_ASSERT((interceptFlag ? (nBetas >= 0) : (nBetas >= 1)));
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
    const size_t nBlocks       = (nRows / nRowsPerBlock) + (bool(nRows % nRowsPerBlock) ? 1 : 0);

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nBlocks, nRowsPerBlock);

    services::internal::Buffer<algorithmFPType> sumXBuf;
    services::internal::Buffer<algorithmFPType> sumYBuf;
    services::internal::Buffer<algorithmFPType> onesBuf;

    if (interceptFlag)
    {
        const TypeIds::Id idType = TypeIds::id<algorithmFPType>();
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nBetasIntercept, nCols);
        DAAL_OVERFLOW_CHECK_BY_ADDING(size_t, nBetasIntercept, (nBetasIntercept * nCols));
        DAAL_ASSERT(xtxBuff.size() >= (nBetasIntercept + nBetasIntercept * nCols));
        sumXBuf = xtxBuff.getSubBuffer(nBetasIntercept * nCols, nBetasIntercept, status);
        DAAL_CHECK_STATUS_VAR(status);

        UniversalBuffer sumYBufTmp = context.allocate(idType, nResponses, status);
        DAAL_CHECK_STATUS_VAR(status);

        sumYBuf = sumYBufTmp.get<algorithmFPType>();
        context.fill(sumYBuf, algorithmFPType(0), status);
        DAAL_CHECK_STATUS_VAR(status);

        UniversalBuffer onesBufTmp = context.allocate(idType, nRowsPerBlock, status);
        DAAL_CHECK_STATUS_VAR(status);
        onesBuf = onesBufTmp.get<algorithmFPType>();

        context.fill(onesBuf, algorithmFPType(1), status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    for (size_t blockIdx = 0; blockIdx < nBlocks; ++blockIdx)
    {
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, blockIdx, nRowsPerBlock);
        const size_t startRow = blockIdx * nRowsPerBlock;
        DAAL_OVERFLOW_CHECK_BY_ADDING(size_t, startRow, nRowsPerBlock);
        const size_t endRow = ((startRow + nRowsPerBlock) > nRows) ? nRows : (startRow + nRowsPerBlock);

        BlockDescriptor<algorithmFPType> xBlock;
        BlockDescriptor<algorithmFPType> yBlock;

        DAAL_CHECK_STATUS(status, xTable.getBlockOfRows(startRow, endRow - startRow, ReadWriteMode::readOnly, xBlock));
        DAAL_CHECK_STATUS(status, yTable.getBlockOfRows(startRow, endRow - startRow, ReadWriteMode::readOnly, yBlock));

        const services::internal::Buffer<algorithmFPType> xBuf = xBlock.getBuffer();
        const services::internal::Buffer<algorithmFPType> yBuf = yBlock.getBuffer();

        DAAL_ASSERT(endRow >= startRow);
        const size_t xNRows   = endRow - startRow;
        const size_t xNCols   = nCols;
        const size_t xtxNCols = nBetasIntercept;
        const size_t yNCols   = nResponses;
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(computeUpdate.syrkX);

            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, xNRows, xNCols);
            DAAL_ASSERT(xBuf.size() >= xNRows * xNCols);
            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, xNCols, xNCols);
            DAAL_ASSERT(xtxBuff.size() >= xNCols * xNCols);

            /* Compute XTX for each block and reduce to final result */
            status = BlasGpu<algorithmFPType>::xsyrk(math::Layout::RowMajor, math::UpLo::Upper, math::Transpose::Trans, xNCols, xNRows,
                                                     algorithmFPType(1.0), xBuf, xNCols, 0, algorithmFPType(1.0), xtxBuff, xtxNCols, 0);
        }
        DAAL_CHECK_STATUS_VAR(status);

        {
            DAAL_ITTNOTIFY_SCOPED_TASK(computeUpdate.gemmXY);

            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, xNRows, xNCols);
            DAAL_ASSERT(xBuf.size() >= xNRows * xNCols);
            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, yNCols, xNCols);
            DAAL_ASSERT(xtyBuff.size() >= yNCols * xNCols);
            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, xNRows, yNCols);
            DAAL_ASSERT(yBuf.size() >= xNRows * yNCols);

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

                DAAL_ASSERT(onesBuf.size() >= xNRows);
                DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, xNCols, xNRows);
                DAAL_ASSERT(xBuf.size() >= xNCols * xNRows);
                DAAL_ASSERT(sumXBuf.size() >= xNCols);

                /* Compute reduce X in columns for each block and reduce it to final result*/
                status = BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::NoTrans, math::Transpose::NoTrans, 1, xNCols,
                                                         xNRows, algorithmFPType(1.0), onesBuf, nRowsPerBlock, 0, xBuf, xNCols, 0,
                                                         algorithmFPType(1.0), sumXBuf, xNCols, 0);
            }
            DAAL_CHECK_STATUS_VAR(status);

            {
                DAAL_ITTNOTIFY_SCOPED_TASK(computeUpdate.gemm1Y);

                DAAL_ASSERT(onesBuf.size() >= xNRows);
                DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, yNCols, xNRows);
                DAAL_ASSERT(yBuf.size() >= yNCols * xNRows);
                DAAL_ASSERT(sumYBuf.size() >= yNCols);

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
        const services::internal::Buffer<algorithmFPType> nrowsBuf(&nrowsVal, 1, status);
        DAAL_CHECK_STATUS_VAR(status);

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

    KernelArguments args(6, status);
    DAAL_CHECK_STATUS_VAR(status);

    DAAL_ASSERT(count <= services::internal::MaxVal<uint32_t>::get());
    DAAL_ASSERT(count >= 1);

    DAAL_ASSERT(dstStride <= services::internal::MaxVal<uint32_t>::get());
    DAAL_ASSERT(dstOffset <= services::internal::MaxVal<uint32_t>::get());

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, dstStride, (count - 1));
    DAAL_OVERFLOW_CHECK_BY_ADDING(uint32_t, dstOffset, (dstStride * (count - 1)));
    DAAL_ASSERT(dst.size() >= (dstStride * (count - 1) + dstOffset));

    args.set(0, dst, AccessModeIds::write);
    args.set(1, static_cast<uint32_t>(dstOffset));
    args.set(2, static_cast<uint32_t>(dstStride));

    DAAL_ASSERT(srcStride <= services::internal::MaxVal<uint32_t>::get());
    DAAL_ASSERT(srcOffset <= services::internal::MaxVal<uint32_t>::get());

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, srcStride, (count - 1));
    DAAL_OVERFLOW_CHECK_BY_ADDING(uint32_t, srcOffset, (srcStride * (count - 1)));
    DAAL_ASSERT(src.size() >= (srcStride * (count - 1) + srcOffset));

    args.set(3, src, AccessModeIds::read);
    args.set(4, static_cast<uint32_t>(srcOffset));
    args.set(5, static_cast<uint32_t>(srcStride));

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
