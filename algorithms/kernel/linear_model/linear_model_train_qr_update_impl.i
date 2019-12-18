/* file: linear_model_train_qr_update_impl.i */
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
//  Implementation of common base classes for normal equations model training.
//--
*/

#include "linear_model_train_qr_kernel.h"
#include "service_lapack.h"
#include "service_error_handling.h"
#include "threading.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace qr
{
namespace training
{
namespace internal
{
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services::internal;

template <typename algorithmFPType, CpuType cpu>
ThreadingTask<algorithmFPType, cpu>::ThreadingTask(DAAL_INT nBetasIntercept, DAAL_INT nRows, DAAL_INT nResponses, Status & st)
    : _nBetasIntercept(nBetasIntercept),
      _nRows(nRows),
      _nResponses(nResponses),
      _lwork(-1),
      work(),
      _xBlock(),
      _yBlock(),
      tau(nBetasIntercept),
      qrBuffer(nBetasIntercept * nRows),
      qtyBuffer(nResponses * nRows),
      qrR(nBetasIntercept * nBetasIntercept),
      qrQTY(nBetasIntercept * nResponses),
      qrRNew(nBetasIntercept * nBetasIntercept),
      qrQTYNew(nBetasIntercept * nResponses),
      qrRMerge(2 * nBetasIntercept * nBetasIntercept),
      qrQTYMerge(2 * nBetasIntercept * nResponses)
{
    if (!tau.get() || !qrBuffer.get() || !qtyBuffer.get() || !qrR.get() || !qrQTY.get() || !qrRNew.get() || !qrQTYNew.get() || !qrRMerge.get()
        || !qrQTYMerge.get())
    {
        st.add(ErrorMemoryAllocationFailed);
        return;
    }

    st |= allocateWorkBuffer();
}

template <typename algorithmFPType, CpuType cpu>
ThreadingTask<algorithmFPType, cpu> * ThreadingTask<algorithmFPType, cpu>::create(DAAL_INT nBetasIntercept, DAAL_INT nRows, DAAL_INT nResponses)
{
    Status st;
    ThreadingTask<algorithmFPType, cpu> * res = new ThreadingTask<algorithmFPType, cpu>(nBetasIntercept, nRows, nResponses, st);
    if (!st)
    {
        delete res;
        return nullptr;
    }
    return res;
}

template <typename algorithmFPType, CpuType cpu>
Status ThreadingTask<algorithmFPType, cpu>::allocateWorkBuffer()
{
    DAAL_INT nRowsMax = ((_nRows > _nBetasIntercept) ? _nRows + _nBetasIntercept : 2 * _nBetasIntercept);
    Status st         = CommonKernel<algorithmFPType, cpu>::computeWorkSize(nRowsMax, _nBetasIntercept, _nResponses, _lwork);
    DAAL_CHECK_STATUS_VAR(st);

    work.reset(_lwork);
    DAAL_CHECK_MALLOC(work.get());

    return Status();
}

template <typename algorithmFPType, CpuType cpu>
Status ThreadingTask<algorithmFPType, cpu>::copyDataToBuffer(DAAL_INT startRow, DAAL_INT nRows, const NumericTable & xTable,
                                                             const NumericTable & yTable)
{
    const size_t nFeatures = xTable.getNumberOfColumns();
    _xBlock.set(const_cast<NumericTable &>(xTable), startRow, nRows);
    DAAL_CHECK_BLOCK_STATUS(_xBlock);
    const algorithmFPType * x = _xBlock.get();
    int result                = 0;

    if (nRows > _nRows)
    {
        qrBuffer.reset(nRows * _nBetasIntercept);
        qtyBuffer.reset(nRows * _nResponses);
    }
    /* Copy matrix X to temporary buffer in order not to damage it */
    if (nFeatures == _nBetasIntercept)
    {
        const size_t xSize = nFeatures * nRows * sizeof(algorithmFPType);
        result |= daal::services::internal::daal_memcpy_s(qrBuffer.get(), xSize, x, xSize);
    }
    else
    {
        const size_t rowSize = nFeatures * sizeof(algorithmFPType);
        algorithmFPType * qr = qrBuffer.get();
        for (size_t i = 0; i < nRows; i++)
        {
            result |= daal::services::internal::daal_memcpy_s(qr + i * _nBetasIntercept, rowSize, x + i * nFeatures, rowSize);
            qr[i * _nBetasIntercept + _nBetasIntercept - 1] = 1.0;
        }
    }

    _yBlock.set(const_cast<NumericTable &>(yTable), startRow, nRows);
    DAAL_CHECK_BLOCK_STATUS(_yBlock);
    const algorithmFPType * y = _yBlock.get();

    /* Copy matrix Y to temporary buffer in order not to damage it */
    const size_t ySize = _nResponses * nRows * sizeof(algorithmFPType);
    result |= daal::services::internal::daal_memcpy_s(qtyBuffer.get(), ySize, y, ySize);

    return (!result) ? Status() : Status(ErrorMemoryCopyFailedInternal);
}

template <typename algorithmFPType, CpuType cpu>
Status ThreadingTask<algorithmFPType, cpu>::reduce(algorithmFPType * r, algorithmFPType * qty)
{
    return CommonKernel<algorithmFPType, cpu>::merge(_nBetasIntercept, _nResponses, qrRNew.get(), qrQTYNew.get(), qrR.get(), qrQTY.get(),
                                                     qrRMerge.get(), qrQTYMerge.get(), r, qty, tau.get(), work.get(), _lwork);
}

template <typename algorithmFPType, CpuType cpu>
Status ThreadingTask<algorithmFPType, cpu>::reduceInPlace(algorithmFPType * r, algorithmFPType * qty)
{
    return CommonKernel<algorithmFPType, cpu>::merge(_nBetasIntercept, _nResponses, r, qty, qrR.get(), qrQTY.get(), qrRMerge.get(), qrQTYMerge.get(),
                                                     r, qty, tau.get(), work.get(), _lwork);
}

template <typename algorithmFPType, CpuType cpu>
Status ThreadingTask<algorithmFPType, cpu>::update(DAAL_INT startRow, DAAL_INT nRows, const NumericTable & xTable, const NumericTable & yTable)
{
    Status st = copyDataToBuffer(startRow, nRows, xTable, yTable);

    if (st)
    {
        st = CommonKernel<algorithmFPType, cpu>::computeQRForBlock(_nBetasIntercept, nRows, qrBuffer.get(), _nResponses, qtyBuffer.get(),
                                                                   qrRNew.get(), qrQTYNew.get(), tau.get(), work.get(), _lwork);
        return st ? reduce(qrR.get(), qrQTY.get()) : st;
    }
    return st;
}

template <typename algorithmFPType, CpuType cpu>
Status UpdateKernel<algorithmFPType, cpu>::compute(const NumericTable & xTable, const NumericTable & yTable, NumericTable & rTable,
                                                   NumericTable & qtyTable, bool initializeResult, bool interceptFlag)
{
    DAAL_INT nRows      = (DAAL_INT)xTable.getNumberOfRows();    /* vectors */
    DAAL_INT nFeatures  = (DAAL_INT)xTable.getNumberOfColumns(); /* features */
    DAAL_INT nResponses = (DAAL_INT)yTable.getNumberOfColumns(); /* variables */
    DAAL_INT nBetas     = nFeatures + 1;

    size_t nBetasIntercept = (interceptFlag ? nBetas : (nBetas - 1));

    WriteRowsType rBlock(rTable, 0, nBetasIntercept);
    DAAL_CHECK_BLOCK_STATUS(rBlock);
    algorithmFPType * r = rBlock.get();

    WriteRowsType qtyBlock(qtyTable, 0, nResponses);
    DAAL_CHECK_BLOCK_STATUS(qtyBlock);
    algorithmFPType * qty = qtyBlock.get();

    /* Initialize output arrays by zero in case of batch mode */
    if (initializeResult)
    {
        service_memset<algorithmFPType, cpu>(r, 0, nBetasIntercept * nBetasIntercept);
        service_memset<algorithmFPType, cpu>(qty, 0, nResponses * nBetasIntercept);
    }

    /* Split rows by blocks. Each block should contain at least nBetasIntercept rows */
#if (__CPUID__(DAAL_CPU) == __avx512_mic__)
    size_t nRowsInBlock = (nRows <= 10000) ? 1024 : ((nRows >= 1000000) ? 512 : 2048);
#else
    size_t nRowsInBlock = 128;
#endif
    if (nRowsInBlock < nBetasIntercept) nRowsInBlock = nBetasIntercept;

    size_t nBlocks  = nRows / nRowsInBlock;
    size_t tailSize = nRows - nBlocks * nRowsInBlock;
    if (tailSize > nBetasIntercept) nBlocks++;

    /* Create TLS */
    daal::tls<ThreadingTaskType *> tls([=]() -> ThreadingTaskType * { return ThreadingTaskType::create(nBetasIntercept, nRowsInBlock, nResponses); });

    SafeStatus safeStat;
    daal::threader_for(nBlocks, nBlocks, [=, &tls, &xTable, &yTable, &safeStat](int iBlock) {
        ThreadingTaskType * tlsLocal = tls.local();
        if (!tlsLocal)
        {
            safeStat.add(services::ErrorMemoryAllocationFailed);
            return;
        }

        size_t startRow  = iBlock * nRowsInBlock;
        size_t blockSize = (iBlock == nBlocks - 1) ? nRows - startRow : nRowsInBlock;

        Status localSt = tlsLocal->update(startRow, blockSize, xTable, yTable);
        DAAL_CHECK_STATUS_THR(localSt);
    });

    Status st = safeStat.detach();
    tls.reduce([=, &st](ThreadingTaskType * tlsLocal) {
        if (!tlsLocal) return;

        if (st) st = tlsLocal->reduceInPlace(r, qty);
        delete tlsLocal;
    });

    return st;
}

} // namespace internal
} // namespace training
} // namespace qr
} // namespace linear_model
} // namespace algorithms
} // namespace daal
