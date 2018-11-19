/* file: linear_model_train_qr_merge_impl.i */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of common base classes for normal equations model training.
//--
*/

#include "linear_model_train_qr_kernel.h"
#include "service_lapack.h"

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
Status MergeKernel<algorithmFPType, cpu>::compute(size_t n, NumericTable **partialr, NumericTable **partialqty,
                          NumericTable &rTable, NumericTable &qtyTable)
{
    size_t nBetas    (rTable.getNumberOfRows());
    size_t nBetas2   (2 * nBetas);
    size_t nResponses(qtyTable.getNumberOfRows());

    TArray<algorithmFPType, cpu> rMerge(nBetas * nBetas2);
    DAAL_CHECK_MALLOC(rMerge.get());

    TArray<algorithmFPType, cpu> qtyMerge(nResponses * nBetas2);
    DAAL_CHECK_MALLOC(qtyMerge.get());

    TArray<algorithmFPType, cpu> tau(nBetas);
    DAAL_CHECK_MALLOC(tau.get());

    WriteRowsType rFinalBlock(rTable, 0, nBetas);
    DAAL_CHECK_BLOCK_STATUS(rFinalBlock);
    algorithmFPType *rFinal = rFinalBlock.get();

    WriteRowsType qtyFinalBlock(qtyTable, 0, nResponses);
    DAAL_CHECK_BLOCK_STATUS(qtyFinalBlock);
    algorithmFPType *qtyFinal = qtyFinalBlock.get();

    ReadRowsType rBlock(partialr[0], 0, nBetas);
    DAAL_CHECK_BLOCK_STATUS(rBlock);
    const algorithmFPType *r = rBlock.get();

    ReadRowsType qtyBlock(partialqty[0], 0, nResponses);
    DAAL_CHECK_BLOCK_STATUS(qtyBlock);
    const algorithmFPType *qty = qtyBlock.get();

    const size_t rSizeInBytes   =     nBetas * nBetas * sizeof(algorithmFPType);
    const size_t qtySizeInBytes = nResponses * nBetas * sizeof(algorithmFPType);
    daal_memcpy_s(rFinal, rSizeInBytes, r, rSizeInBytes);
    daal_memcpy_s(qtyFinal, qtySizeInBytes, qty, qtySizeInBytes);

    DAAL_INT lwork;
    Status st = CommonKernel<algorithmFPType, cpu>::computeWorkSize(nBetas2, nBetas, nResponses, lwork);
    DAAL_CHECK_STATUS_VAR(st);

    TArray<algorithmFPType, cpu> work(lwork);
    DAAL_CHECK_MALLOC(work.get());

    for (size_t i = 1; i < n; i++)
    {
        rBlock.set(partialr[i], 0, nBetas);
        DAAL_CHECK_BLOCK_STATUS(rBlock);
        r = rBlock.get();

        qtyBlock.set(partialqty[i], 0, nResponses);
        DAAL_CHECK_BLOCK_STATUS(qtyBlock);
        qty = qtyBlock.get();

        st |= CommonKernel<algorithmFPType, cpu>::merge(nBetas, nResponses,
            r, qty, rFinal, qtyFinal, rMerge.get(), qtyMerge.get(), rFinal, qtyFinal, tau.get(), work.get(),
            lwork);
        DAAL_CHECK_STATUS_VAR(st);
    }
    return st;
}

}
}
}
}
}
}
