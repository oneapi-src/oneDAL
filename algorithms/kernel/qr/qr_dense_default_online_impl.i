/* file: qr_dense_default_online_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of qrs
//--
*/

#ifndef __QR_KERNEL_ONLINE_IMPL_I__
#define __QR_KERNEL_ONLINE_IMPL_I__

#include "service_lapack.h"
#include "service_memory.h"
#include "service_math.h"
#include "service_defines.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

#include "qr_dense_default_kernel.h"

#include "qr_dense_default_impl.i"
#include "qr_dense_default_batch_impl.i"

#include "threading.h"

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace qr
{
namespace internal
{

/**
 *  \brief Kernel for QR QR calculation
 */
template <typename algorithmFPType, daal::algorithms::qr::Method method, CpuType cpu>
Status QROnlineKernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable *const *a,
                                                           const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par)
{
    QRBatchKernel<algorithmFPType, method, cpu> kernel;
    return kernel.compute(na, a, nr, r, par);
}

/**
 *  \brief Kernel for QR QR calculation
 */
template <typename algorithmFPType, daal::algorithms::qr::Method method, CpuType cpu>
Status QROnlineKernel<algorithmFPType, method, cpu>::finalizeCompute(const size_t na, const NumericTable *const *a,
                                                                   const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par)
{
    const NumericTable *ntAux2_0 = a[0];
    NumericTable       *ntR      = r[1];

    size_t nBlocks = na / 2;

    size_t n       = ntAux2_0->getNumberOfColumns();

    /* Step 2 */

    const NumericTable *const *step2ntIn = a;
    TArray<NumericTable *, cpu> step2ntOut(nBlocks);
    Status s;
    for(auto k = 0; k < nBlocks; k++)
    {
        step2ntOut[k] = new HomogenNumericTableCPU<algorithmFPType, cpu>(n, n, s);
        DAAL_CHECK_STATUS_VAR(s);
    }

    QRDistributedStep2Kernel<algorithmFPType, method, cpu> kernel;
    s = kernel.compute( nBlocks, step2ntIn, nBlocks + 2, ntR, step2ntOut.get(), par );
    if(s)
    {
        /* Step 3 */
        BlockMicroTable<algorithmFPType, writeOnly, cpu> mtQ(r[0]);
        size_t computedRows   = 0;
        for (auto i = 0; i < nBlocks; i++)
        {
            const NumericTable *ntAux1i = a[nBlocks + i];
            size_t m = ntAux1i->getNumberOfRows();

            algorithmFPType *Qi;
            mtQ.getBlockOfRows( computedRows, m, &Qi );

            HomogenNumericTableCPU<algorithmFPType, cpu> ntQi   (Qi, n, m, s);
            DAAL_CHECK_STATUS_VAR(s);

            const NumericTable *step3ntIn[2] = {ntAux1i, step2ntOut[i]};
            NumericTable *step3ntOut[1] = {&ntQi};

            QRDistributedStep3Kernel<algorithmFPType, method, cpu> kernelStep3;
            s = kernelStep3.compute(2, step3ntIn, 1, step3ntOut, par);

            mtQ.release();

            computedRows += m;
            if(!s)
                break;
        }
    }
    for(auto k = 0; k < nBlocks; k++)
        delete step2ntOut[k];
    return s;
}

} // namespace daal::internal
}
}
} // namespace daal

#endif
