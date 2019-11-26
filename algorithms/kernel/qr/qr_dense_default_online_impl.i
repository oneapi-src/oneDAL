/* file: qr_dense_default_online_impl.i */
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
Status QROnlineKernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[],
                                                             const daal::algorithms::Parameter * par)
{
    QRBatchKernel<algorithmFPType, method, cpu> kernel;
    return kernel.compute(na, a, nr, r, par);
}

/**
 *  \brief Kernel for QR QR calculation
 */
template <typename algorithmFPType, daal::algorithms::qr::Method method, CpuType cpu>
Status QROnlineKernel<algorithmFPType, method, cpu>::finalizeCompute(const size_t na, const NumericTable * const * a, const size_t nr,
                                                                     NumericTable * r[], const daal::algorithms::Parameter * par)
{
    const NumericTable * ntAux2_0 = a[0];
    NumericTable * ntR            = r[1];

    size_t nBlocks = na / 2;

    size_t n = ntAux2_0->getNumberOfColumns();

    /* Step 2 */

    const NumericTable * const * step2ntIn = a;
    TArray<NumericTable *, cpu> step2ntOut(nBlocks);
    Status s;
    for (auto k = 0; k < nBlocks; k++)
    {
        step2ntOut[k] = new HomogenNumericTableCPU<algorithmFPType, cpu>(n, n, s);
        DAAL_CHECK_STATUS_VAR(s);
    }

    QRDistributedStep2Kernel<algorithmFPType, method, cpu> kernel;
    s = kernel.compute(nBlocks, step2ntIn, nBlocks + 2, ntR, step2ntOut.get(), par);
    if (s)
    {
        /* Step 3 */
        BlockMicroTable<algorithmFPType, writeOnly, cpu> mtQ(r[0]);
        size_t computedRows = 0;
        for (auto i = 0; i < nBlocks; i++)
        {
            const NumericTable * ntAux1i = a[nBlocks + i];
            size_t m                     = ntAux1i->getNumberOfRows();

            algorithmFPType * Qi;
            mtQ.getBlockOfRows(computedRows, m, &Qi);

            HomogenNumericTableCPU<algorithmFPType, cpu> ntQi(Qi, n, m, s);
            DAAL_CHECK_STATUS_VAR(s);

            const NumericTable * step3ntIn[2] = { ntAux1i, step2ntOut[i] };
            NumericTable * step3ntOut[1]      = { &ntQi };

            QRDistributedStep3Kernel<algorithmFPType, method, cpu> kernelStep3;
            s = kernelStep3.compute(2, step3ntIn, 1, step3ntOut, par);

            mtQ.release();

            computedRows += m;
            if (!s) break;
        }
    }
    for (auto k = 0; k < nBlocks; k++)
    {
        delete step2ntOut[k];
        step2ntOut[k] = nullptr;
    }
    return s;
}

} // namespace internal
} // namespace qr
} // namespace algorithms
} // namespace daal

#endif
