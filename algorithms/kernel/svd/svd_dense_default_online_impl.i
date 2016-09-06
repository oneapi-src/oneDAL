/* file: svd_dense_default_online_impl.i */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of svds
//--
*/

#ifndef __SVD_KERNEL_ONLINE_IMPL_I__
#define __SVD_KERNEL_ONLINE_IMPL_I__

#include "service_memory.h"
#include "service_math.h"
#include "service_defines.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

#include "svd_dense_default_impl.i"
#include "svd_dense_default_distr_impl.i"

#include "threading.h"

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace svd
{
namespace internal
{

template <typename interm, daal::algorithms::svd::Method method, CpuType cpu>
void SVDOnlineKernel<interm, method, cpu>::compute(const size_t na, const NumericTable *const *a,
                                                   const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par)
{
    size_t i, j;
    svd::Parameter defaultParams;
    const svd::Parameter *svdPar = &defaultParams;

    if ( par != 0 )
    {
        svdPar = static_cast<const svd::Parameter *>(par);
    }

    BlockMicroTable<interm, readOnly , cpu> mtAi   (a[0]);
    BlockMicroTable<interm, writeOnly, cpu> mtAux2i(r[1]);

    interm *Ai;
    interm *Aux2i;

    size_t  n = mtAi.getFullNumberOfColumns();
    size_t  m = mtAi.getFullNumberOfRows();

    interm *Aux1iT = (interm *)daal::services::daal_malloc(n * m * sizeof(interm));
    interm *Aux2iT = (interm *)daal::services::daal_malloc(n * n * sizeof(interm));

    mtAi   .getBlockOfRows( 0, m, &Ai    ); /*         Ai [m][n] */
    mtAux2i.getBlockOfRows( 0, n, &Aux2i ); /* Aux2i = Ri [n][n] */

    for ( i = 0 ; i < n ; i++ )
    {
        for ( j = 0 ; j < m; j++ )
        {
            Aux1iT[i * m + j] = Ai[i + j * n];
        }
    }

    compute_QR_on_one_node<interm, cpu>( m, n, Aux1iT, m, Aux2iT, n );

    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        interm *Aux1i;
        BlockMicroTable<interm, writeOnly, cpu> mtAux1i(r[0]);
        mtAux1i.getBlockOfRows( 0, m, &Aux1i ); /* Aux1i = Qin[m][n] */
        for ( i = 0 ; i < n ; i++ )
        {
            for ( j = 0 ; j < m; j++ )
            {
                Aux1i[i + j * n] = Aux1iT[i * m + j];
            }
        }
        mtAux1i.release();
    }

    for ( i = 0 ; i < n ; i++ )
    {
        for ( j = 0 ; j <= i; j++ )
        {
            Aux2i[i + j * n] = Aux2iT[i * n + j];
        }
        for (     ; j < n; j++ )
        {
            Aux2i[i + j * n] = 0.0;
        }
    }

    mtAi   .release();
    mtAux2i.release();

    daal::services::daal_free(Aux1iT);
    daal::services::daal_free(Aux2iT);
}

template <typename interm, daal::algorithms::svd::Method method, CpuType cpu>
void SVDOnlineKernel<interm, method, cpu>::finalizeCompute(const size_t na, const NumericTable *const *a,
                                                           const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par)
{
    size_t i, j;
    svd::Parameter defaultParams;
    const svd::Parameter *svdPar = &defaultParams;

    if ( par != 0 )
    {
        svdPar = static_cast<const svd::Parameter *>(par);
    }

    const NumericTable *ntAux2_0 = a[0];
    NumericTable       *ntSigma  = r[0];
    NumericTable       *ntV      = r[2];

    size_t nBlocks = na / 2;

    size_t n       = ntAux2_0->getNumberOfColumns();

    /* Step 2 */

    const NumericTable *const *step2ntIn = a;
    NumericTable **step2ntOut = new NumericTable*[nBlocks + 2];
    step2ntOut[0] = ntSigma;
    step2ntOut[1] = ntV;
    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        for(size_t k = 0; k < nBlocks; k++)
        {
            step2ntOut[2 + k] = new HomogenNumericTableCPU<interm, cpu>(n, n);
        }
    }

    SVDDistributedStep2Kernel<interm, method, cpu> kernel;
    kernel.compute( nBlocks, step2ntIn, nBlocks + 2, step2ntOut, par );

    /* Step 3 */

    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        BlockMicroTable<interm, writeOnly, cpu> mtQ(r[1]);

        size_t computedRows   = 0;

        for (i = 0; i < nBlocks; i++)
        {
            const NumericTable *ntAux1i = a[nBlocks + i];
            size_t m = ntAux1i->getNumberOfRows();

            interm *Qi;
            mtQ.getBlockOfRows( computedRows, m, &Qi );

            HomogenNumericTableCPU<interm, cpu> ntQi   (Qi, n, m);

            const NumericTable *step3ntIn[2] = {ntAux1i, step2ntOut[2 + i]};
            NumericTable *step3ntOut[1] = {&ntQi};

            SVDDistributedStep3Kernel<interm, method, cpu> kernelStep3;
            kernelStep3.compute(2, step3ntIn, 1, step3ntOut, par);

            mtQ.release();

            computedRows += m;

            delete step2ntOut[2 + i];
        }
    }

    delete[] step2ntOut;
}

} // namespace daal::internal
}
}
} // namespace daal

#endif
