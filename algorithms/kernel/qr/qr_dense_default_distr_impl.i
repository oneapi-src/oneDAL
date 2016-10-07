/* file: qr_dense_default_distr_impl.i */
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
//  Implementation of qrs
//--
*/

#ifndef __QR_KERNEL_DISTR_IMPL_I__
#define __QR_KERNEL_DISTR_IMPL_I__

#include "service_lapack.h"
#include "service_memory.h"
#include "service_math.h"
#include "service_defines.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

#include "qr_dense_default_impl.i"

#include "threading.h"

using namespace daal::internal;
using namespace daal::services::internal;

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
void QRDistributedStep2Kernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable *const *a,
                                                                     const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par)
{
    size_t i, j;
    qr::Parameter defaultParams;

    const NumericTable *ntAux2_0 = a[0];

    size_t nBlocks = na;

    size_t n   = ntAux2_0->getNumberOfColumns(); /* size of observations block */
    size_t nxb = n * nBlocks;

    algorithmFPType *Aux2T = (algorithmFPType *)daal::services::daal_malloc( sizeof(algorithmFPType) * n * nxb );
    algorithmFPType *RT    = (algorithmFPType *)daal::services::daal_malloc( sizeof(algorithmFPType) * n * n   );

    daal::threader_for( nBlocks, nBlocks, [=](int k)
    {
        algorithmFPType *Aux2 ;
        BlockMicroTable<algorithmFPType, readOnly, cpu> mtAux2 (a[k]);
        mtAux2.getBlockOfRows( 0, n, &Aux2  ); /* Aux2  [nxb][n] */
        for ( size_t i = 0 ; i < n ; i++ )
        {
            for ( size_t j = 0 ; j < n; j++ )
            {
                Aux2T[j * nxb + k * n + i] = Aux2[i * n + j];
            }
        }
        mtAux2.release();
    } );

    {
        /* By some reason, there was this part in Sample */
        for (size_t i = 0; i < n * n; i++) { RT[i] = 0.0; }
    }

    compute_QR_on_one_node<algorithmFPType, cpu>( nxb, n, Aux2T, nxb, RT, n );

    {
        daal::threader_for( nBlocks, nBlocks, [=](int k)
        {
            algorithmFPType *Aux3 ;
            BlockMicroTable<algorithmFPType, readOnly, cpu> mtAux3 (r[1 + k]);
            mtAux3.getBlockOfRows( 0, n, &Aux3  ); /* Aux2  [nxb][n] */
            for ( size_t i = 0 ; i < n ; i++ )
            {
                for ( size_t j = 0 ; j < n; j++ )
                {
                    Aux3[i * n + j] = Aux2T[j * nxb + k * n + i];
                }
            }
            mtAux3.release();
        } );

    }

    BlockMicroTable<algorithmFPType, writeOnly, cpu> mtR(r[0]);

    algorithmFPType *R;
    mtR.getBlockOfRows( 0, n, &R ); /* V[n][n] */

    for ( i = 0 ; i < n ; i++ )
    {
        for ( j = 0 ; j < n; j++ )
        {
            R[i + j * n] = RT[i * n + j];
        }
    }

    mtR.release();

    daal::services::daal_free( RT );
    daal::services::daal_free( Aux2T );
}

/**
 *  \brief Kernel for QR QR calculation
 */
template <typename algorithmFPType, daal::algorithms::qr::Method method, CpuType cpu>
void QRDistributedStep3Kernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable *const *a,
                                                                     const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par)
{
    size_t i, j;
    qr::Parameter defaultParams;

    size_t nBlocks     = na / 2;
    size_t mCalculated = 0;

    for(size_t k = 0; k < nBlocks; k++)
    {
        BlockMicroTable<algorithmFPType, readOnly, cpu> mtAux1i(a[k          ]);
        BlockMicroTable<algorithmFPType, readOnly, cpu> mtAux3i(a[k + nBlocks]);
        BlockMicroTable<algorithmFPType, writeOnly   , cpu> mtQi   (r[0]);

        size_t  n   = mtAux1i.getFullNumberOfColumns();
        size_t  m   = mtAux1i.getFullNumberOfRows();

        algorithmFPType *Qi   ;
        algorithmFPType *Aux1i;
        algorithmFPType *Aux3i;

        mtAux1i.getBlockOfRows( 0,           m, &Aux1i ); /* Aux1i = Qin[m][n] */
        mtAux3i.getBlockOfRows( 0,           n, &Aux3i ); /* Aux3i = Ri [n][n] */
        mtQi   .getBlockOfRows( mCalculated, m, &Qi    ); /*         Qi [m][n] */

        algorithmFPType *QiT    = (algorithmFPType *)daal::services::daal_malloc(sizeof(algorithmFPType) * n * m);
        algorithmFPType *Aux1iT = (algorithmFPType *)daal::services::daal_malloc(sizeof(algorithmFPType) * n * m);
        algorithmFPType *Aux3iT = (algorithmFPType *)daal::services::daal_malloc(sizeof(algorithmFPType) * n * n);

        for ( i = 0 ; i < n ; i++ )
        {
            for ( j = 0 ; j < m; j++ )
            {
                Aux1iT[i * m + j] = Aux1i[i + j * n];
            }
        }
        for ( i = 0 ; i < n ; i++ )
        {
            for ( j = 0 ; j < n; j++ )
            {
                Aux3iT[i * n + j] = Aux3i[i + j * n];
            }
        }

        DAAL_INT ldAux1i = m;
        DAAL_INT ldAux3i = n;
        DAAL_INT ldQi    = m;

        compute_gemm_on_one_node<algorithmFPType, cpu>( m, n, Aux1iT, ldAux1i, Aux3iT, ldAux3i, QiT, ldQi );

        for ( i = 0 ; i < n ; i++ )
        {
            for ( j = 0 ; j < m; j++ )
            {
                Qi[i + j * n] = QiT[i * m + j];
            }
        }

        mtAux1i.release();
        mtAux3i.release();
        mtQi   .release();

        daal::services::daal_free( QiT    );
        daal::services::daal_free( Aux1iT );
        daal::services::daal_free( Aux3iT );

        mCalculated += m;
    }
}

} // namespace daal::internal
}
}
} // namespace daal

#endif
