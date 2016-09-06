/* file: svd_dense_default_distr_impl.i */
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

#ifndef __SVD_KERNEL_DISTR_IMPL_I__
#define __SVD_KERNEL_DISTR_IMPL_I__

#include "service_memory.h"
#include "service_math.h"
#include "service_defines.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

#include "svd_dense_default_impl.i"

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
void SVDDistributedStep2Kernel<interm, method, cpu>::compute(const size_t na, const NumericTable *const *a,
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
    BlockMicroTable<interm, writeOnly, cpu> mtSigma(r[0]);

    size_t nBlocks = na;

    size_t n   = ntAux2_0->getNumberOfColumns();
    size_t nxb = n * nBlocks;

    interm *Sigma;

    mtSigma.getBlockOfRows( 0, 1, &Sigma ); /* Sigma [1][n]   */

    interm *Aux2T = (interm *)daal::services::daal_malloc( sizeof(interm) * n * nxb );
    interm *VT    = (interm *)daal::services::daal_malloc( sizeof(interm) * n * n   );
    interm *Aux3T = (interm *)daal::services::daal_malloc( sizeof(interm) * n * nxb );

    daal::threader_for( nBlocks, nBlocks, [=](int k)
    {
        interm *Aux2 ;
        BlockMicroTable<interm, readOnly, cpu> mtAux2 (a[k]);
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

    MKL_INT ldAux2 = nxb;
    MKL_INT ldAux3 = nxb;
    MKL_INT ldV    = n;

    MKL_INT ldR = n;
    MKL_INT ldU = n;
    interm *R = (interm *)daal::services::daal_malloc( sizeof(interm) * n * ldR); /* [n][n] */
    interm *U = (interm *)daal::services::daal_malloc( sizeof(interm) * n * ldU); /* [n][n] */

    {
        /* By some reason, there was this part in Sample */
        for (size_t i = 0; i < n * ldR; i++) { R[i] = 0.0; }
    }

    // Rc = P*R
    compute_QR_on_one_node<interm, cpu>( nxb, n, Aux2T, ldAux2, R, ldR );

    // Qn*R -> Qn*(U*Sigma*V) -> (Qn*U)*Sigma*V
    compute_svd_on_one_node<interm, cpu>( n, n, R, ldR, Sigma, U, ldU, VT, ldV );

    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        compute_gemm_on_one_node<interm, cpu>( nxb, n, Aux2T, ldAux2, U, ldU, Aux3T, ldAux3 );
    }

    daal::services::daal_free( R );
    daal::services::daal_free( U );

    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        daal::threader_for( nBlocks, nBlocks, [=](int k)
        {
            interm *Aux3 ;
            BlockMicroTable<interm, readOnly, cpu> mtAux3 (r[2 + k]);
            mtAux3.getBlockOfRows( 0, n, &Aux3  ); /* Aux2  [nxb][n] */
            for ( size_t i = 0 ; i < n ; i++ )
            {
                for ( size_t j = 0 ; j < n; j++ )
                {
                    Aux3[i * n + j] = Aux3T[j * nxb + k * n + i];
                }
            }
            mtAux3.release();
        } );
    }

    if (svdPar->rightSingularMatrix == requiredInPackedForm)
    {
        BlockMicroTable<interm, writeOnly, cpu> mtV(r[1]);

        interm *V;
        mtV.getBlockOfRows( 0, n, &V ); /* V[n][n] */

        for ( i = 0 ; i < n ; i++ )
        {
            for ( j = 0 ; j < n; j++ )
            {
                V[i + j * n] = VT[i * n + j];
            }
        }

        mtV.release();
    }

    mtSigma.release();

    daal::services::daal_free( Aux2T );
    daal::services::daal_free( VT    );
    daal::services::daal_free( Aux3T );
}

template <typename interm, daal::algorithms::svd::Method method, CpuType cpu>
void SVDDistributedStep3Kernel<interm, method, cpu>::compute(const size_t na, const NumericTable *const *a,
                                                             const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par)
{
    size_t i, j;
    svd::Parameter defaultParams;
    const svd::Parameter *svdPar = &defaultParams;

    if ( par != 0 )
    {
        svdPar = static_cast<const svd::Parameter *>(par);
    }

    size_t nBlocks     = na / 2;
    size_t mCalculated = 0;

    for(size_t k = 0; k < nBlocks; k++)
    {
        BlockMicroTable<interm, readOnly, cpu> mtAux1i(a[k          ]);
        BlockMicroTable<interm, readOnly, cpu> mtAux3i(a[k + nBlocks]);
        BlockMicroTable<interm, writeOnly   , cpu> mtQi   (r[0]);

        size_t  n   = mtAux1i.getFullNumberOfColumns();
        size_t  m   = mtAux1i.getFullNumberOfRows();

        interm *Qi   ;
        interm *Aux1i;
        interm *Aux3i;

        mtAux1i.getBlockOfRows( 0,           m, &Aux1i ); /* Aux1i = Qin[m][n] */
        mtAux3i.getBlockOfRows( 0,           n, &Aux3i ); /* Aux3i = Ri [n][n] */
        mtQi   .getBlockOfRows( mCalculated, m, &Qi    ); /*         Qi [m][n] */

        interm *QiT    = (interm *)daal::services::daal_malloc(sizeof(interm) * n * m);
        interm *Aux1iT = (interm *)daal::services::daal_malloc(sizeof(interm) * n * m);
        interm *Aux3iT = (interm *)daal::services::daal_malloc(sizeof(interm) * n * n);

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

        MKL_INT ldAux1i = m;
        MKL_INT ldAux3i = n;
        MKL_INT ldQi    = m;

        compute_gemm_on_one_node<interm, cpu>( m, n, Aux1iT, ldAux1i, Aux3iT, ldAux3i, QiT, ldQi );

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
