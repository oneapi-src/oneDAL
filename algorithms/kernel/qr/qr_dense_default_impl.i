/* file: qr_dense_default_impl.i */
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

#ifndef __QR_UTILS_IMPL_I__
#define __QR_UTILS_IMPL_I__

#include "service_blas.h"
#include "service_lapack.h"
#include "service_memory.h"
#include "service_math.h"
#include "service_defines.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

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

/*
    assumed n < m
  Input:
    a_q at input : a[m][lda_q] -> A (m x n)
  Output:
    a_q at output: q[m][lda_q] -> Qn(m x n) = n leading columns of orthogonal Q
    r   at output: r[n][ldr  ] -> R (n x n) = upper triangular matrix written in lower triangular (fortran is evil)

*/
template <typename interm, CpuType cpu>
ServiceStatus compute_QR_on_one_node( MKL_INT m, MKL_INT n, interm *a_q, MKL_INT lda_q, interm *r,
                                      MKL_INT ldr )
{
    typedef Lapack<interm, cpu> lapack;

    // .. Local arrays
    // .. Memory allocation block
    interm *tau;
    tau = (interm *)daal::services::daal_malloc( sizeof(interm) * n );

    // buffers
    interm  workQuery[2];

    MKL_INT mklStatus =  0;
    MKL_INT workDim   = -1;

    // buffer size query
    lapack::xgeqrf( m, n, a_q, lda_q, tau, workQuery, workDim, &mklStatus );

    workDim = workQuery[0];

    // allocate buffer
    interm *work;
    work = (interm *)daal::services::daal_malloc( sizeof(interm) * workDim );

    // Compute QR decomposition
    lapack::xgeqrf( m, n, a_q, lda_q, tau, work, workDim, &mklStatus );

    if ( mklStatus != 0 )
    {
        return SERV_ERR_MKL_QR_ITH_PARAM_ILLEGAL_VALUE;
    }

    // Get R of the QR factorization formed by xgeqrf
    MKL_INT i, j;
    for ( i = 1; i <= n; i++ )
    {
        for ( j = 0; j < i; j++ )
        {
            r[(i - 1)*ldr + j] = a_q[(i - 1) * lda_q + j];
        }
    }

    // Get Q of the QR factorization formed by xgeqrf
    lapack::xorgqr( m, n, n, a_q, lda_q, tau, work, workDim, &mklStatus );

    if ( mklStatus != 0 )
    {
        return SERV_ERR_MKL_QR_ITH_PARAM_ILLEGAL_VALUE;
    }

    // Memory deallocation block
    daal::services::daal_free(tau);
    daal::services::daal_free(work);

    return SERV_ERR_OK;
}


template <typename interm, CpuType cpu>
ServiceStatus compute_QR_on_one_node_seq( MKL_INT m, MKL_INT n, interm *a_q, MKL_INT lda_q, interm *r,
                                      MKL_INT ldr )
{
    typedef Lapack<interm, cpu> lapack;

    // .. Local arrays
    // .. Memory allocation block
    interm *tau;
    tau = service_scalable_malloc<interm,cpu>( n );

    // buffers
    interm  workQuery[2];

    MKL_INT mklStatus =  0;
    MKL_INT workDim   = -1;

    // buffer size query
    lapack::xxgeqrf( m, n, a_q, lda_q, tau, workQuery, workDim, &mklStatus );

    workDim = workQuery[0];

    // allocate buffer
    interm *work;
    work = service_scalable_malloc<interm,cpu>( workDim );

    // Compute QR decomposition
    lapack::xxgeqrf( m, n, a_q, lda_q, tau, work, workDim, &mklStatus );

    if ( mklStatus != 0 )
    {
        return SERV_ERR_MKL_QR_ITH_PARAM_ILLEGAL_VALUE;
    }

    // Get R of the QR factorization formed by xgeqrf
    MKL_INT i, j;
    for ( i = 1; i <= n; i++ )
    {
        for ( j = 0; j < i; j++ )
        {
            r[(i - 1)*ldr + j] = a_q[(i - 1) * lda_q + j];
        }
    }

    // Get Q of the QR factorization formed by xgeqrf
    lapack::xxorgqr( m, n, n, a_q, lda_q, tau, work, workDim, &mklStatus );

    if ( mklStatus != 0 )
    {
        return SERV_ERR_MKL_QR_ITH_PARAM_ILLEGAL_VALUE;
    }

    // Memory deallocation block
    service_scalable_free<interm,cpu>(tau);
    service_scalable_free<interm,cpu>(work);

    return SERV_ERR_OK;
}

template <typename interm, CpuType cpu>
ServiceStatus compute_gemm_on_one_node( MKL_INT m, MKL_INT n, interm *a, MKL_INT lda, interm *b, MKL_INT ldb,
                                        interm *c, MKL_INT ldc)
{
    interm one  = interm(1.0);
    interm zero = interm(0.0);

    char notrans = 'N';

    Blas<interm, cpu>::xgemm( &notrans, &notrans, &m, &n, &n, &one, a, &lda, b, &ldb, &zero, c, &ldc);

    return SERV_ERR_OK;
}


template <typename interm, CpuType cpu>
ServiceStatus compute_gemm_on_one_node_seq( MKL_INT m, MKL_INT n, interm *a, MKL_INT lda, interm *b, MKL_INT ldb,
                                        interm *c, MKL_INT ldc)
{
    interm one  = interm(1.0);
    interm zero = interm(0.0);

    char notrans = 'N';

    Blas<interm, cpu>::xxgemm( &notrans, &notrans, &m, &n, &n, &one, a, &lda, b, &ldb, &zero, c, &ldc);

    return SERV_ERR_OK;
}


template <typename interm, CpuType cpu>
ServiceStatus compute_gemm_on_one_node( MKL_INT m, MKL_INT n, interm *a, MKL_INT lda, interm *b, MKL_INT ldb)
{
    MKL_INT ldc = m;

    interm *c = (interm *)daal::services::daal_malloc( sizeof(interm) * n * ldc );

    compute_gemm_on_one_node<interm, cpu>( m, n, a, lda, b, ldb, c, ldc);

    MKL_INT i, j;
    for ( i = 0; i < n; i++ )
    {
        for ( j = 0; j < m; j++ )
        {
            a[i * lda + j] = c[i * ldc + j];
        }
    }

    daal::services::daal_free(c);

    return SERV_ERR_OK;
}


template <typename interm, CpuType cpu>
ServiceStatus compute_gemm_on_one_node_seq( MKL_INT m, MKL_INT n, interm *a, MKL_INT lda, interm *b, MKL_INT ldb)
{
    MKL_INT ldc = m;

    interm *c = service_scalable_malloc<interm,cpu>( n * ldc );

    compute_gemm_on_one_node_seq<interm, cpu>( m, n, a, lda, b, ldb, c, ldc);

    MKL_INT i, j;
    for ( i = 0; i < n; i++ )
    {
        for ( j = 0; j < m; j++ )
        {
            a[i * lda + j] = c[i * ldc + j];
        }
    }

    service_scalable_free<interm,cpu>(c);

    return SERV_ERR_OK;
}

} // namespace daal::internal
}
}
} // namespace daal

#endif
