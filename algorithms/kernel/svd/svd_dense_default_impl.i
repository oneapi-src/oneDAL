/* file: svd_dense_default_impl.i */
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

#ifndef __SVD_KERNEL_IMPL_I__
#define __SVD_KERNEL_IMPL_I__

#include "service_memory.h"
#include "service_math.h"
#include "service_defines.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"
#include "service_blas.h"
#include "service_lapack.h"

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

template <typename interm, CpuType cpu>
ServiceStatus compute_svd_on_one_node( MKL_INT m, MKL_INT n, interm *a, MKL_INT lda,
                                       interm *s, interm *u, MKL_INT ldu, interm *vt, MKL_INT ldvt )
{
    /* Specifies options for computing all or part of the matrix U                                       */
    /* 'S': the first min(m, n) columns of U (the left singular vectors) are returned in the array u     */
    char jobu = 'S';

    /* Specifies options for computing all or part of the matrix V^T/V^H                                 */
    /* 'S': the first min(m,n) rows of V^T/V^H (the right singular vectors) are returned in the array vt */
    char jobvt = 'S';

    MKL_INT workDim   = -1; /* =lwork in MKL API */
    MKL_INT mklStatus =  0; /* =info in MKL API  */

    /* buffers */
    interm  workQuery[2]; /* align? */
    interm *work;

    /* buffer size query */
    Lapack<interm, cpu>::xgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, workQuery, workDim, &mklStatus);
    workDim = workQuery[0];

    /* computation block */
    work = (interm *)daal::services::daal_malloc( sizeof(interm) * workDim );
    Lapack<interm, cpu>::xgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, workDim, &mklStatus);
    daal::services::daal_free(work);

    if ( mklStatus != 0 )
    {
        if ( mklStatus > 0 )
        {
            return SERV_ERR_MKL_SVD_XBDSQR_DID_NOT_CONVERGE;
        }
        else
        {
            return SERV_ERR_MKL_SVD_ITH_PARAM_ILLEGAL_VALUE;
        }
    }

    return SERV_ERR_OK;
}


template <typename interm, CpuType cpu>
ServiceStatus compute_svd_on_one_node_seq( MKL_INT m, MKL_INT n, interm *a, MKL_INT lda,
                                       interm *s, interm *u, MKL_INT ldu, interm *vt, MKL_INT ldvt )
{
    /* Specifies options for computing all or part of the matrix U                                       */
    /* 'S': the first min(m, n) columns of U (the left singular vectors) are returned in the array u     */
    char jobu = 'S';

    /* Specifies options for computing all or part of the matrix V^T/V^H                                 */
    /* 'S': the first min(m,n) rows of V^T/V^H (the right singular vectors) are returned in the array vt */
    char jobvt = 'S';

    MKL_INT workDim   = -1; /* =lwork in MKL API */
    MKL_INT mklStatus =  0; /* =info in MKL API  */

    /* buffers */
    interm  workQuery[2]; /* align? */
    interm *work;

    /* buffer size query */
    Lapack<interm, cpu>::xxgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, workQuery, workDim, &mklStatus);
    workDim = workQuery[0];

    /* computation block */
    work = (interm *)daal::services::daal_malloc( sizeof(interm) * workDim );
    Lapack<interm, cpu>::xxgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, workDim, &mklStatus);
    daal::services::daal_free(work);

    if ( mklStatus != 0 )
    {
        if ( mklStatus > 0 )
        {
            return SERV_ERR_MKL_SVD_XBDSQR_DID_NOT_CONVERGE;
        }
        else
        {
            return SERV_ERR_MKL_SVD_ITH_PARAM_ILLEGAL_VALUE;
        }
    }

    return SERV_ERR_OK;
}

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
    // .. Local arrays
    // .. Memory allocation block
    interm *tau;
    tau = (interm *)daal::services::daal_malloc( sizeof(interm) * n );

    // buffers
    interm  workQuery[2]; /* align? */

    MKL_INT mklStatus =  0;
    MKL_INT workDim   = -1;

    // buffer size query
    Lapack<interm, cpu>::xgeqrf( m, n, a_q, lda_q, tau, workQuery, workDim, &mklStatus );
    workDim = workQuery[0];

    // a bug in MKL with XORGQR workDim query, to be fixed
    // XORGQR( m, n, n, a, lda, tau, work, &workDim, &mklStatus );
    // workDim = max(workDim, work[0]);

    // allocate buffer
    interm *work;
    work = (interm *)daal::services::daal_malloc( sizeof(interm) * workDim );

    // Compute QR decomposition
    Lapack<interm, cpu>::xgeqrf( m, n, a_q, lda_q, tau, work, workDim, &mklStatus );

    if ( mklStatus != 0 )
    {
        return SERV_ERR_MKL_SVD_ITH_PARAM_ILLEGAL_VALUE;
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
    Lapack<interm, cpu>::xorgqr( m, n, n, a_q, lda_q, tau, work, workDim, &mklStatus );

    if ( mklStatus != 0 )
    {
        return SERV_ERR_MKL_SVD_ITH_PARAM_ILLEGAL_VALUE;
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
    // .. Local arrays
    // .. Memory allocation block
    interm *tau;
    tau = (interm *)daal::services::daal_malloc( sizeof(interm) * n );

    // buffers
    interm  workQuery[2]; /* align? */

    MKL_INT mklStatus =  0;
    MKL_INT workDim   = -1;

    // buffer size query
    Lapack<interm, cpu>::xxgeqrf( m, n, a_q, lda_q, tau, workQuery, workDim, &mklStatus );
    workDim = workQuery[0];

    // a bug in MKL with XORGQR workDim query, to be fixed
    // XORGQR( m, n, n, a, lda, tau, work, &workDim, &mklStatus );
    // workDim = max(workDim, work[0]);

    // allocate buffer
    interm *work;
    work = (interm *)daal::services::daal_malloc( sizeof(interm) * workDim );

    // Compute QR decomposition
    Lapack<interm, cpu>::xxgeqrf( m, n, a_q, lda_q, tau, work, workDim, &mklStatus );

    if ( mklStatus != 0 )
    {
        return SERV_ERR_MKL_SVD_ITH_PARAM_ILLEGAL_VALUE;
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
    Lapack<interm, cpu>::xxorgqr( m, n, n, a_q, lda_q, tau, work, workDim, &mklStatus );

    if ( mklStatus != 0 )
    {
        return SERV_ERR_MKL_SVD_ITH_PARAM_ILLEGAL_VALUE;
    }

    // Memory deallocation block
    daal::services::daal_free(tau);
    daal::services::daal_free(work);

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

    interm *c = (interm *)daal::services::daal_malloc( sizeof(interm) * n * ldc );

    compute_gemm_on_one_node_seq<interm, cpu>( m, n, a, lda, b, ldb, c, ldc);

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

} // namespace daal::internal
}
}
} // namespace daal

#endif
