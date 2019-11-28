/* file: svd_dense_default_impl.i */
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
//  Implementation of svds
//--
*/

#ifndef __SVD_KERNEL_IMPL_I__
#define __SVD_KERNEL_IMPL_I__

#include "service_math.h"
#include "service_defines.h"
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
template <typename algorithmFPType, CpuType cpu>
Status compute_svd_on_one_node(DAAL_INT m, DAAL_INT n, algorithmFPType * a, DAAL_INT lda, algorithmFPType * s, algorithmFPType * u, DAAL_INT ldu,
                               algorithmFPType * vt, DAAL_INT ldvt)
{
    /* Specifies options for computing all or part of the matrix U                                       */
    /* 'S': the first min(m, n) columns of U (the left singular vectors) are returned in the array u     */
    char jobu = 'S';

    /* Specifies options for computing all or part of the matrix V^T/V^H                                 */
    /* 'S': the first min(m,n) rows of V^T/V^H (the right singular vectors) are returned in the array vt */
    char jobvt = 'S';

    DAAL_INT workDim   = -1; /* =lwork in Intel(R) MKL API */
    DAAL_INT mklStatus = 0;  /* =info in Intel(R) MKL API  */

    /* buffers */
    algorithmFPType workQuery[2]; /* align? */

    /* buffer size query */
    Lapack<algorithmFPType, cpu>::xgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, workQuery, workDim, &mklStatus);
    workDim = workQuery[0];

    /* computation block */
    TArray<algorithmFPType, cpu> workPtr(workDim);
    algorithmFPType * work = workPtr.get();
    DAAL_CHECK(work, ErrorMemoryAllocationFailed);

    Lapack<algorithmFPType, cpu>::xgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, workDim, &mklStatus);

    if (mklStatus != 0)
    {
        if (mklStatus > 0)
        {
            return Status(ErrorSvdXBDSQRDidNotConverge);
        }
        else
        {
            return Status(ErrorSvdIthParamIllegalValue);
        }
    }

    return Status();
}

template <typename algorithmFPType, CpuType cpu>
Status compute_svd_on_one_node_seq(DAAL_INT m, DAAL_INT n, algorithmFPType * a, DAAL_INT lda, algorithmFPType * s, algorithmFPType * u, DAAL_INT ldu,
                                   algorithmFPType * vt, DAAL_INT ldvt)
{
    /* Specifies options for computing all or part of the matrix U                                       */
    /* 'S': the first min(m, n) columns of U (the left singular vectors) are returned in the array u     */
    char jobu = 'S';

    /* Specifies options for computing all or part of the matrix V^T/V^H                                 */
    /* 'S': the first min(m,n) rows of V^T/V^H (the right singular vectors) are returned in the array vt */
    char jobvt = 'S';

    DAAL_INT workDim   = -1; /* =lwork in Intel(R) MKL API */
    DAAL_INT mklStatus = 0;  /* =info in Intel(R) MKL API  */

    /* buffers */
    algorithmFPType workQuery[2]; /* align? */

    /* buffer size query */
    Lapack<algorithmFPType, cpu>::xxgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, workQuery, workDim, &mklStatus);
    workDim = workQuery[0];

    /* computation block */
    TArray<algorithmFPType, cpu> workPtr(workDim);
    algorithmFPType * work = workPtr.get();
    DAAL_CHECK(work, ErrorMemoryAllocationFailed);

    Lapack<algorithmFPType, cpu>::xxgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, workDim, &mklStatus);

    if (mklStatus != 0)
    {
        if (mklStatus > 0)
        {
            return Status(ErrorSvdXBDSQRDidNotConverge);
        }
        else
        {
            return Status(ErrorSvdIthParamIllegalValue);
        }
    }

    return Status();
}

/*
    assumed n < m
  Input:
    a_q at input : a[m][lda_q] -> A (m x n)
  Output:
    a_q at output: q[m][lda_q] -> Qn(m x n) = n leading columns of orthogonal Q
    r   at output: r[n][ldr  ] -> R (n x n) = upper triangular matrix written in lower triangular (fortran is evil)

*/
template <typename algorithmFPType, CpuType cpu>
Status compute_QR_on_one_node(DAAL_INT m, DAAL_INT n, algorithmFPType * a_q, DAAL_INT lda_q, algorithmFPType * r, DAAL_INT ldr)
{
    // .. Local arrays
    // .. Memory allocation block
    TArray<algorithmFPType, cpu> tauPtr(n);
    algorithmFPType * tau = tauPtr.get();
    DAAL_CHECK(tau, ErrorMemoryAllocationFailed);

    // buffers
    algorithmFPType workQuery[2]; /* align? */

    DAAL_INT mklStatus = 0;
    DAAL_INT workDim   = -1;

    // buffer size query
    Lapack<algorithmFPType, cpu>::xgeqrf(m, n, a_q, lda_q, tau, workQuery, workDim, &mklStatus);
    workDim = workQuery[0];

    // a bug in Intel(R) MKL with XORGQR workDim query, to be fixed
    // XORGQR( m, n, n, a, lda, tau, work, &workDim, &mklStatus );
    // workDim = max(workDim, work[0]);

    // allocate buffer
    TArray<algorithmFPType, cpu> workPtr(workDim);
    algorithmFPType * work = workPtr.get();
    DAAL_CHECK(work, ErrorMemoryAllocationFailed);

    // Compute QR decomposition
    Lapack<algorithmFPType, cpu>::xgeqrf(m, n, a_q, lda_q, tau, work, workDim, &mklStatus);

    if (mklStatus != 0)
    {
        return Status(ErrorSvdIthParamIllegalValue);
    }

    // Get R of the QR factorization formed by xgeqrf
    DAAL_INT i, j;
    for (i = 1; i <= n; i++)
    {
        for (j = 0; j < i; j++)
        {
            r[(i - 1) * ldr + j] = a_q[(i - 1) * lda_q + j];
        }
    }

    // Get Q of the QR factorization formed by xgeqrf
    Lapack<algorithmFPType, cpu>::xorgqr(m, n, n, a_q, lda_q, tau, work, workDim, &mklStatus);

    if (mklStatus != 0)
    {
        return Status(ErrorSvdIthParamIllegalValue);
    }

    return Status();
}

template <typename algorithmFPType, CpuType cpu>
Status compute_QR_on_one_node_seq(DAAL_INT m, DAAL_INT n, algorithmFPType * a_q, DAAL_INT lda_q, algorithmFPType * r, DAAL_INT ldr)
{
    // .. Local arrays
    // .. Memory allocation block
    TArray<algorithmFPType, cpu> tauPtr(n);
    algorithmFPType * tau = tauPtr.get();
    DAAL_CHECK(tau, ErrorMemoryAllocationFailed);

    // buffers
    algorithmFPType workQuery[2]; /* align? */

    DAAL_INT mklStatus = 0;
    DAAL_INT workDim   = -1;

    // buffer size query
    Lapack<algorithmFPType, cpu>::xxgeqrf(m, n, a_q, lda_q, tau, workQuery, workDim, &mklStatus);
    workDim = workQuery[0];

    // a bug in Intel(R) MKL with XORGQR workDim query, to be fixed
    // XORGQR( m, n, n, a, lda, tau, work, &workDim, &mklStatus );
    // workDim = max(workDim, work[0]);

    // allocate buffer
    TArray<algorithmFPType, cpu> workPtr(workDim);
    algorithmFPType * work = workPtr.get();
    DAAL_CHECK(work, ErrorMemoryAllocationFailed);

    // Compute QR decomposition
    Lapack<algorithmFPType, cpu>::xxgeqrf(m, n, a_q, lda_q, tau, work, workDim, &mklStatus);

    if (mklStatus != 0)
    {
        return Status(ErrorSvdIthParamIllegalValue);
    }

    // Get R of the QR factorization formed by xgeqrf
    DAAL_INT i, j;
    for (i = 1; i <= n; i++)
    {
        for (j = 0; j < i; j++)
        {
            r[(i - 1) * ldr + j] = a_q[(i - 1) * lda_q + j];
        }
    }

    // Get Q of the QR factorization formed by xgeqrf
    Lapack<algorithmFPType, cpu>::xxorgqr(m, n, n, a_q, lda_q, tau, work, workDim, &mklStatus);

    if (mklStatus != 0)
    {
        return Status(ErrorSvdIthParamIllegalValue);
    }

    return Status();
}

template <typename algorithmFPType, CpuType cpu>
Status compute_gemm_on_one_node(DAAL_INT m, DAAL_INT n, algorithmFPType * a, DAAL_INT lda, algorithmFPType * b, DAAL_INT ldb, algorithmFPType * c,
                                DAAL_INT ldc)
{
    algorithmFPType one  = algorithmFPType(1.0);
    algorithmFPType zero = algorithmFPType(0.0);

    char notrans = 'N';

    Blas<algorithmFPType, cpu>::xgemm(&notrans, &notrans, &m, &n, &n, &one, a, &lda, b, &ldb, &zero, c, &ldc);

    return Status();
}

template <typename algorithmFPType, CpuType cpu>
Status compute_gemm_on_one_node_seq(DAAL_INT m, DAAL_INT n, algorithmFPType * a, DAAL_INT lda, algorithmFPType * b, DAAL_INT ldb, algorithmFPType * c,
                                    DAAL_INT ldc)
{
    algorithmFPType one  = algorithmFPType(1.0);
    algorithmFPType zero = algorithmFPType(0.0);

    char notrans = 'N';

    Blas<algorithmFPType, cpu>::xxgemm(&notrans, &notrans, &m, &n, &n, &one, a, &lda, b, &ldb, &zero, c, &ldc);

    return Status();
}

} // namespace internal
} // namespace svd
} // namespace algorithms
} // namespace daal

#endif
