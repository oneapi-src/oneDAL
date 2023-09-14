/* file: qr_dense_default_impl.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#include "src/externals/service_blas.h"
#include "src/externals/service_lapack.h"
#include "src/externals/service_memory.h"
#include "src/externals/service_math.h"
#include "src/services/service_defines.h"
#include "src/data_management/service_micro_table.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"

#include "src/threading/threading.h"

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
    typedef LapackInst<algorithmFPType, cpu> lapack;

    // .. Local arrays
    // .. Memory allocation block
    TArray<algorithmFPType, cpu> tau(n);
    DAAL_CHECK(tau.get(), services::ErrorMemoryAllocationFailed);

    // buffers
    algorithmFPType workQuery[2];

    DAAL_INT mklStatus = 0;
    DAAL_INT workDim   = -1;

    // buffer size query
    lapack::xgeqrf(m, n, a_q, lda_q, tau.get(), workQuery, workDim, &mklStatus);

    workDim = workQuery[0];

    // allocate buffer
    TArray<algorithmFPType, cpu> work(workDim);
    DAAL_CHECK(work.get(), services::ErrorMemoryAllocationFailed);

    // Compute QR decomposition
    lapack::xgeqrf(m, n, a_q, lda_q, tau.get(), work.get(), workDim, &mklStatus);

    if (mklStatus != 0) return services::ErrorQRInternal;

    // Get R of the QR factorization formed by xgeqrf
    for (auto i = 1; i <= n; i++)
    {
        for (auto j = 0; j < i; j++)
        {
            r[(i - 1) * ldr + j] = a_q[(i - 1) * lda_q + j];
        }
        for (auto j = i; j < n; j++)
        {
            r[(i - 1) * ldr + j] = (algorithmFPType)0.0;
        }
    }

    // Get Q of the QR factorization formed by xgeqrf
    lapack::xorgqr(m, n, n, a_q, lda_q, tau.get(), work.get(), workDim, &mklStatus);

    DAAL_CHECK(!mklStatus, services::ErrorQRInternal);
    return Status();
}

template <typename algorithmFPType, CpuType cpu>
Status compute_QR_on_one_node_seq(DAAL_INT m, DAAL_INT n, algorithmFPType * a_q, DAAL_INT lda_q, algorithmFPType * r, DAAL_INT ldr)
{
    typedef LapackInst<algorithmFPType, cpu> lapack;

    // .. Local arrays
    // .. Memory allocation block
    TArrayScalable<algorithmFPType, cpu> tau(n);

    // buffers
    algorithmFPType workQuery[2];

    DAAL_INT mklStatus = 0;
    DAAL_INT workDim   = -1;

    // buffer size query
    lapack::xxgeqrf(m, n, a_q, lda_q, tau.get(), workQuery, workDim, &mklStatus);

    workDim = workQuery[0];

    // allocate buffer
    TArrayScalable<algorithmFPType, cpu> work(workDim);

    // Compute QR decomposition
    lapack::xxgeqrf(m, n, a_q, lda_q, tau.get(), work.get(), workDim, &mklStatus);

    if (mklStatus) return services::ErrorQRInternal;

    // Get R of the QR factorization formed by xgeqrf
    for (auto i = 1; i <= n; i++)
    {
        for (auto j = 0; j < i; j++)
        {
            r[(i - 1) * ldr + j] = a_q[(i - 1) * lda_q + j];
        }
        for (auto j = i; j < n; j++)
        {
            r[(i - 1) * ldr + j] = (algorithmFPType)0.0;
        }
    }

    // Get Q of the QR factorization formed by xgeqrf
    lapack::xxorgqr(m, n, n, a_q, lda_q, tau.get(), work.get(), workDim, &mklStatus);

    DAAL_CHECK(!mklStatus, services::ErrorQRInternal);
    return Status();
}

template <typename algorithmFPType, CpuType cpu>
void compute_gemm_on_one_node(DAAL_INT m, DAAL_INT n, algorithmFPType * a, DAAL_INT lda, algorithmFPType * b, DAAL_INT ldb, algorithmFPType * c,
                              DAAL_INT ldc)
{
    algorithmFPType one  = algorithmFPType(1.0);
    algorithmFPType zero = algorithmFPType(0.0);

    char notrans = 'N';

    BlasInst<algorithmFPType, cpu>::xgemm(&notrans, &notrans, &m, &n, &n, &one, a, &lda, b, &ldb, &zero, c, &ldc);
}

template <typename algorithmFPType, CpuType cpu>
void compute_gemm_on_one_node_seq(DAAL_INT m, DAAL_INT n, algorithmFPType * a, DAAL_INT lda, algorithmFPType * b, DAAL_INT ldb, algorithmFPType * c,
                                  DAAL_INT ldc)
{
    algorithmFPType one  = algorithmFPType(1.0);
    algorithmFPType zero = algorithmFPType(0.0);

    char notrans = 'N';

    BlasInst<algorithmFPType, cpu>::xxgemm(&notrans, &notrans, &m, &n, &n, &one, a, &lda, b, &ldb, &zero, c, &ldc);
}

} // namespace internal
} // namespace qr
} // namespace algorithms
} // namespace daal

#endif
