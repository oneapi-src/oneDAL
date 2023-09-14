/* file: pivoted_qr_impl.i */
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

#ifndef __PIVOTED_QR_IMPL_I__
#define __PIVOTED_QR_IMPL_I__

#include "src/externals/service_lapack.h"
#include "src/externals/service_memory.h"
#include "src/externals/service_math.h"
#include "src/services/service_defines.h"
#include "src/data_management/service_numeric_table.h"

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace pivoted_qr
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
ServiceStatus compute_pivoted_QR_on_one_node(const DAAL_INT m, const DAAL_INT n, algorithmFPType * a_q, const DAAL_INT lda_q, algorithmFPType * r,
                                             const DAAL_INT ldr, DAAL_INT * jpvt)
{
    // .. Local arrays
    // .. Memory allocation block
    TArray<algorithmFPType, cpu> tauPtr(n);
    algorithmFPType * tau = tauPtr.get();
    if (!tau) return SERV_ERR_MALLOC;

    // buffers
    algorithmFPType workQuery[2]; /* align? */

    DAAL_INT mklStatus = 0;
    DAAL_INT workDim   = -1;

    // buffer size query
    LapackInst<algorithmFPType, cpu>::xgeqp3(m, n, a_q, lda_q, jpvt, tau, workQuery, workDim, &mklStatus);
    workDim = workQuery[0];

    // allocate buffer
    TArray<algorithmFPType, cpu> workPtr(workDim);
    algorithmFPType * work = workPtr.get();
    if (!work) return SERV_ERR_MALLOC;

    // Compute QR decomposition
    LapackInst<algorithmFPType, cpu>::xgeqp3(m, n, a_q, lda_q, jpvt, tau, work, workDim, &mklStatus);

    if (mklStatus != 0)
    {
        return SERV_ERR_MKL_QR_ITH_PARAM_ILLEGAL_VALUE;
    }

    // Get R of the QR factorization formed by xgeqp3
    for (DAAL_INT i = 1; i <= n; i++)
    {
        for (DAAL_INT j = 0; j < i; j++)
        {
            r[(i - 1) * ldr + j] = a_q[(i - 1) * lda_q + j];
        }
    }

    // Get Q of the QR factorization formed by xgeqp3
    LapackInst<algorithmFPType, cpu>::xorgqr(m, n, n, a_q, lda_q, tau, work, workDim, &mklStatus);

    if (mklStatus != 0)
    {
        return SERV_ERR_MKL_QR_ITH_PARAM_ILLEGAL_VALUE;
    }

    return SERV_ERR_OK;
}

/**
 *  \brief Kernel for Pivoted QR calculation
 */
template <daal::algorithms::pivoted_qr::Method method, typename algorithmFPType, CpuType cpu>
services::Status PivotedQRKernel<method, algorithmFPType, cpu>::compute(const NumericTable & dataTable, NumericTable & QTable, NumericTable & RTable,
                                                                        NumericTable & PTable, NumericTable * permutedColumns)
{
    const size_t n = dataTable.getNumberOfColumns();
    const size_t m = dataTable.getNumberOfRows();

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(DAAL_INT, n, sizeof(DAAL_INT));
    TArray<DAAL_INT, cpu> jpvtPtr(n);
    DAAL_INT * jpvt = jpvtPtr.get();
    DAAL_CHECK_MALLOC(jpvt);
    if (permutedColumns)
    {
        ReadRows<int, cpu> permutedColumnsBlock(permutedColumns, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(permutedColumnsBlock);
        const int * jpvtFromParameter = permutedColumnsBlock.get();
        for (size_t i = 0; i < n; i++)
        {
            jpvt[i] = jpvtFromParameter[i];
        }
    }
    else
    {
        for (size_t i = 0; i < n; i++)
        {
            jpvt[i] = 0;
        }
    }

    DAAL_INT ldAi = m;
    DAAL_INT ldRi = n;

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, m);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * m, sizeof(algorithmFPType));
    TArray<algorithmFPType, cpu> QiTPtr(n * m);
    algorithmFPType * QiT = QiTPtr.get();
    DAAL_CHECK_MALLOC(QiT);

    {
        ReadRows<algorithmFPType, cpu> blockAi(const_cast<NumericTable &>(dataTable), 0, m);
        DAAL_CHECK_BLOCK_STATUS(blockAi);
        const algorithmFPType * Ai = blockAi.get();
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < m; j++)
            {
                QiT[i * m + j] = Ai[i + j * n];
            }
        }
    }

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, n);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * n, sizeof(algorithmFPType));
    TArray<algorithmFPType, cpu> RiTPtr(n * n);
    algorithmFPType * RiT = RiTPtr.get();
    DAAL_CHECK_MALLOC(RiT);

    ServiceStatus status = compute_pivoted_QR_on_one_node<algorithmFPType, cpu>(m, n, QiT, ldAi, RiT, ldRi, jpvt);
    if (status != SERV_ERR_OK)
    {
        if (status == SERV_ERR_MALLOC)
            return Status(ErrorMemoryAllocationFailed);
        else
            return Status(ErrorPivotedQRInternal);
    }

    {
        WriteOnlyRows<algorithmFPType, cpu> blockQi(QTable, 0, m);
        DAAL_CHECK_BLOCK_STATUS(blockQi);
        algorithmFPType * Qi = blockQi.get();
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < m; j++)
            {
                Qi[i + j * n] = QiT[i * m + j];
            }
        }
    }

    {
        WriteOnlyRows<algorithmFPType, cpu> blockRi(RTable, 0, n);
        DAAL_CHECK_BLOCK_STATUS(blockRi);
        algorithmFPType * Ri = blockRi.get();
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j <= i; j++)
            {
                Ri[i + j * n] = RiT[i * n + j];
            }
            for (size_t j = i + 1; j < n; j++)
            {
                Ri[i + j * n] = 0.0;
            }
        }
    }

    {
        WriteOnlyRows<algorithmFPType, cpu> blockPi(PTable, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(blockPi);
        algorithmFPType * Pi = blockPi.get();
        for (size_t i = 0; i < n; i++)
        {
            Pi[i] = jpvt[i];
        }
    }

    return Status();
}

} //namespace internal

} //namespace pivoted_qr

} //namespace algorithms

} //namespace daal

#endif
