/* file: pivoted_qr_impl.i */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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

#ifndef __PIVOTED_QR_IMPL_I__
#define __PIVOTED_QR_IMPL_I__

#include "service_lapack.h"
#include "service_memory.h"
#include "service_math.h"
#include "service_defines.h"
#include "service_numeric_table.h"

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
ServiceStatus compute_pivoted_QR_on_one_node( const DAAL_INT m, const DAAL_INT n, algorithmFPType *a_q, const DAAL_INT lda_q, algorithmFPType *r,
        const DAAL_INT ldr, DAAL_INT *jpvt)
{
    // .. Local arrays
    // .. Memory allocation block
    TArray<algorithmFPType, cpu> tauPtr(n);
    algorithmFPType *tau = tauPtr.get();
    if(!tau) return SERV_ERR_MALLOC;

    // buffers
    algorithmFPType workQuery[2]; /* align? */

    DAAL_INT mklStatus =  0;
    DAAL_INT workDim   = -1;

    // buffer size query
    Lapack<algorithmFPType, cpu>::xgeqp3( m, n, a_q, lda_q, jpvt, tau, workQuery, workDim, &mklStatus );
    workDim = workQuery[0];

    // allocate buffer
    TArray<algorithmFPType, cpu> workPtr(workDim);
    algorithmFPType *work = workPtr.get();
    if(!work) return SERV_ERR_MALLOC;

    // Compute QR decomposition
    Lapack<algorithmFPType, cpu>::xgeqp3( m, n, a_q, lda_q, jpvt, tau, work, workDim, &mklStatus );

    if ( mklStatus != 0 )
    {
        return SERV_ERR_MKL_QR_ITH_PARAM_ILLEGAL_VALUE;
    }

    // Get R of the QR factorization formed by xgeqp3
    for ( int i = 1; i <= n; i++ )
    {
        for ( int j = 0; j < i; j++ )
        {
            r[(i - 1)*ldr + j] = a_q[(i - 1) * lda_q + j];
        }
    }

    // Get Q of the QR factorization formed by xgeqp3
    Lapack<algorithmFPType, cpu>::xorgqr( m, n, n, a_q, lda_q, tau, work, workDim, &mklStatus );

    if ( mklStatus != 0 )
    {
        return SERV_ERR_MKL_QR_ITH_PARAM_ILLEGAL_VALUE;
    }

    return SERV_ERR_OK;
}

/**
 *  \brief Kernel for Pivoted QR calculation
 */
template <daal::algorithms::pivoted_qr::Method method, typename algorithmFPType, CpuType cpu>
services::Status PivotedQRKernel<method, algorithmFPType, cpu>::compute(
    const NumericTable &dataTable, NumericTable &QTable, NumericTable &RTable, NumericTable &PTable, NumericTable *permutedColumns)
{
    const size_t n = dataTable.getNumberOfColumns();
    const size_t m = dataTable.getNumberOfRows();

    TArray<DAAL_INT, cpu> jpvtPtr(n);
    DAAL_INT *jpvt = jpvtPtr.get();
    DAAL_CHECK_MALLOC(jpvt);
    if ( permutedColumns )
    {
        ReadRows<int, cpu> permutedColumnsBlock(permutedColumns, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(permutedColumnsBlock);
        const int *jpvtFromParameter = permutedColumnsBlock.get();
        for(size_t i = 0; i < n; i++)
        {
            jpvt[i] = jpvtFromParameter[i];
        }
    }
    else
    {
        for(size_t i = 0; i < n; i++)
        {
            jpvt[i] = 0;
        }
    }

    DAAL_INT ldAi = m;
    DAAL_INT ldRi = n;

    TArray<algorithmFPType, cpu> QiTPtr(n * m);
    algorithmFPType *QiT = QiTPtr.get();
    DAAL_CHECK_MALLOC(QiT);

    {
        ReadRows<algorithmFPType, cpu> blockAi(const_cast<NumericTable &>(dataTable), 0, m);
        DAAL_CHECK_BLOCK_STATUS(blockAi);
        const algorithmFPType *Ai = blockAi.get();
        for ( int i = 0 ; i < n ; i++ )
        {
            for ( int j = 0 ; j < m; j++ )
            {
                QiT[i * m + j] = Ai[i + j * n];
            }
        }
    }

    TArray<algorithmFPType, cpu> RiTPtr(n * n);
    algorithmFPType *RiT = RiTPtr.get();
    DAAL_CHECK_MALLOC(RiT);

    ServiceStatus status = compute_pivoted_QR_on_one_node<algorithmFPType, cpu>( m, n, QiT, ldAi, RiT, ldRi, jpvt);
    if(status != SERV_ERR_OK)
    {
        if(status == SERV_ERR_MALLOC) return Status(ErrorMemoryAllocationFailed);
        else return Status(ErrorPivotedQRInternal);
    }

    {
        WriteOnlyRows<algorithmFPType, cpu> blockQi(QTable, 0, m);
        DAAL_CHECK_BLOCK_STATUS(blockQi);
        algorithmFPType *Qi = blockQi.get();
        for ( int i = 0 ; i < n ; i++ )
        {
            for ( int j = 0 ; j < m; j++ )
            {
                Qi[i + j * n] = QiT[i * m + j];
            }
        }
    }

    {
        WriteOnlyRows<algorithmFPType, cpu> blockRi(RTable, 0, n);
        DAAL_CHECK_BLOCK_STATUS(blockRi);
        algorithmFPType *Ri = blockRi.get();
        for ( int i = 0 ; i < n ; i++ )
        {
            for ( int j = 0 ; j <= i; j++ )
            {
                Ri[i + j * n] = RiT[i * n + j];
            }
            for ( int j = i + 1 ; j < n; j++ )
            {
                Ri[i + j * n] = 0.0;
            }
        }
    }

    {
        WriteOnlyRows<algorithmFPType, cpu> blockPi(PTable, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(blockPi);
        algorithmFPType *Pi = blockPi.get();
        for( int i = 0; i < n ; i++)
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
