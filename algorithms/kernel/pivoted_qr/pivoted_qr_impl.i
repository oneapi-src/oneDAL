/* file: pivoted_qr_impl.i */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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

#include "service_lapack.h"
#include "service_memory.h"
#include "service_math.h"
#include "service_defines.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

using namespace daal::internal;
using namespace daal::services::internal;

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
ServiceStatus compute_pivoted_QR_on_one_node( DAAL_INT m, DAAL_INT n, algorithmFPType *a_q, DAAL_INT lda_q, algorithmFPType *r,
                                              DAAL_INT ldr, DAAL_INT *jpvt)
{
    // .. Local arrays
    // .. Memory allocation block
    algorithmFPType *tau;
    tau = (algorithmFPType *)daal::services::daal_malloc( sizeof(algorithmFPType) * n );

    // buffers
    algorithmFPType  workQuery[2]; /* align? */

    DAAL_INT mklStatus =  0;
    DAAL_INT workDim   = -1;

    // buffer size query
    Lapack<algorithmFPType, cpu>::xgeqp3( m, n, a_q, lda_q, jpvt, tau, workQuery, workDim, &mklStatus );
    workDim = workQuery[0];

    // allocate buffer
    algorithmFPType *work;
    work = (algorithmFPType *)daal::services::daal_malloc( sizeof(algorithmFPType) * workDim );

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

    // Memory deallocation block
    daal::services::daal_free(tau);
    daal::services::daal_free(work);

    return SERV_ERR_OK;
}

/**
 *  \brief Kernel for Pivoted QR calculation
 */
template <daal::algorithms::pivoted_qr::Method method, typename algorithmFPType, CpuType cpu>
services::Status PivotedQRKernel<method, algorithmFPType, cpu>::compute(const size_t na, const NumericTable *const *a,
                                                                        const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par)
{
    BlockMicroTable<algorithmFPType, readOnly, cpu> ntAi(const_cast<NumericTable *>(a[0]));

    algorithmFPType *Ai;

    size_t  n   = a[0]->getNumberOfColumns();
    size_t  m   = a[0]->getNumberOfRows();

    DAAL_INT *jpvt = (DAAL_INT *)daal::services::daal_malloc(n * sizeof(DAAL_INT));
    const pivoted_qr::Parameter *parameter = static_cast<const pivoted_qr::Parameter *>(par);
    if ( parameter->permutedColumns.get() != 0 )
    {
        int *jpvtFromParameter;
        BlockMicroTable<int, readOnly, cpu> permutedColumnsMicroTable(const_cast<NumericTable *>(parameter->permutedColumns.get()));
        size_t readRows = permutedColumnsMicroTable.getBlockOfRows(0, 1, &jpvtFromParameter);
        if(readRows != 1)
        {
            permutedColumnsMicroTable.release();
            ntAi.release();
            this->_errors->add(services::ErrorMemoryAllocationFailed);
            DAAL_RETURN_STATUS()
        }
        for(size_t i = 0; i < n; i++)
        {
            jpvt[i] = jpvtFromParameter[i];
        }
        permutedColumnsMicroTable.release();
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

    algorithmFPType *QiT = (algorithmFPType *)daal::services::daal_malloc(n * m * sizeof(algorithmFPType));
    algorithmFPType *RiT = (algorithmFPType *)daal::services::daal_malloc(n * n * sizeof(algorithmFPType));

    ntAi.getBlockOfRows( 0, m, &Ai); /*      Ai [m][n] */
    for ( int i = 0 ; i < n ; i++ )
    {
        for ( int j = 0 ; j < m; j++ )
        {
            QiT[i * m + j] = Ai[i + j * n];
        }
    }
    ntAi.release();

    compute_pivoted_QR_on_one_node<algorithmFPType, cpu>( m, n, QiT, ldAi, RiT, ldRi, jpvt);

    algorithmFPType *Qi;
    BlockMicroTable<algorithmFPType, writeOnly, cpu> ntQi(const_cast<NumericTable *>(r[0]));
    ntQi.getBlockOfRows( 0, m, &Qi); /* Qi = Qin[m][n] */
    for ( int i = 0 ; i < n ; i++ )
    {
        for ( int j = 0 ; j < m; j++ )
        {
            Qi[i + j * n] = QiT[i * m + j];
        }
    }
    ntQi.release();

    BlockMicroTable<algorithmFPType, writeOnly, cpu> ntRi(const_cast<NumericTable *>(r[1]));
    algorithmFPType *Ri;
    ntRi.getBlockOfRows( 0, n, &Ri); /* Ri = Ri [n][n] */
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
    ntRi.release();

    BlockMicroTable<algorithmFPType, writeOnly, cpu> ntPi(const_cast<NumericTable *>(r[2]));
    algorithmFPType *Pi;
    ntPi.getBlockOfRows( 0, 1, &Pi); /* Pi = Pi [m][1] */
    for( int i = 0; i < n ; i++)
    {
        Pi[i] = jpvt[i];
    }
    ntPi.release();

    daal::services::daal_free(QiT);
    daal::services::daal_free(RiT);
    daal::services::daal_free(jpvt);
    DAAL_RETURN_STATUS()
}

} //namespace internal

} //namespace pivoted_qr

} //namespace algorithms

} //namespace daal


#endif
