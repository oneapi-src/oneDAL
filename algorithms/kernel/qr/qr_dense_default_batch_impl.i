/* file: qr_dense_default_batch_impl.i */
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

#ifndef __QR_KERNEL_BATCH_IMPL_I__
#define __QR_KERNEL_BATCH_IMPL_I__

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
template <typename interm, daal::algorithms::qr::Method method, CpuType cpu>
void QRBatchKernel<interm, method, cpu>::compute(const size_t na, const NumericTable *const *a,
                                                 const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par)
{
    NumericTable *ntAi = const_cast<NumericTable *>(a[0]);

    size_t  n = ntAi->getNumberOfColumns();
    size_t  m = ntAi->getNumberOfRows();

    if(m >= 2*n)
    {
        QRBatchKernel<interm, method, cpu>::compute_thr( na, a, nr, r, par);
    }
    else
    {
        QRBatchKernel<interm, method, cpu>::compute_seq( na, a, nr, r, par);
    }
}


template <typename interm, daal::algorithms::qr::Method method, CpuType cpu>
void QRBatchKernel<interm, method, cpu>::compute_seq(const size_t na, const NumericTable *const *a,
                                                 const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par)
{
    size_t i, j;
    qr::Parameter defaultParams;

    NumericTable *ntAi = const_cast<NumericTable *>(a[0]);
    NumericTable *ntRi = const_cast<NumericTable *>(r[1]);

    size_t  n   = ntAi->getNumberOfColumns();
    size_t  m   = ntAi->getNumberOfRows();

    MKL_INT ldAi = m;
    MKL_INT ldRi = n;

    interm *QiT = (interm *)daal::services::daal_malloc(n * m * sizeof(interm));
    interm *RiT = (interm *)daal::services::daal_malloc(n * n * sizeof(interm));

    BlockDescriptor<interm> aiBlock;
    BlockDescriptor<interm> riBlock;

    ntAi->getBlockOfRows( 0, m, readOnly,  aiBlock); /*      Ai [m][n] */
    ntRi->getBlockOfRows( 0, n, writeOnly, riBlock); /* Ri = Ri [n][n] */

    interm *Ai = aiBlock.getBlockPtr();
    interm *Ri = riBlock.getBlockPtr();

    for ( i = 0 ; i < n ; i++ )
    {
        for ( j = 0 ; j < m; j++ )
        {
            QiT[i * m + j] = Ai[i + j * n];
        }
    }

    compute_QR_on_one_node<interm, cpu>( m, n, QiT, ldAi, RiT, ldRi );

    NumericTable *ntQi = const_cast<NumericTable *>(r[0]);
    BlockDescriptor<interm> qiBlock;
    ntQi->getBlockOfRows( 0, m, writeOnly, qiBlock ); /* Qi = Qin[m][n] */
    interm *Qi = qiBlock.getBlockPtr();
    for ( i = 0 ; i < n ; i++ )
    {
        for ( j = 0 ; j < m; j++ )
        {
            Qi[i + j * n] = QiT[i * m + j];
        }
    }
    ntQi->releaseBlockOfRows( qiBlock );

    for ( i = 0 ; i < n ; i++ )
    {
        for ( j = 0 ; j <= i; j++ )
        {
            Ri[i + j * n] = RiT[i * n + j];
        }
        for (     ; j < n; j++ )
        {
            Ri[i + j * n] = 0.0;
        }
    }

    ntAi->releaseBlockOfRows( aiBlock );
    ntRi->releaseBlockOfRows( riBlock );

    daal::services::daal_free(QiT);
    daal::services::daal_free(RiT);
}


/* Max number of blocks depending on arch */
#if( __CPUID__(DAAL_CPU) >= __avx512_mic__ )
    #define DEF_MAX_BLOCKS 256
#else
    #define DEF_MAX_BLOCKS 128
#endif


/*
    Algorithm for parallel QR computation:
    -------------------------------------
    A[m,n] input matrix to be factorized by output Q[m,n] and R[n,n]

    1st step:
    Split A[m,n] matrix to 'b' blocks -> a1[m1,n],a2[m2,]...ab[mb,n]
    Compute QR decomposition for each block in threads:
                               a1[m1,n] -> q1[m1,n] , r1[n,n] ... ab[mb,n] -> qb[mb,n] and rb[n,n]

    2nd step:
    Concatenate r1[n,n] , r2[n,n] ... rb[n,n] into one matrix B[n*b,n]
    Compute QR decomposition for B[n*b,n] -> P[n*b,n] , R[n,n]. R - resulted matrix

    3rd step: Split P[n*b,n] matrix to 'b' blocks -> p1[n,n],p2[n,n]...pb[n,n]
    Multiply by q1..qb matrices from 1st step using GEMM for each block in threads:
                               q1[m1,n] * p1[n,n] -> q'1[m1,n] ... qb[mb,n] * pb[n,n] -> q'b[mb,n]
    Concatenate q'1[m1,n]...q'b[mb,n] into one resulted Q[m,n]  matrix.

    Notice: before and after QR and GEMM computations matrices need to be transposed
*/
template <typename interm, daal::algorithms::qr::Method method, CpuType cpu>
void QRBatchKernel<interm, method, cpu>::compute_thr(const size_t na, const NumericTable *const *a,
                                                 const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par)
{
    qr::Parameter defaultParams;

    NumericTable *ntA_input  = const_cast<NumericTable *>(a[0]);
    NumericTable *ntQ_output = const_cast<NumericTable *>(r[0]);
    NumericTable *ntR_output = const_cast<NumericTable *>(r[1]);

    size_t  n = ntA_input->getNumberOfColumns();
    size_t  m = ntA_input->getNumberOfRows();

    MKL_INT ldA_input = m;
    MKL_INT ldR_output = n;

    size_t rows = m;
    size_t cols = n;

    BlockDescriptor<interm> bkA_input;
    BlockDescriptor<interm> bkQ_output;
    BlockDescriptor<interm> bkR_output;

    ntA_input->getBlockOfRows(  0, m, readOnly,  bkA_input);
    ntQ_output->getBlockOfRows( 0, m, writeOnly, bkQ_output);
    ntR_output->getBlockOfRows( 0, n, writeOnly, bkR_output);

    /* Getting real pointers to input and output arrays */
    interm *A_input  = bkA_input.getBlockPtr();
    interm *Q_output = bkQ_output.getBlockPtr();
    interm *R_output = bkR_output.getBlockPtr();

    /* Block size calculation (empirical) */
    int bshift = (rows <= 10000 )?11:12;
    int bsize  = ((rows*cols)>>bshift) & (~0xf);
    bsize = (bsize < 200)?200:bsize;

    /*
    Calculate sizes:
    blocks     = number of blocks,
    brows      = number of rows in blocks,
    brows_last = number of rows in last block,
    */
    size_t def_min_brows = rows/DEF_MAX_BLOCKS; // min block size
    size_t brows      = ( rows > bsize )? bsize:rows; /* brows cannot be less than rows */
    brows             = ( brows < cols )? cols:brows; /* brows cannot be less than cols */
    brows             = (brows < def_min_brows)?def_min_brows:brows; /* brows cannot be less than n/DEF_MAX_BLOCKS */
    size_t blocks     = rows / brows;

    size_t brows_last = brows + (rows - blocks * brows); /* last block is generally biggest */

    interm *R_buff   = service_calloc<interm,cpu>(blocks * n * n); /* zeroing */
    interm *RT_buff  = service_malloc<interm,cpu>(blocks * n * n);

    if((R_buff) && (RT_buff) )
    {
        /* Step1: calculate QR on local nodes */
        /* ================================== */

        daal::threader_for( blocks, blocks, [=,&A_input,&RT_buff](int k)
        {
            interm *A_block = A_input  + k * brows * cols;
            interm *Q_block = Q_output + k * brows * cols;

            /* Last block size brows_last (generally larger than other blocks) */
            size_t brows_local = (k==(blocks-1))?brows_last:brows;
            size_t cols_local  = cols;

            interm *QT_local = service_scalable_malloc<interm,cpu>(cols_local * brows_local);
            interm *RT_local = service_scalable_malloc<interm,cpu>(cols_local * cols_local);

            if( (QT_local) && (RT_local) )
            {
                /* Get transposed Q from A */
                for ( int i = 0 ; i < cols_local ; i++ ) {
                PRAGMA_IVDEP
                    for ( int j = 0 ; j < brows_local; j++ ) {
                        QT_local[i*brows_local+j] = A_block[i+j*cols_local]; } }

                /* Call QR on local nodes */
                compute_QR_on_one_node_seq<interm, cpu>( brows_local, cols_local, QT_local, brows_local, RT_local, cols_local );

                /* Transpose Q */
                for ( int i = 0 ; i < cols_local ; i++ ) {
                PRAGMA_IVDEP
                    for ( int j = 0 ; j < brows_local; j++ ) {
                        Q_block[i+j*cols_local] = QT_local[i*brows_local+j]; } }

                /* Transpose R and zero lower values */
                for ( int i = 0 ; i < cols_local ; i++ ) {
                    int j;
                    PRAGMA_IVDEP
                           for ( j = 0 ; j <= i; j++ ) {
                               RT_buff[k*cols_local + i*cols_local*blocks +  j] = RT_local[i*cols_local+j];}
                    PRAGMA_IVDEP
                           for (     ; j < cols_local; j++ ) {
                               RT_buff[k*cols_local + i*cols_local*blocks +  j] = 0.0; } }
            }
            else /* if( (QT_local) && (RT_local) ) */
            {
                this->_errors->add(services::ErrorMemoryAllocationFailed);
            }

            if(QT_local){ service_scalable_free<interm,cpu>(QT_local); }
            if(RT_local){ service_scalable_free<interm,cpu>(RT_local); }

        } );

        /* Step2: calculate QR on master node for resulted RB[blocks*n*n] */
        /* ============================================================== */

        /* Call QR on master node for RB */
        compute_QR_on_one_node_seq<interm, cpu>( cols*blocks, cols, RT_buff, cols*blocks, R_buff, cols );

        /* Transpose R */
        for ( int i = 0 ; i < cols ; i++ ) {
        PRAGMA_IVDEP
            for ( int j = 0 ; j < cols; j++ ) {
                R_output[i + j * cols] = R_buff[i * cols + j]; } }


        /* Step3: calculate Q by merging Q*RB */
        /* ================================== */

        daal::threader_for( blocks, blocks, [=,&R_buff,&Q_output](int k)
        {
            interm *Q_block         = Q_output  + k * brows * cols;
            interm *R_block         = R_buff    + k * cols  * cols;

            /* Last block size brows_last (generally larger than other blocks) */
            size_t brows_local = (k==(blocks-1))?brows_last:brows;
            size_t cols_local  = cols;

            interm *QT_local        = service_scalable_malloc<interm,cpu>(cols_local * brows_local);
            interm *QT_result_local = service_scalable_malloc<interm,cpu>(cols_local * brows_local);
            interm *RT_local        = service_scalable_malloc<interm,cpu>(cols_local * cols_local);

            if( (QT_local) && (QT_result_local) && (RT_local))
            {
                /* Transpose RB */
                for ( int i = 0 ; i < cols_local ; i++ ) {
                PRAGMA_IVDEP
                    for ( int j = 0 ; j < cols_local; j++ ) {
                        RT_local[j*cols_local + i] = RT_buff[j*cols_local*blocks + k*cols_local + i]; } }

                /* Transpose Q to QT */
                for ( int i = 0 ; i < cols_local ; i++ ) {
                PRAGMA_IVDEP
                    for ( int j = 0 ; j < brows_local; j++ ) {
                        QT_local[i*brows_local + j] = Q_block[i + j*cols_local]; } }

                /* Call GEMMs to multiply Q*R */
                compute_gemm_on_one_node_seq<interm, cpu>( brows_local, cols_local, QT_local, brows_local, RT_local, cols_local, QT_result_local, brows_local );

                /* Transpose result Q */
                for ( int i = 0 ; i < cols_local ; i++ ) {
                PRAGMA_IVDEP
                    for ( int j = 0 ; j < brows_local; j++ ) {
                        Q_block[i + j*cols_local] = QT_result_local[i*brows_local + j]; } }
            }
            else /* if( (QT_local) && (QT_result_local) && (RT_local)) */
            {
                this->_errors->add(services::ErrorMemoryAllocationFailed);
            }

            if(QT_local){        service_scalable_free<interm,cpu>(QT_local);        }
            if(RT_local){        service_scalable_free<interm,cpu>(RT_local);        }
            if(QT_result_local){ service_scalable_free<interm,cpu>(QT_result_local); }

        } );
    }
    else /* if( (R_buff) && (RT_buff) ) */
    {
        this->_errors->add(services::ErrorMemoryAllocationFailed);
    }

    ntA_input->releaseBlockOfRows( bkA_input);
    ntQ_output->releaseBlockOfRows( bkQ_output);
    ntR_output->releaseBlockOfRows( bkR_output);

    if(R_buff){  service_free<interm,cpu>(R_buff);  }
    if(RT_buff){ service_free<interm,cpu>(RT_buff); }

    return;
}


} // namespace daal::internal
}
}
} // namespace daal

#endif
