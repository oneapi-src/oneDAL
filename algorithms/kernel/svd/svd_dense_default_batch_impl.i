/* file: svd_dense_default_batch_impl.i */
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

#ifndef __SVD_KERNEL_BATCH_IMPL_I__
#define __SVD_KERNEL_BATCH_IMPL_I__

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

/**
 *  \brief Kernel for SVD calculation
 */
template <typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
void SVDBatchKernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable *const *a,
                                                           const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par)
{
    NumericTable *ntAi = const_cast<NumericTable *>(a[0]);

    size_t  n = ntAi->getNumberOfColumns();
    size_t  m = ntAi->getNumberOfRows();

    if(m >= 2*n)
    {
        SVDBatchKernel<algorithmFPType, method, cpu>::compute_thr( na, a, nr, r, par);
    }
    else
    {
        SVDBatchKernel<algorithmFPType, method, cpu>::compute_seq( na, a, nr, r, par);
    }
}



template <typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
void SVDBatchKernel<algorithmFPType, method, cpu>::compute_seq(const size_t na, const NumericTable *const *a,
                                                               const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par)
{
    size_t i, j;
    svd::Parameter defaultParams;
    const svd::Parameter *svdPar = &defaultParams;

    if ( par != 0 )
    {
        svdPar = static_cast<const svd::Parameter *>(par);
    }

    BlockMicroTable<algorithmFPType, readOnly , cpu> mtA    (a[0]);
    BlockMicroTable<algorithmFPType, writeOnly, cpu> mtSigma(r[0]);

    size_t n = mtA.getFullNumberOfColumns();
    size_t m = mtA.getFullNumberOfRows();

    algorithmFPType *A;
    algorithmFPType *Sigma;

    mtA    .getBlockOfRows( 0, m, &A     );
    mtSigma.getBlockOfRows( 0, 1, &Sigma );

    algorithmFPType *AT = (algorithmFPType *)daal::services::daal_malloc(m * n * sizeof(algorithmFPType));
    algorithmFPType *QT = (algorithmFPType *)daal::services::daal_malloc(m * n * sizeof(algorithmFPType));
    algorithmFPType *VT = (algorithmFPType *)daal::services::daal_malloc(n * n * sizeof(algorithmFPType));

    for ( i = 0 ; i < n ; i++ )
    {
        for ( j = 0 ; j < m; j++ )
        {
            AT[i * m + j] = A[i + j * n];
        }
    }

    compute_svd_on_one_node<algorithmFPType, cpu>( m, n, AT, m, Sigma, QT, m, VT, n );

    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        BlockMicroTable<algorithmFPType, writeOnly, cpu> mtQ(r[1]);
        algorithmFPType *Q;
        mtQ.getBlockOfRows( 0, m, &Q );
        for ( i = 0 ; i < n ; i++ )
        {
            for ( j = 0 ; j < m; j++ )
            {
                Q[i + j * n] = QT[i * m + j];
            }
        }
        mtQ.release();
    }

    if (svdPar->rightSingularMatrix == requiredInPackedForm)
    {
        BlockMicroTable<algorithmFPType, writeOnly, cpu> mtV(r[2]);
        algorithmFPType *V;
        mtV.getBlockOfRows( 0, n, &V );
        for ( i = 0 ; i < n ; i++ )
        {
            for ( j = 0 ; j < n; j++ )
            {
                V[i + j * n] = VT[i * n + j];
            }
        }
        mtV.release();
    }

    mtA    .release();
    mtSigma.release();

    daal::services::daal_free(AT);
    daal::services::daal_free(QT);
    daal::services::daal_free(VT);
}


/* Max number of blocks depending on arch */
#if( __CPUID__(DAAL_CPU) >= __avx512_mic__ )
    #define DEF_MAX_BLOCKS 256
#else
    #define DEF_MAX_BLOCKS 128
#endif

/*
    Algorithm for parallel SVD computation:
    -------------------------------------
    A[m,n] input matrix to be factorized by output U[m,n] and V[n,n]

    1st step:
    Split A[m,n] matrix to 'b' blocks -> a1[m1,n],a2[m2,]...ab[mb,n]
    Compute QR decomposition for each block in threads:
                               a1[m1,n] -> q1[m1,n] , r1[n,n] ... ab[mb,n] -> qb[mb,n] and rb[n,n]

    2nd step:
    Concatenate r1[n,n] , r2[n,n] ... rb[n,n] into one matrix B[n*b,n]
    Compute QR decomposition for B[n*b,n] -> P[n*b,n] , R[n,n]. R - resulted matrix
    Compute SVD decomposition for R, get U, V matrices and Sigma vector.
    V - resulted matrix.
    Multiply U by B matrices.

    3rd step: Split P[n*b,n] matrix to 'b' blocks -> p1[n,n],p2[n,n]...pb[n,n]
    Multiply by q1..qb matrices from 1st step using GEMM for each block in threads:
                               q1[m1,n] * p1[n,n] -> q'1[m1,n] ... qb[mb,n] * pb[n,n] -> q'b[mb,n]
    Concatenate q'1[m1,n]...q'b[mb,n] into one resulted U[m,n]  matrix.

    Notice: before and after QR and GEMM computations matrices need to be transposed
*/
template <typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
void SVDBatchKernel<algorithmFPType, method, cpu>::compute_thr(const size_t na, const NumericTable *const *a,
                                                               const size_t nr, NumericTable *r[], const daal::algorithms::Parameter *par)
{
    svd::Parameter defaultParams;
    const svd::Parameter *svdPar = &defaultParams;

    if ( par != 0 )
    {
        svdPar = static_cast<const svd::Parameter *>(par);
    }

    NumericTable *ntA_input  = const_cast<NumericTable *>(a[0]);
    BlockDescriptor<algorithmFPType> bkA_input;

    size_t  n = ntA_input->getNumberOfColumns();
    size_t  m = ntA_input->getNumberOfRows();
    size_t rows = m;
    size_t cols = n;

    ntA_input->getBlockOfRows(  0, m, readOnly,  bkA_input);
    algorithmFPType *A_input  = bkA_input.getBlockPtr();

    NumericTable *ntS_output = const_cast<NumericTable *>(r[0]);
    BlockDescriptor<algorithmFPType> bkS_output;
    ntS_output->getBlockOfRows( 0, 1, writeOnly, bkS_output);
    algorithmFPType *S_output = bkS_output.getBlockPtr();

    NumericTable *ntU_output;
    BlockDescriptor<algorithmFPType> bkU_output;
    algorithmFPType *U_output;
    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        ntU_output = const_cast<NumericTable *>(r[1]);
        ntU_output->getBlockOfRows( 0, m, writeOnly, bkU_output);
        U_output = bkU_output.getBlockPtr();
    }

    NumericTable *ntV_output;
    BlockDescriptor<algorithmFPType> bkV_output;
    algorithmFPType *V_output;
    if (svdPar->rightSingularMatrix == requiredInPackedForm)
    {
        ntV_output = const_cast<NumericTable *>(r[2]);
        ntV_output->getBlockOfRows( 0, n, writeOnly, bkV_output);
        V_output = bkV_output.getBlockPtr();
    }

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

    algorithmFPType *R_buff   = service_calloc<algorithmFPType, cpu>(blocks * cols * cols); /* zeroing */
    algorithmFPType *RT_buff  = service_malloc<algorithmFPType, cpu>(blocks * cols * cols);
    algorithmFPType *Q_buff   = 0;
    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        Q_buff = U_output; /* re-use output U for Q_buff */
    }
    else
    {
        Q_buff = service_malloc<algorithmFPType, cpu>(rows * cols); /* allocate new Q_buff */
    }

    if((R_buff) && (RT_buff) )
    {
        /* Step1: calculate QR on local nodes */
        /* ================================== */

        daal::threader_for( blocks, blocks, [=,&A_input,&Q_buff,&RT_buff](int k)
        {
            algorithmFPType *A_block = A_input  + k * brows * cols;
            algorithmFPType *Q_block = Q_buff   + k * brows * cols;

            /* Last block size brows_last (generally larger than other blocks) */
            size_t brows_local = (k==(blocks-1))?brows_last:brows;
            size_t cols_local  = cols;

            algorithmFPType *QT_local = service_scalable_malloc<algorithmFPType, cpu>(cols_local * brows_local);
            algorithmFPType *RT_local = service_scalable_malloc<algorithmFPType, cpu>(cols_local * cols_local);
            if( (QT_local) && (RT_local) )
            {
                /* Get transposed Q from A */
                for ( int i = 0 ; i < cols_local ; i++ ) {
                PRAGMA_IVDEP
                    for ( int j = 0 ; j < brows_local; j++ ) {
                        QT_local[i*brows_local+j] = A_block[i+j*cols_local]; } }

                /* Call QR on local nodes */
                compute_QR_on_one_node_seq<algorithmFPType, cpu>( brows_local, cols_local, QT_local, brows_local, RT_local, cols_local );

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
                               RT_buff[k*cols_local + i*cols_local*blocks + j] = RT_local[i*cols_local+j];}
                    PRAGMA_IVDEP
                           for (     ; j < cols_local; j++ ) {
                               RT_buff[k*cols_local + i*cols_local*blocks + j] = 0.0; } }
            }
            else /* if( (QT_local) && (RT_local) ) */
            {
                this->_errors->add(services::ErrorMemoryAllocationFailed);
            }
            if(QT_local) { service_scalable_free<algorithmFPType, cpu>(QT_local); }
            if(RT_local) { service_scalable_free<algorithmFPType, cpu>(RT_local); }
        } );

        /* Step2: calculate QR on master node for resulted RB[blocks*n*n] */

        algorithmFPType *U_buff  = service_malloc<algorithmFPType, cpu>(n * n);
        algorithmFPType *V_buff  = service_malloc<algorithmFPType, cpu>(n * n);
        if( (U_buff) && (V_buff))
        {
            /* Call QR on master node for RB */
            compute_QR_on_one_node_seq<algorithmFPType, cpu>( cols * blocks, cols, RT_buff, cols * blocks, R_buff, cols );
            compute_svd_on_one_node_seq<algorithmFPType, cpu>( cols, cols, R_buff, cols, S_output, U_buff, cols, V_buff, cols );

            if (svdPar->leftSingularMatrix == requiredInPackedForm)
            {
                compute_gemm_on_one_node_seq<algorithmFPType, cpu>( cols * blocks, cols, RT_buff, cols * blocks, U_buff, cols, R_buff, cols * blocks );
            }

            if (svdPar->rightSingularMatrix == requiredInPackedForm)
            {
                /* Transpose result R and save to V output */
                for ( int i = 0 ; i < cols ; i++ ) {
                PRAGMA_IVDEP
                    for ( int j = 0 ; j < cols; j++ ) {
                        V_output[i + j * cols] = V_buff[i * cols + j]; } }
            }
        }
        else /* if( (U_buff) && (V_buff)) */
        {
            this->_errors->add(services::ErrorMemoryAllocationFailed);
        }
        if(U_buff) { service_free<algorithmFPType, cpu>(U_buff); }
        if(V_buff) { service_free<algorithmFPType, cpu>(V_buff); }


        /* Step3: calculate Q by merging Q*RB */
        /* ================================== */

        if (svdPar->leftSingularMatrix == requiredInPackedForm)
        {
            daal::threader_for( blocks, blocks, [=,&RT_buff,&U_output,&Q_buff](int k)
            {
                algorithmFPType *Q_block         = Q_buff     + k * brows * cols;
                algorithmFPType *RT_block        = RT_buff    + k * cols  * cols;
                algorithmFPType *U_block         = U_output   + k * brows * cols; /* the same as Q_buff here */

                /* Last block size brows_last (generally larger than other blocks) */
                size_t brows_local = (k==(blocks-1))?brows_last:brows;
                size_t cols_local  = cols;

                algorithmFPType *QT_local        = service_scalable_malloc<algorithmFPType, cpu>(cols_local * brows_local);
                algorithmFPType *QT_result_local = service_scalable_malloc<algorithmFPType, cpu>(cols_local * brows_local);
                algorithmFPType *RT_local        = service_scalable_malloc<algorithmFPType, cpu>(cols_local * cols_local);

                if( (QT_local) && (QT_result_local) && (RT_local))
                {
                    /* Transpose RB */
                    for ( int i = 0 ; i < cols_local ; i++ ) {
                    PRAGMA_IVDEP
                        for ( int j = 0 ; j < cols_local; j++ ) {
                            RT_block[i*cols_local + j] = R_buff[j*cols_local*blocks + k*cols_local + i]; } }

                    /* Transpose Q to QT */
                    for ( int i = 0 ; i < cols_local ; i++ ) {
                    PRAGMA_IVDEP
                        for ( int j = 0 ; j < brows_local; j++ ) {
                            QT_local[i*brows_local + j] = Q_block[i + j*cols_local]; } }

                    /* Transpose R to RT */
                    for ( int i = 0 ; i < cols_local ; i++ ) {
                    PRAGMA_IVDEP
                        for ( int j = 0 ; j < cols_local; j++ )  {
                            RT_local[i*cols_local + j] = RT_block[i + j*cols_local]; } }

                    /* Call GEMMs to multiply Q*R */
                    compute_gemm_on_one_node_seq<algorithmFPType, cpu>( brows_local, cols_local, QT_local, brows_local, RT_local, cols_local, QT_result_local,
                                                                        brows_local );

                    /* Transpose result Q */
                    for ( int i = 0 ; i < cols_local ; i++ ) {
                    PRAGMA_IVDEP
                        for ( int j = 0 ; j < brows_local; j++ ) {
                            U_block[i + j*cols_local] = QT_result_local[i*brows_local + j]; } }
                }
                else /* if( (QT_local) && (QT_result_local) && (RT_local)) */
                {
                    this->_errors->add(services::ErrorMemoryAllocationFailed);
                }

                if(QT_local) {        service_scalable_free<algorithmFPType, cpu>(QT_local);        }
                if(RT_local) {        service_scalable_free<algorithmFPType, cpu>(RT_local);        }
                if(QT_result_local) { service_scalable_free<algorithmFPType, cpu>(QT_result_local); }

            } );
        } /* if (svdPar->leftSingularMatrix == requiredInPackedForm) */
    }
    else /* if( (R_buff) && (RT_buff) ) */
    {
        this->_errors->add(services::ErrorMemoryAllocationFailed);
    }
    if(R_buff) {  service_free<algorithmFPType, cpu>(R_buff);  }
    if(RT_buff) { service_free<algorithmFPType, cpu>(RT_buff); }

    ntA_input->releaseBlockOfRows( bkA_input);
    ntS_output->releaseBlockOfRows( bkS_output);
    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        ntU_output->releaseBlockOfRows( bkU_output);
    }
    else
    {
        if(Q_buff) { service_free<algorithmFPType, cpu>(Q_buff); } /* free Q_buff if it was allocated */
    }

    if (svdPar->rightSingularMatrix == requiredInPackedForm)
    {
        ntV_output->releaseBlockOfRows( bkV_output);
    }

    return;
}


} // namespace daal::internal
}
}
} // namespace daal

#endif
