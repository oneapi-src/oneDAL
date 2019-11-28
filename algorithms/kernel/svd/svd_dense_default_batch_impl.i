/* file: svd_dense_default_batch_impl.i */
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

#ifndef __SVD_KERNEL_BATCH_IMPL_I__
#define __SVD_KERNEL_BATCH_IMPL_I__

#include "service_memory.h"
#include "service_math.h"
#include "service_defines.h"
#include "service_numeric_table.h"
#include "service_error_handling.h"

#include "svd_dense_default_impl.i"

#include "threading.h"

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace svd
{
namespace internal
{
#include "./../qr/qr_dense_default_pcl_impl.i"

/**
 *  \brief Kernel for SVD calculation
 */
template <typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
Status SVDBatchKernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[],
                                                             const daal::algorithms::Parameter * par)
{
    svd::Parameter defaultParams;
    const svd::Parameter * svdPar = &defaultParams;

    if (par != 0)
    {
        svdPar = static_cast<const svd::Parameter *>(par);
    }

    NumericTable * ntAi = const_cast<NumericTable *>(a[0]);

    const size_t n = ntAi->getNumberOfColumns();
    const size_t m = ntAi->getNumberOfRows();
    const size_t t = threader_get_threads_number();

    if (m < 2 * n) return SVDBatchKernel<algorithmFPType, method, cpu>::compute_seq(na, a, nr, r, svdPar);

    if ((m > n * t) && (n > 10) && (!(n >= 200 && m <= 100000)))
        return SVDBatchKernel<algorithmFPType, method, cpu>::compute_pcl(na, a, nr, r, svdPar);

    return SVDBatchKernel<algorithmFPType, method, cpu>::compute_thr(na, a, nr, r, svdPar);
}

template <typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
Status SVDBatchKernel<algorithmFPType, method, cpu>::compute_seq(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[],
                                                                 const Parameter * svdPar)
{
    NumericTable * ntA     = const_cast<NumericTable *>(a[0]);
    NumericTable * ntSigma = const_cast<NumericTable *>(r[0]);

    const size_t n           = ntA->getNumberOfColumns();
    const size_t m           = ntA->getNumberOfRows();
    const size_t nComponents = ntSigma->getNumberOfColumns();

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, sizeof(algorithmFPType));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, m);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * m, sizeof(algorithmFPType));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, n);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * n, sizeof(algorithmFPType));
    TArray<algorithmFPType, cpu> ATPtr(n * m);
    algorithmFPType * AT = ATPtr.get();
    TArray<algorithmFPType, cpu> QTPtr(n * m);
    algorithmFPType * QT = QTPtr.get();
    TArray<algorithmFPType, cpu> VTPtr(n * n);
    algorithmFPType * VT = VTPtr.get();
    DAAL_CHECK(AT && QT && VT, ErrorMemoryAllocationFailed);

    {
        ReadRows<algorithmFPType, cpu, NumericTable> aBlock(ntA, 0, m);
        DAAL_CHECK_BLOCK_STATUS(aBlock);
        const algorithmFPType * A = aBlock.get();

        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < m; j++)
            {
                AT[i * m + j] = A[i + j * n];
            }
        }
    }

    {
        TArray<algorithmFPType, cpu> sigmaPtr(n);
        DAAL_CHECK_MALLOC(sigmaPtr.get());
        algorithmFPType * Sigma = sigmaPtr.get();

        compute_svd_on_one_node<algorithmFPType, cpu>(m, n, AT, m, Sigma, QT, m, VT, n);

        WriteOnlyRows<algorithmFPType, cpu, NumericTable> sigmaBlock(ntSigma, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(sigmaBlock);
        algorithmFPType * tSigma = sigmaBlock.get();

        for (size_t i = 0; i < nComponents; i++)
        {
            tSigma[i] = Sigma[i];
        }
    }

    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        WriteOnlyRows<algorithmFPType, cpu, NumericTable> qBlock(r[1], 0, m);
        DAAL_CHECK_BLOCK_STATUS(qBlock);
        algorithmFPType * Q = qBlock.get();

        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < m; j++)
            {
                Q[i + j * n] = QT[i * m + j];
            }
        }
    }

    if (svdPar->rightSingularMatrix == requiredInPackedForm)
    {
        WriteOnlyRows<algorithmFPType, cpu, NumericTable> vBlock(r[2], 0, nComponents);
        DAAL_CHECK_BLOCK_STATUS(vBlock);
        algorithmFPType * V = vBlock.get();

        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < nComponents; j++)
            {
                V[i + j * n] = VT[i * n + j];
            }
        }
    }

    return Status();
}

/* Max number of blocks depending on arch */
#if (__CPUID__(DAAL_CPU) >= __avx512_mic__)
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
Status SVDBatchKernel<algorithmFPType, method, cpu>::compute_thr(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[],
                                                                 const Parameter * svdPar)
{
    NumericTable * ntA_input = const_cast<NumericTable *>(a[0]);
    NumericTable * ntS_output(r[0]);
    NumericTable * ntU_output(r[1]);
    NumericTable * ntV_output(r[2]);

    const size_t n = ntA_input->getNumberOfColumns();
    const size_t m = ntA_input->getNumberOfRows();

    const size_t nComponents = ntS_output->getNumberOfColumns();

    size_t rows = m;
    size_t cols = n;

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, sizeof(algorithmFPType));
    /* Getting real pointers to output array */
    TArray<algorithmFPType, cpu> aS_output(n);
    DAAL_CHECK_MALLOC(aS_output.get());
    algorithmFPType * S_output = aS_output.get();

    WriteOnlyRows<algorithmFPType, cpu, NumericTable> bkU_output;
    algorithmFPType * U_output;
    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        U_output = bkU_output.set(ntU_output, 0, m);
        DAAL_CHECK_BLOCK_STATUS(bkU_output);
    }

    WriteOnlyRows<algorithmFPType, cpu, NumericTable> bkV_output;
    algorithmFPType * V_output;
    if (svdPar->rightSingularMatrix == requiredInPackedForm)
    {
        V_output = bkV_output.set(ntV_output, 0, nComponents);
        DAAL_CHECK_BLOCK_STATUS(bkV_output);
    }

    /* Block size calculation (empirical) */
    const int bshift = (rows <= 10000) ? 11 : 12;
    int bsize        = ((rows * cols) >> bshift) & (~0xf);
    bsize            = (bsize < 200) ? 200 : bsize;

    /*
    Calculate sizes:
    blocks     = number of blocks,
    brows      = number of rows in blocks,
    brows_last = number of rows in last block,
    */
    size_t def_min_brows = rows / DEF_MAX_BLOCKS;                           // min block size
    size_t brows         = (rows > bsize) ? bsize : rows;                   /* brows cannot be less than rows */
    brows                = (brows < cols) ? cols : brows;                   /* brows cannot be less than cols */
    brows                = (brows < def_min_brows) ? def_min_brows : brows; /* brows cannot be less than n/DEF_MAX_BLOCKS */
    size_t blocks        = rows / brows;

    size_t brows_last = brows + (rows - blocks * brows); /* last block is generally biggest */

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, n);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * n, sizeof(algorithmFPType));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * n, blocks);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * n * blocks, sizeof(algorithmFPType));
    size_t len = blocks * n * n;
    TArrayCalloc<algorithmFPType, cpu> R_buffPtr(len); /* zeroing */
    algorithmFPType * R_buff = R_buffPtr.get();
    DAAL_CHECK(R_buff, ErrorMemoryAllocationFailed);

    TArray<algorithmFPType, cpu> RT_buffPtr(len);
    algorithmFPType * RT_buff = RT_buffPtr.get();
    DAAL_CHECK(RT_buff, ErrorMemoryAllocationFailed);

    TArray<algorithmFPType, cpu> Q_buffPtr;
    algorithmFPType * Q_buff = 0;
    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        Q_buff = U_output; /* re-use output U for Q_buff */
    }
    else
    {
        Q_buffPtr.reset(rows * cols);
        Q_buff = Q_buffPtr.get(); /* allocate new Q_buff */
    }

    SafeStatus safeStat;
    /* Step1: calculate QR on local nodes */
    /* ================================== */
    {
        /* Getting real pointers to input array */
        ReadRows<algorithmFPType, cpu, NumericTable> bkA_input(ntA_input, 0, m);
        DAAL_CHECK_BLOCK_STATUS(bkA_input);
        const algorithmFPType * A_input = bkA_input.get();

        daal::threader_for(blocks, blocks, [=, &A_input, &Q_buff, &RT_buff, &safeStat](int k) {
            const algorithmFPType * A_block = A_input + k * brows * cols;
            algorithmFPType * Q_block       = Q_buff + k * brows * cols;

            /* Last block size brows_last (generally larger than other blocks) */
            const size_t brows_local = (k == (blocks - 1)) ? brows_last : brows;
            const size_t cols_local  = cols;

            TArrayScalable<algorithmFPType, cpu> QT_local_Arr(cols_local * brows_local);
            algorithmFPType * QT_local = QT_local_Arr.get();
            TArrayScalable<algorithmFPType, cpu> RT_local_Arr(cols_local * cols_local);
            algorithmFPType * RT_local = RT_local_Arr.get();

            if (!(QT_local && RT_local))
            {
                safeStat.add(services::ErrorMemoryAllocationFailed);
                return;
            }
            /* Get transposed Q from A */
            for (int i = 0; i < cols_local; i++)
            {
                PRAGMA_IVDEP
                for (int j = 0; j < brows_local; j++)
                {
                    QT_local[i * brows_local + j] = A_block[i + j * cols_local];
                }
            }

            /* Call QR on local nodes */
            const auto ec = compute_QR_on_one_node_seq<algorithmFPType, cpu>(brows_local, cols_local, QT_local, brows_local, RT_local, cols_local);
            if (!ec)
            {
                safeStat.add(ec);
                return;
            }

            /* Transpose Q */
            for (int i = 0; i < cols_local; i++)
            {
                PRAGMA_IVDEP
                for (int j = 0; j < brows_local; j++)
                {
                    Q_block[i + j * cols_local] = QT_local[i * brows_local + j];
                }
            }

            /* Transpose R and zero lower values */
            for (int i = 0; i < cols_local; i++)
            {
                int j;
                PRAGMA_IVDEP
                for (j = 0; j <= i; j++)
                {
                    RT_buff[k * cols_local + i * cols_local * blocks + j] = RT_local[i * cols_local + j];
                }
                PRAGMA_IVDEP
                for (; j < cols_local; j++)
                {
                    RT_buff[k * cols_local + i * cols_local * blocks + j] = 0.0;
                }
            }
        });
    }
    if (!safeStat) return safeStat.detach();

    /* Step2: calculate QR on master node for resulted RB[blocks*n*n] */
    /* ============================================================== */

    {
        TArray<algorithmFPType, cpu> U_buffPtr(n * n);
        algorithmFPType * U_buff = U_buffPtr.get();
        DAAL_CHECK(U_buff, ErrorMemoryAllocationFailed);

        TArray<algorithmFPType, cpu> V_buffPtr(n * n);
        algorithmFPType * V_buff = V_buffPtr.get();
        DAAL_CHECK(V_buff, ErrorMemoryAllocationFailed);

        /* Call QR on master node for RB */
        const auto ecQr = compute_QR_on_one_node_seq<algorithmFPType, cpu>(cols * blocks, cols, RT_buff, cols * blocks, R_buff, cols);
        if (!ecQr) return ecQr;
        const auto ecSvd = compute_svd_on_one_node_seq<algorithmFPType, cpu>(cols, cols, R_buff, cols, S_output, U_buff, cols, V_buff, cols);
        if (!ecSvd) return ecSvd;

        WriteOnlyRows<algorithmFPType, cpu, NumericTable> bkS_output(ntS_output, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(bkS_output);
        algorithmFPType * tS_output = bkS_output.get();

        for (size_t i = 0; i < nComponents; i++)
        {
            tS_output[i] = S_output[i];
        }

        if (svdPar->leftSingularMatrix == requiredInPackedForm)
        {
            compute_gemm_on_one_node_seq<algorithmFPType, cpu>(cols * blocks, cols, RT_buff, cols * blocks, U_buff, cols, R_buff, cols * blocks);
        }

        if (svdPar->rightSingularMatrix == requiredInPackedForm)
        {
            /* Transpose result R and save to V output */
            for (int i = 0; i < cols; i++)
            {
                PRAGMA_IVDEP
                for (int j = 0; j < nComponents; j++)
                {
                    V_output[i + j * cols] = V_buff[i * cols + j];
                }
            }
        }
    }

    /* Step3: calculate Q by merging Q*RB */
    /* ================================== */

    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        daal::threader_for(blocks, blocks, [=, &RT_buff, &U_output, &Q_buff, &safeStat](int k) {
            algorithmFPType * Q_block  = Q_buff + k * brows * cols;
            algorithmFPType * RT_block = RT_buff + k * cols * cols;
            algorithmFPType * U_block  = U_output + k * brows * cols; /* the same as Q_buff here */

            /* Last block size brows_last (generally larger than other blocks) */
            size_t brows_local = (k == (blocks - 1)) ? brows_last : brows;
            size_t cols_local  = cols;

            TArrayScalable<algorithmFPType, cpu> QT_local_Arr(cols_local * brows_local);
            algorithmFPType * QT_local = QT_local_Arr.get();
            TArrayScalable<algorithmFPType, cpu> RT_local_Arr(cols_local * cols_local);
            algorithmFPType * RT_local = RT_local_Arr.get();
            TArrayScalable<algorithmFPType, cpu> QT_result_local_Arr(cols_local * brows_local);
            algorithmFPType * QT_result_local = QT_result_local_Arr.get();

            if (!(QT_local && QT_result_local && RT_local))
            {
                safeStat.add(services::ErrorMemoryAllocationFailed);
                return;
            }
            /* Transpose RB */
            for (int i = 0; i < cols_local; i++)
            {
                PRAGMA_IVDEP
                for (int j = 0; j < cols_local; j++)
                {
                    RT_block[i * cols_local + j] = R_buff[j * cols_local * blocks + k * cols_local + i];
                }
            }

            /* Transpose Q to QT */
            for (int i = 0; i < cols_local; i++)
            {
                PRAGMA_IVDEP
                for (int j = 0; j < brows_local; j++)
                {
                    QT_local[i * brows_local + j] = Q_block[i + j * cols_local];
                }
            }

            /* Transpose R to RT */
            for (int i = 0; i < cols_local; i++)
            {
                PRAGMA_IVDEP
                for (int j = 0; j < cols_local; j++)
                {
                    RT_local[i * cols_local + j] = RT_block[i + j * cols_local];
                }
            }

            /* Call GEMMs to multiply Q*R */
            compute_gemm_on_one_node_seq<algorithmFPType, cpu>(brows_local, cols_local, QT_local, brows_local, RT_local, cols_local, QT_result_local,
                                                               brows_local);

            /* Transpose result Q */
            for (int i = 0; i < cols_local; i++)
            {
                PRAGMA_IVDEP
                for (int j = 0; j < brows_local; j++)
                {
                    U_block[i + j * cols_local] = QT_result_local[i * brows_local + j];
                }
            }
        });
    } /* if (svdPar->leftSingularMatrix == requiredInPackedForm) */

    return safeStat.detach();
}

template <typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
Status SVDBatchKernel<algorithmFPType, method, cpu>::compute_pcl(const size_t na, const NumericTable * const * a, const size_t nr, NumericTable * r[],
                                                                 const Parameter * svdPar)
{
    NumericTable * ntA = const_cast<NumericTable *>(a[0]);
    NumericTable * ntSigma(r[0]);
    NumericTable * ntQ(r[1]);
    NumericTable * ntV(r[2]);

    size_t n           = ntA->getNumberOfColumns();
    size_t m           = ntA->getNumberOfRows();
    size_t nComponents = ntSigma->getNumberOfColumns();

    ReadRows<algorithmFPType, cpu, NumericTable> aBlock(ntA, 0, m);
    DAAL_CHECK_BLOCK_STATUS(aBlock);
    algorithmFPType * A = const_cast<algorithmFPType *>(aBlock.get());

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, sizeof(algorithmFPType));
    TArray<algorithmFPType, cpu> sigmaArray(n);
    DAAL_CHECK_MALLOC(sigmaArray.get());
    algorithmFPType * Sigma = sigmaArray.get();

    WriteOnlyRows<algorithmFPType, cpu, NumericTable> qBlock;
    algorithmFPType * Q = nullptr;

    TArray<algorithmFPType, cpu> vArray;
    algorithmFPType * V = nullptr;

    bool needsU  = false;
    bool needsVT = false;

    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        Q = qBlock.set(ntQ, 0, m);
        DAAL_CHECK_BLOCK_STATUS(qBlock);
        needsU = true;
    }

    if (svdPar->rightSingularMatrix == requiredInPackedForm)
    {
        vArray.reset(n * n);
        DAAL_CHECK_MALLOC(vArray.get());
        V       = vArray.get();
        needsVT = true;
    }

    const services::ErrorID ec = (services::ErrorID)svd_pcl<algorithmFPType, cpu>(A, m, n, needsU, Q, Sigma, needsVT, V);

    WriteOnlyRows<algorithmFPType, cpu, NumericTable> sBlock(ntSigma, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(sBlock);
    algorithmFPType * tSigma = sBlock.get();

    for (size_t i = 0; i < nComponents; i++)
    {
        tSigma[i] = Sigma[i];
    }

    if (svdPar->rightSingularMatrix == requiredInPackedForm)
    {
        WriteOnlyRows<algorithmFPType, cpu, NumericTable> vBlock(ntV, 0, n);
        DAAL_CHECK_BLOCK_STATUS(vBlock);
        algorithmFPType * tV = vBlock.get();
        for (size_t i = 0; i < nComponents; i++)
        {
            PRAGMA_IVDEP
            for (size_t j = 0; j < n; j++)
            {
                tV[i * n + j] = V[i * n + j];
            }
        }
    }

    return (ec ? Status(ec) : Status());
}

} // namespace internal
} // namespace svd
} // namespace algorithms
} // namespace daal

#endif
