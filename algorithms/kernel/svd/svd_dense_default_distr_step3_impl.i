/* file: svd_dense_default_distr_step3_impl.i */
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

#ifndef __SVD_KERNEL_DISTR_IMPL_I__
#define __SVD_KERNEL_DISTR_IMPL_I__

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
template <typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
Status SVDDistributedStep2Kernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable * const * a, const size_t nr,
                                                                        NumericTable * r[], const daal::algorithms::Parameter * par)
{
    svd::Parameter defaultParams;
    const svd::Parameter * svdPar = &defaultParams;

    if (par != 0)
    {
        svdPar = static_cast<const svd::Parameter *>(par);
    }

    const NumericTable * ntAux2_0 = a[0];
    NumericTable * ntSigma        = const_cast<NumericTable *>(r[0]);

    size_t nBlocks = na;

    size_t n   = ntAux2_0->getNumberOfColumns();
    size_t nxb = n * nBlocks;

    WriteOnlyRows<algorithmFPType, cpu, NumericTable> sigmaBlock(ntSigma, 0, 1); /* Sigma [1][n]   */
    DAAL_CHECK_BLOCK_STATUS(sigmaBlock);
    algorithmFPType * Sigma = sigmaBlock.get();

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, nxb);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, n);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * n, sizeof(algorithmFPType));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n * nxb, sizeof(algorithmFPType));

    TArray<algorithmFPType, cpu> Aux2TPtr(n * nxb);
    TArray<algorithmFPType, cpu> VTPtr(n * n);
    TArray<algorithmFPType, cpu> Aux3TPtr(n * nxb);
    algorithmFPType * Aux2T = Aux2TPtr.get();
    algorithmFPType * VT    = VTPtr.get();
    algorithmFPType * Aux3T = Aux3TPtr.get();

    DAAL_CHECK(Aux2T && VT && Aux3T, ErrorMemoryAllocationFailed);

    SafeStatus safeStat;

    daal::threader_for(nBlocks, nBlocks, [=, &safeStat](int k) {
        ReadRows<algorithmFPType, cpu, NumericTable> aux2Block(const_cast<NumericTable *>(a[k]), 0, n); /* Aux2  [nxb][n] */
        DAAL_CHECK_BLOCK_STATUS_THR(aux2Block);
        const algorithmFPType * Aux2 = aux2Block.get();

        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                Aux2T[j * nxb + k * n + i] = Aux2[i * n + j];
            }
        }
    });

    if (!safeStat) return safeStat.detach();

    {
        DAAL_INT ldAux2 = nxb;
        DAAL_INT ldAux3 = nxb;
        DAAL_INT ldV    = n;

        DAAL_INT ldR = n;
        DAAL_INT ldU = n;

        TArray<algorithmFPType, cpu> RPtr(n * ldR);
        TArray<algorithmFPType, cpu> UPtr(n * ldU);
        algorithmFPType * R = RPtr.get();
        algorithmFPType * U = UPtr.get();

        DAAL_CHECK(U && R, ErrorMemoryAllocationFailed);

        /* By some reason, there was this part in Sample */
        for (size_t i = 0; i < n * ldR; i++)
        {
            R[i] = 0.0;
        }

        // Rc = P*R
        const auto ecQr = compute_QR_on_one_node<algorithmFPType, cpu>(nxb, n, Aux2T, ldAux2, R, ldR);
        if (!ecQr) return ecQr;

        // Qn*R -> Qn*(U*Sigma*V) -> (Qn*U)*Sigma*V
        const auto ecSvd = compute_svd_on_one_node<algorithmFPType, cpu>(n, n, R, ldR, Sigma, U, ldU, VT, ldV);
        if (!ecSvd) return ecSvd;

        if (svdPar->leftSingularMatrix == requiredInPackedForm)
        {
            const auto ecGemm = compute_gemm_on_one_node<algorithmFPType, cpu>(nxb, n, Aux2T, ldAux2, U, ldU, Aux3T, ldAux3);
            if (!ecGemm) return ecGemm;
        }
    }

    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        daal::threader_for(nBlocks, nBlocks, [=, &safeStat](int k) {
            WriteOnlyRows<algorithmFPType, cpu, NumericTable> aux3Block(r[2 + k], 0, n); /* Aux3  [nxb][n] */
            DAAL_CHECK_BLOCK_STATUS_THR(aux3Block);
            algorithmFPType * Aux3 = aux3Block.get();

            for (size_t i = 0; i < n; i++)
            {
                for (size_t j = 0; j < n; j++)
                {
                    Aux3[i * n + j] = Aux3T[j * nxb + k * n + i];
                }
            }
        });
        if (!safeStat) return safeStat.detach();
    }

    if (svdPar->rightSingularMatrix == requiredInPackedForm)
    {
        WriteOnlyRows<algorithmFPType, cpu, NumericTable> VBlock(r[1], 0, n); /* V[n][n] */
        DAAL_CHECK_BLOCK_STATUS(VBlock);
        algorithmFPType * V = VBlock.get();

        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                V[i + j * n] = VT[i * n + j];
            }
        }
    }

    return Status();
}

template <typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
Status SVDDistributedStep3Kernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable * const * a, const size_t nr,
                                                                        NumericTable * r[], const daal::algorithms::Parameter * par)
{
    size_t i, j;
    size_t nBlocks     = na / 2;
    size_t mCalculated = 0;

    ReadRows<algorithmFPType, cpu, NumericTable> Aux1iBlock;
    ReadRows<algorithmFPType, cpu, NumericTable> Aux3iBlock;
    WriteOnlyRows<algorithmFPType, cpu, NumericTable> QiBlock;

    for (size_t k = 0; k < nBlocks; k++)
    {
        NumericTable * ntAux1i = const_cast<NumericTable *>(a[k]);
        NumericTable * ntAux3i = const_cast<NumericTable *>(a[k + nBlocks]);

        size_t n = ntAux1i->getNumberOfColumns();
        size_t m = ntAux1i->getNumberOfRows();

        const algorithmFPType * Aux1i = Aux1iBlock.set(ntAux1i, 0, m); /* Aux1i = Qin[m][n] */
        DAAL_CHECK_BLOCK_STATUS(Aux1iBlock);

        const algorithmFPType * Aux3i = Aux3iBlock.set(ntAux3i, 0, n); /* Aux3i = Ri [n][n] */
        DAAL_CHECK_BLOCK_STATUS(Aux3iBlock);

        algorithmFPType * Qi = QiBlock.set(r[0], mCalculated, m); /* Qi [m][n] */
        DAAL_CHECK_BLOCK_STATUS(QiBlock);

        TArray<algorithmFPType, cpu> QiTPtr(n * m);
        TArray<algorithmFPType, cpu> Aux1iTPtr(n * m);
        TArray<algorithmFPType, cpu> Aux3iTPtr(n * n);
        algorithmFPType * QiT    = QiTPtr.get();
        algorithmFPType * Aux1iT = Aux1iTPtr.get();
        algorithmFPType * Aux3iT = Aux3iTPtr.get();

        DAAL_CHECK(QiT && Aux1iT && Aux3iT, ErrorMemoryAllocationFailed);

        for (i = 0; i < n; i++)
        {
            for (j = 0; j < m; j++)
            {
                Aux1iT[i * m + j] = Aux1i[i + j * n];
            }
        }
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < n; j++)
            {
                Aux3iT[i * n + j] = Aux3i[i + j * n];
            }
        }

        DAAL_INT ldAux1i = m;
        DAAL_INT ldAux3i = n;
        DAAL_INT ldQi    = m;

        const auto ec = compute_gemm_on_one_node<algorithmFPType, cpu>(m, n, Aux1iT, ldAux1i, Aux3iT, ldAux3i, QiT, ldQi);
        if (!ec) return ec;

        for (i = 0; i < n; i++)
        {
            for (j = 0; j < m; j++)
            {
                Qi[i + j * n] = QiT[i * m + j];
            }
        }

        mCalculated += m;
    }

    return Status();
}

} // namespace internal
} // namespace svd
} // namespace algorithms
} // namespace daal

#endif
