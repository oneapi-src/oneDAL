/* file: linear_model_train_qr_finalize_impl.i */
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
//  Implementation of common base classes for normal equations model training.
//--
*/

#include "src/algorithms/linear_model/linear_model_train_qr_kernel.h"
#include "src/externals/service_lapack.h"
#include "src/externals/service_blas.h"
#include "src/services/service_data_utils.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace qr
{
namespace training
{
namespace internal
{
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services::internal;

template <typename algorithmFPType, CpuType cpu>
Status FinalizeKernel<algorithmFPType, cpu>::compute(const NumericTable & rTable, const NumericTable & qtyTable, NumericTable & rFinalTable,
                                                     NumericTable & qtyFinalTable, NumericTable & betaTable, bool interceptFlag)
{
    const DAAL_INT nBetas(betaTable.getNumberOfColumns());
    const DAAL_INT nResponses(betaTable.getNumberOfRows());
    const DAAL_INT nBetasIntercept = (interceptFlag ? nBetas : (nBetas - 1));

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nResponses, nBetasIntercept);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nResponses * nBetasIntercept, sizeof(algorithmFPType));

    TArrayScalable<algorithmFPType, cpu> betaBufferArray(nResponses * nBetasIntercept);
    algorithmFPType * betaBuffer = betaBufferArray.get();
    DAAL_CHECK_MALLOC(betaBuffer);

    {
        ReadRowsType rBlock(const_cast<NumericTable &>(rTable), 0, nBetasIntercept);
        DAAL_CHECK_BLOCK_STATUS(rBlock);
        const algorithmFPType * r = rBlock.get();

        {
            ReadRowsType qtyBlock(const_cast<NumericTable &>(qtyTable), 0, nResponses);
            DAAL_CHECK_BLOCK_STATUS(qtyBlock);
            const algorithmFPType * qty = qtyBlock.get();

            if (&rTable != &rFinalTable || &qtyTable != &qtyFinalTable)
            {
                int result = 0;
                WriteOnlyRowsType rFinalBlock(rFinalTable, 0, nBetasIntercept);
                DAAL_CHECK_BLOCK_STATUS(rFinalBlock);
                algorithmFPType * rFinal = rFinalBlock.get();

                WriteOnlyRowsType qtyFinalBlock(qtyFinalTable, 0, nResponses);
                DAAL_CHECK_BLOCK_STATUS(qtyFinalBlock);
                algorithmFPType * qtyFinal = qtyFinalBlock.get();

                const size_t rSizeInBytes(sizeof(algorithmFPType) * nBetasIntercept * nBetasIntercept);
                const size_t qtySizeInBytes(sizeof(algorithmFPType) * nBetasIntercept * nResponses);

                result |= daal::services::internal::daal_memcpy_s(rFinal, rSizeInBytes, r, rSizeInBytes);
                result |= daal::services::internal::daal_memcpy_s(qtyFinal, qtySizeInBytes, qty, qtySizeInBytes);
                DAAL_CHECK(!result, services::ErrorMemoryCopyFailedInternal);
            }

            for (size_t i = 0; i < nResponses; i++)
            {
                PRAGMA_OMP_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j < nBetasIntercept; j++)
                {
                    betaBuffer[i * nBetasIntercept + j] = qty[j * nResponses + i];
                }
            }
        }

        /* R'*beta = Y*Q'. If R is singular, use the minimum-norm solution from its SVD, dropping
         * singular values <= eps * max(S) (as gelsd does). */
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nBetasIntercept, nBetasIntercept);
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nBetasIntercept * nBetasIntercept, sizeof(algorithmFPType));
        const size_t rSize        = size_t(nBetasIntercept) * size_t(nBetasIntercept);
        const size_t rSizeInBytes = rSize * sizeof(algorithmFPType);

        TArrayScalable<algorithmFPType, cpu> aArray(rSize);
        TArrayScalable<algorithmFPType, cpu> sArray(nBetasIntercept);
        algorithmFPType * a = aArray.get();
        algorithmFPType * s = sArray.get();
        DAAL_CHECK_MALLOC(a);
        DAAL_CHECK_MALLOC(s);

        /* gesvd destroys its input */
        int copyResult = daal::services::internal::daal_memcpy_s(a, rSizeInBytes, r, rSizeInBytes);
        DAAL_CHECK(!copyResult, services::ErrorMemoryCopyFailedInternal);

        const algorithmFPType eps = daal::services::internal::EpsilonVal<algorithmFPType>::get();

        DAAL_INT info(0);
        DAAL_INT lwork(-1);
        algorithmFPType workQuery(0);
        algorithmFPType dummy(0);

        /* Singular values only, to detect numerical rank deficiency */
        LapackInst<algorithmFPType, cpu>::xgesvd('N', 'N', nBetasIntercept, nBetasIntercept, a, nBetasIntercept, s, &dummy, 1, &dummy, 1, &workQuery,
                                                 lwork, &info);
        DAAL_CHECK(info == 0, services::ErrorLinearRegressionInternal);
        lwork = static_cast<DAAL_INT>(workQuery);
        {
            TArrayScalable<algorithmFPType, cpu> workArray(lwork);
            algorithmFPType * work = workArray.get();
            DAAL_CHECK_MALLOC(work);
            LapackInst<algorithmFPType, cpu>::xgesvd('N', 'N', nBetasIntercept, nBetasIntercept, a, nBetasIntercept, s, &dummy, 1, &dummy, 1, work,
                                                     lwork, &info);
            DAAL_CHECK(info == 0, services::ErrorLinearRegressionInternal);
        }

        /* Singular values are sorted in decreasing order */
        const algorithmFPType tol = eps * s[0];
        if (s[nBetasIntercept - 1] > tol)
        {
            char up     = 'U';
            char trans  = 'T';
            char nodiag = 'N';
            LapackInst<algorithmFPType, cpu>::xtrtrs(
                &up, &trans, &nodiag, const_cast<DAAL_INT *>(&nBetasIntercept), const_cast<DAAL_INT *>(&nResponses), const_cast<algorithmFPType *>(r),
                const_cast<DAAL_INT *>(&nBetasIntercept), betaBuffer, const_cast<DAAL_INT *>(&nBetasIntercept), &info);
            DAAL_CHECK(info == 0, services::ErrorLinearRegressionInternal);
        }
        else
        {
            TArrayScalable<algorithmFPType, cpu> uArray(rSize);
            TArrayScalable<algorithmFPType, cpu> vtArray(rSize);
            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nResponses, nBetasIntercept);
            TArrayScalable<algorithmFPType, cpu> tmpArray(nResponses * nBetasIntercept);
            algorithmFPType * u   = uArray.get();
            algorithmFPType * vt  = vtArray.get();
            algorithmFPType * tmp = tmpArray.get();
            DAAL_CHECK_MALLOC(u);
            DAAL_CHECK_MALLOC(vt);
            DAAL_CHECK_MALLOC(tmp);

            copyResult = daal::services::internal::daal_memcpy_s(a, rSizeInBytes, r, rSizeInBytes);
            DAAL_CHECK(!copyResult, services::ErrorMemoryCopyFailedInternal);

            lwork = -1;
            LapackInst<algorithmFPType, cpu>::xgesvd('A', 'A', nBetasIntercept, nBetasIntercept, a, nBetasIntercept, s, u, nBetasIntercept, vt,
                                                     nBetasIntercept, &workQuery, lwork, &info);
            DAAL_CHECK(info == 0, services::ErrorLinearRegressionInternal);
            lwork = static_cast<DAAL_INT>(workQuery);

            TArrayScalable<algorithmFPType, cpu> workArray(lwork);
            algorithmFPType * work = workArray.get();
            DAAL_CHECK_MALLOC(work);
            LapackInst<algorithmFPType, cpu>::xgesvd('A', 'A', nBetasIntercept, nBetasIntercept, a, nBetasIntercept, s, u, nBetasIntercept, vt,
                                                     nBetasIntercept, work, lwork, &info);
            DAAL_CHECK(info == 0, services::ErrorLinearRegressionInternal);

            /* Store 1/s[j] for the retained singular values and 0 for the discarded ones */
            const algorithmFPType tolFull = eps * s[0];
            for (DAAL_INT j = 0; j < nBetasIntercept; ++j)
            {
                s[j] = (s[j] > tolFull) ? (algorithmFPType(1.0) / s[j]) : algorithmFPType(0.0);
            }

            const algorithmFPType one(1.0);
            const algorithmFPType zero(0.0);
            /* tmp = V' * (Y*Q') */
            BlasInst<algorithmFPType, cpu>::xgemm("N", "N", &nBetasIntercept, &nResponses, &nBetasIntercept, &one, vt, &nBetasIntercept, betaBuffer,
                                                  &nBetasIntercept, &zero, tmp, &nBetasIntercept);
            /* tmp = pinv(S) * tmp */
            for (DAAL_INT i = 0; i < nResponses; ++i)
            {
                for (DAAL_INT j = 0; j < nBetasIntercept; ++j)
                {
                    tmp[i * nBetasIntercept + j] *= s[j];
                }
            }
            /* beta = U * tmp */
            BlasInst<algorithmFPType, cpu>::xgemm("N", "N", &nBetasIntercept, &nResponses, &nBetasIntercept, &one, u, &nBetasIntercept, tmp,
                                                  &nBetasIntercept, &zero, betaBuffer, &nBetasIntercept);
        }
    }

    WriteOnlyRowsType betaBlock(betaTable, 0, nResponses);
    DAAL_CHECK_BLOCK_STATUS(betaBlock);
    algorithmFPType * beta = betaBlock.get();

    if (nBetasIntercept == nBetas)
    {
        for (size_t i = 0; i < nResponses; i++)
        {
            PRAGMA_OMP_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = 1; j < nBetas; j++)
            {
                beta[i * nBetas + j] = betaBuffer[i * nBetas + j - 1];
            }
            beta[i * nBetas] = betaBuffer[i * nBetas + nBetas - 1];
        }
    }
    else
    {
        for (size_t i = 0; i < nResponses; i++)
        {
            PRAGMA_OMP_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = 0; j < nBetas - 1; j++)
            {
                beta[i * nBetas + j + 1] = betaBuffer[i * nBetasIntercept + j];
            }
            beta[i * nBetas] = 0.0;
        }
    }

    return services::Status();
}

} // namespace internal
} // namespace training
} // namespace qr
} // namespace linear_model
} // namespace algorithms
} // namespace daal
