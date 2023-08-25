/* file: service_spblas_ref.h */
/*******************************************************************************
* Copyright 2023 Intel Corporation
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
//  Template wrappers for common Sparse BLAS functions.
//--
*/

#ifndef __SERVICE_SPBLAS_REF_H__
#define __SERVICE_SPBLAS_REF_H__

namespace daal
{
namespace internal
{
namespace ref
{
template <typename fpType, CpuType cpu>
struct RefSpBlas
{
    typedef DAAL_INT SizeType;

    static void xcsrmultd(const char * transa, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, fpType * a, DAAL_INT * ja, DAAL_INT * ia,
                          fpType * b, DAAL_INT * jb, DAAL_INT * ib, fpType * c, DAAL_INT * ldc)
    {
        services::throwIfPossible(services::Status(services::ErrorMethodNotImplemented));
    }

    static void xcsrmv(const char * transa, const DAAL_INT * m, const DAAL_INT * k, const fpType * alpha, const char * matdescra, const fpType * val,
                       const DAAL_INT * indx, const DAAL_INT * pntrb, const DAAL_INT * pntre, const fpType * x, const fpType * beta, fpType * y)
    {
        services::throwIfPossible(services::Status(services::ErrorMethodNotImplemented));
    }

    static void xcsrmm(const char * transa, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, const fpType * alpha, const char * matdescra,
                       const fpType * val, const DAAL_INT * indx, const DAAL_INT * pntrb, const fpType * b, const DAAL_INT * ldb, const fpType * beta,
                       fpType * c, const DAAL_INT * ldc)
    {
        csrmm(m, n, k, alpha, val, indx, pntrb, b, ldb, beta, c, ldc);
    }

    static void xxcsrmm(const char * transa, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, const fpType * alpha, const char * matdescra,
                        const fpType * val, const DAAL_INT * indx, const DAAL_INT * pntrb, const fpType * b, const DAAL_INT * ldb,
                        const fpType * beta, fpType * c, const DAAL_INT * ldc)
    {
        csrmm(m, n, k, alpha, val, indx, pntrb, b, ldb, beta, c, ldc);
    }
    inline static fpType csracc(DAAL_INT row, DAAL_INT col, const fpType * val, const DAAL_INT * indx, const DAAL_INT * pntrb)
    {
        DAAL_INT offset = pntrb[row] - 1;
        DAAL_INT nnz    = pntrb[row + 1] - pntrb[row];
        DAAL_INT csrcol = col + 1;
#pragma omp simd
        {
            for (DAAL_INT i = 0; i < nnz; ++i)
            {
                if (csrcol < indx[offset + i]) break;
                if (csrcol == indx[offset + i]) return val[offset + i];
            }
        }
        return fpType(0);
    }
    static void csrmm(const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, const fpType * alpha, const fpType * a, const DAAL_INT * indx,
                      const DAAL_INT * pntrb, const fpType * b, const DAAL_INT * ldb, const fpType * beta, fpType * c, const DAAL_INT * ldc)
    {
        DAAL_INT ldbVal = *ldb;
        DAAL_INT ldcVal = *ldc;
        DAAL_INT nVal   = *n;
        DAAL_INT mVal   = *m;
        DAAL_INT kVal   = *k;
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION_THROW_IF_POSSIBLE(DAAL_INT, ldbVal, nVal);
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION_THROW_IF_POSSIBLE(DAAL_INT, ldcVal, nVal);
        DAAL_OVERFLOW_CHECK_BY_ADDING_THROW_IF_POSSIBLE(DAAL_INT, ldbVal * ((*n) - 1), kVal);
        DAAL_OVERFLOW_CHECK_BY_ADDING_THROW_IF_POSSIBLE(DAAL_INT, ldcVal * ((*n) - 1), mVal);

        for (DAAL_INT mInd = 0; mInd < mVal; ++mInd)
            for (DAAL_INT nInd = 0; nInd < *n; ++nInd)
            {
                fpType sum = 0.0;
                for (DAAL_INT kInd = 0; kInd < kVal; ++kInd)
                {
                    fpType ail  = csracc(mInd, kInd, a, indx, pntrb);
                    fpType btlj = b[ldbVal * nInd + kInd];
                    sum += ail * btlj;
                }
                c[nInd * ldcVal + mInd] *= *beta;
                c[nInd * ldcVal + mInd] += *alpha * sum;
            }
    }
};

} // namespace ref
} // namespace internal
} // namespace daal

#endif
