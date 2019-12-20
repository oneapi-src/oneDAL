/* file: service_blas.h */
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
//  Template wrappers for BLAS functions.
//--
*/

#ifndef __SERVICE_BLAS_H__
#define __SERVICE_BLAS_H__

#include "daal_defines.h"
#include "service_memory.h"
#include "service_blas_mkl.h"
#include "service_error_handling.h"
#include "service_numeric_table.h"

namespace daal
{
namespace internal
{
/*
// Template functions definition
*/
template <typename fpType, CpuType cpu, template <typename, CpuType> class _impl = mkl::MklBlas>
struct Blas
{
    typedef typename _impl<fpType, cpu>::SizeType SizeType;

    static void xsyrk(char * uplo, char * trans, SizeType * p, SizeType * n, fpType * alpha, fpType * a, SizeType * lda, fpType * beta, fpType * ata,
                      SizeType * ldata)
    {
        _impl<fpType, cpu>::xsyrk(uplo, trans, p, n, alpha, a, lda, beta, ata, ldata);
    }

    static void xxsyrk(char * uplo, char * trans, SizeType * p, SizeType * n, fpType * alpha, fpType * a, SizeType * lda, fpType * beta, fpType * ata,
                       SizeType * ldata)
    {
        _impl<fpType, cpu>::xxsyrk(uplo, trans, p, n, alpha, a, lda, beta, ata, ldata);
    }

    static void xsyr(const char * uplo, const SizeType * n, const fpType * alpha, const fpType * x, const SizeType * incx, fpType * a,
                     const SizeType * lda)
    {
        _impl<fpType, cpu>::xsyr(uplo, n, alpha, x, incx, a, lda);
    }

    static void xxsyr(const char * uplo, const SizeType * n, const fpType * alpha, const fpType * x, const SizeType * incx, fpType * a,
                      const SizeType * lda)
    {
        _impl<fpType, cpu>::xxsyr(uplo, n, alpha, x, incx, a, lda);
    }

    static void xgemm(const char * transa, const char * transb, const SizeType * p, const SizeType * ny, const SizeType * n, const fpType * alpha,
                      const fpType * a, const SizeType * lda, const fpType * y, const SizeType * ldy, const fpType * beta, fpType * aty,
                      const SizeType * ldaty)
    {
        _impl<fpType, cpu>::xgemm(transa, transb, p, ny, n, alpha, a, lda, y, ldy, beta, aty, ldaty);
    }

    static void xxgemm(const char * transa, const char * transb, const SizeType * p, const SizeType * ny, const SizeType * n, const fpType * alpha,
                       const fpType * a, const SizeType * lda, const fpType * y, const SizeType * ldy, const fpType * beta, fpType * aty,
                       const SizeType * ldaty)
    {
        _impl<fpType, cpu>::xxgemm(transa, transb, p, ny, n, alpha, a, lda, y, ldy, beta, aty, ldaty);
    }

    static void xsymm(const char * side, const char * uplo, const SizeType * m, const SizeType * n, const fpType * alpha, const fpType * a,
                      const SizeType * lda, const fpType * b, const SizeType * ldb, const fpType * beta, fpType * c, const SizeType * ldc)
    {
        _impl<fpType, cpu>::xsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    static void xxsymm(char * side, char * uplo, SizeType * m, SizeType * n, fpType * alpha, fpType * a, SizeType * lda, fpType * b, SizeType * ldb,
                       fpType * beta, fpType * c, SizeType * ldc)
    {
        _impl<fpType, cpu>::xxsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    static void xgemv(const char * trans, const SizeType * m, const SizeType * n, const fpType * alpha, const fpType * a, const SizeType * lda,
                      const fpType * x, const SizeType * incx, const fpType * beta, fpType * y, const SizeType * incy)
    {
        _impl<fpType, cpu>::xgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }

    static void xxgemv(const char * trans, const SizeType * m, const SizeType * n, const fpType * alpha, const fpType * a, const SizeType * lda,
                       const fpType * x, const SizeType * incx, const fpType * beta, fpType * y, const SizeType * incy)
    {
        _impl<fpType, cpu>::xxgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }

    static void xaxpy(SizeType * n, fpType * a, fpType * x, SizeType * incx, fpType * y, SizeType * incy)
    {
        _impl<fpType, cpu>::xaxpy(n, a, x, incx, y, incy);
    }

    static void xxaxpy(const SizeType * n, const fpType * a, const fpType * x, const SizeType * incx, fpType * y, const SizeType * incy)
    {
        _impl<fpType, cpu>::xxaxpy(n, a, x, incx, y, incy);
    }

    static fpType xxdot(const SizeType * n, const fpType * x, const SizeType * incx, const fpType * y, const SizeType * incy)
    {
        return _impl<fpType, cpu>::xxdot(n, x, incx, y, incy);
    }

    static services::Status xgemm_blocked(const char * transa, const char * transb, const SizeType * na, const SizeType * nb, const SizeType * cols,
                                          const fpType * alpha, const NumericTable * ta, const SizeType * lda, const NumericTable * tb,
                                          const SizeType * ldb, const fpType * beta, NumericTable * tc, const SizeType * ldc,
                                          int a_req_blocksize = -1, int b_req_blocksize = -1)
    {
        SafeStatus safeStat;

        SizeType a_rows  = *na;
        SizeType b_rows  = *nb;
        SizeType ab_cols = *cols;

        /* Read block sizes from parameters or set to deafault value */
        int a_blocksize = (a_req_blocksize > 0) ? a_req_blocksize : 128;
        int b_blocksize = (b_req_blocksize > 0) ? b_req_blocksize : 128;

        /* Block size cannot be greater than whole number of rows */
        a_blocksize = (a_blocksize > a_rows) ? a_rows : a_blocksize;
        b_blocksize = (b_blocksize > b_rows) ? b_rows : b_blocksize;

        /* Number of blocks */
        SizeType a_blocknum = a_rows / a_blocksize;
        SizeType b_blocknum = b_rows / b_blocksize;

        /* Last block size */
        SizeType a_lastblocksize = a_rows - a_blocknum * a_blocksize;
        SizeType b_lastblocksize = b_rows - b_blocknum * b_blocksize;

        /* Increase the number of blocks if last block size is nonzero */
        if (a_lastblocksize != 0)
        {
            a_blocknum++;
        }
        else
        {
            a_lastblocksize = a_blocksize;
        }
        if (b_lastblocksize != 0)
        {
            b_blocknum++;
        }
        else
        {
            b_lastblocksize = b_blocksize;
        }

        /* Threaded loop by whole number of blocks */
        daal::threader_for(b_blocknum, b_blocknum, [&](SizeType b_block) {
            /* Current block size - can be less than general block size for last block */
            SizeType b_cursize = (b_block < (b_blocknum - 1)) ? b_blocksize : b_lastblocksize;

            ReadRows<fpType, cpu> mtb(*const_cast<NumericTable *>(tb), b_block * b_blocksize, b_cursize);
            DAAL_CHECK_BLOCK_STATUS_THR(mtb);
            const fpType * b = const_cast<fpType *>(mtb.get());

            /* Get pointer to write resulted rows */
            WriteOnlyRows<fpType, cpu> mtc(tc, b_block * b_blocksize, b_cursize);
            DAAL_CHECK_BLOCK_STATUS_THR(mtc);
            fpType * c = mtc.get();

            daal::threader_for(a_blocknum, a_blocknum, [&](SizeType a_block) {
                /* Current block size - can be less than general block size for last block */
                SizeType a_cursize = (a_block < (a_blocknum - 1)) ? a_blocksize : a_lastblocksize;

                /* Read rows for numeric tables */
                ReadRows<fpType, cpu> mta(*const_cast<NumericTable *>(ta), a_block * a_blocksize, a_cursize);
                DAAL_CHECK_BLOCK_STATUS_THR(mta);
                const fpType * a = const_cast<fpType *>(mta.get());

                /* Call to sequential GEMM */
                xxgemm(transa, transb, &a_cursize, &b_cursize, &ab_cols, alpha, a, lda, b, ldb, beta, c + a_block * a_blocksize, ldc);
            });
        });

        return safeStat.detach();
    } /* xgemm_blocked */
};

} // namespace internal
} // namespace daal

#endif
