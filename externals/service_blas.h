/* file: service_blas.h */
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
//  Template wrappers for BLAS functions.
//--
*/


#ifndef __SERVICE_BLAS_H__
#define __SERVICE_BLAS_H__

#include "daal_defines.h"
#include "service_memory.h"

#include "service_blas_mkl.h"

namespace daal
{
namespace internal
{

/*
// Template functions definition
*/
template<typename fpType, CpuType cpu, template<typename, CpuType> class _impl=mkl::MklBlas>
struct Blas
{
    typedef typename _impl<fpType,cpu>::SizeType SizeType;

    static void xsyrk(char *uplo, char *trans, SizeType *p, SizeType *n, fpType *alpha, fpType *a, SizeType *lda,
               fpType *beta, fpType *ata, SizeType *ldata)
    {
        _impl<fpType,cpu>::xsyrk(uplo, trans, p, n, alpha, a, lda, beta, ata, ldata);
    }

    static void xxsyrk(char *uplo, char *trans, SizeType *p, SizeType *n, fpType *alpha, fpType *a, SizeType *lda,
               fpType *beta, fpType *ata, SizeType *ldata)
    {
        _impl<fpType,cpu>::xxsyrk(uplo, trans, p, n, alpha, a, lda, beta, ata, ldata);
    }

    static void xsyr(const char *uplo, const SizeType *n, const fpType *alpha,
              const fpType *x, const SizeType *incx, fpType *a, const SizeType *lda)
    {
        _impl<fpType,cpu>::xsyr(uplo, n, alpha, x, incx, a, lda);
    }

    static void xxsyr(const char *uplo, const SizeType *n, const fpType *alpha,
              const fpType *x, const SizeType *incx, fpType *a, const SizeType *lda)
    {
        _impl<fpType,cpu>::xxsyr(uplo, n, alpha, x, incx, a, lda);
    }

    static void xgemm(char *transa, char *transb, SizeType *p, SizeType *ny, SizeType *n, fpType *alpha, fpType *a,
               SizeType *lda, fpType *y, SizeType *ldy, fpType *beta, fpType *aty, SizeType *ldaty)
    {
        _impl<fpType,cpu>::xgemm(transa, transb, p, ny, n, alpha, a, lda, y, ldy, beta, aty, ldaty);
    }

    static void xxgemm(char *transa, char *transb, SizeType *p, SizeType *ny, SizeType *n, fpType *alpha, fpType *a,
               SizeType *lda, fpType *y, SizeType *ldy, fpType *beta, fpType *aty, SizeType *ldaty)
    {
        _impl<fpType,cpu>::xxgemm(transa, transb, p, ny, n, alpha, a, lda, y, ldy, beta, aty, ldaty);
    }

    static void xsymm(char *side, char *uplo, SizeType *m, SizeType *n, fpType *alpha, fpType *a, SizeType *lda,
               fpType *b, SizeType *ldb, fpType *beta, fpType *c, SizeType *ldc)
    {
        _impl<fpType,cpu>::xsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    static void xxsymm(char *side, char *uplo, SizeType *m, SizeType *n, fpType *alpha, fpType *a, SizeType *lda,
               fpType *b, SizeType *ldb, fpType *beta, fpType *c, SizeType *ldc)
    {
        _impl<fpType,cpu>::xxsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    static void xgemv(char *trans, SizeType *m, SizeType *n, fpType *alpha, fpType *a, SizeType *lda, fpType *x,
               SizeType *incx, fpType *beta, fpType *y, SizeType *incy)
    {
        _impl<fpType,cpu>::xgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }

    static void xxgemv(char *trans, SizeType *m, SizeType *n, fpType *alpha, fpType *a, SizeType *lda, fpType *x,
               SizeType *incx, fpType *beta, fpType *y, SizeType *incy)
    {
        _impl<fpType,cpu>::xxgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }

    static void xaxpy(SizeType *n, fpType *a, fpType *x, SizeType *incx, fpType *y, SizeType *incy)
    {
        _impl<fpType,cpu>::xaxpy(n, a, x, incx, y, incy);
    }

    static void xxaxpy(SizeType *n, fpType *a, fpType *x, SizeType *incx, fpType *y, SizeType *incy)
    {
        _impl<fpType,cpu>::xxaxpy(n, a, x, incx, y, incy);
    }

};

} // namespace internal
} // namespace daal

#endif
