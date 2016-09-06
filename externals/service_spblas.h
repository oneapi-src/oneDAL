/* file: service_spblas.h */
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
//  Template wrappers for Sparse BLAS functions.
//--
*/


#ifndef __SERVICE_SPBLAS_H__
#define __SERVICE_SPBLAS_H__

#include "daal_defines.h"
#include "service_memory.h"

#include "service_spblas_mkl.h"

namespace daal
{
namespace internal
{

/*
// Template functions definition
*/
template<typename fpType, CpuType cpu, template<typename, CpuType> class _impl=mkl::MklSpBlas>
struct SpBlas
{
    typedef typename _impl<fpType,cpu>::SizeType SizeType;

    static void xsyrk(char *uplo, char *trans, SizeType *p, SizeType *n, fpType *alpha, fpType *a, SizeType *lda,
               fpType *beta, fpType *ata, SizeType *ldata)
    {
        _impl<fpType,cpu>::xsyrk(uplo, trans, p, n, alpha, a, lda, beta, ata, ldata);
    }

    static void xcsrmultd(const char *transa, const SizeType *m,
                   const SizeType *n, const SizeType *k, fpType *a, SizeType *ja, SizeType *ia,
                   fpType *b, SizeType *jb, SizeType *ib, fpType *c, SizeType *ldc)
    {
        _impl<fpType,cpu>::xcsrmultd(transa, m, n, k, a, ja, ia, b, jb, ib, c, ldc);
    }

    static void xcsrmv(const char *transa, const SizeType *m,
                const SizeType *k, const fpType *alpha, const char *matdescra,
                const fpType *val, const SizeType *indx, const SizeType *pntrb,
                const SizeType *pntre, const fpType *x, const fpType *beta, fpType *y)
    {
        _impl<fpType,cpu>::xcsrmv(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y);
    }

    static void xcsrmm(const char *transa, const SizeType *m, const SizeType *n, const SizeType *k,
                const fpType *alpha, const char *matdescra, const fpType *val, const SizeType *indx,
                const SizeType *pntrb, const fpType *b, const SizeType *ldb, const fpType *beta, fpType *c, const SizeType *ldc)
    {
        _impl<fpType,cpu>::xcsrmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, b, ldb, beta, c, ldc);
    }

    static void xxcsrmm(const char *transa, const SizeType *m, const SizeType *n, const SizeType *k,
                 const fpType *alpha, const char *matdescra, const fpType *val, const SizeType *indx,
                 const SizeType *pntrb, const fpType *b, const SizeType *ldb, const fpType *beta, fpType *c, const SizeType *ldc)
    {
        _impl<fpType,cpu>::xxcsrmm(transa, m, n, k, alpha, matdescra, val, indx, pntrb, b, ldb, beta, c, ldc);
    }
};

} // namespace internal
} // namespace daal

#endif
