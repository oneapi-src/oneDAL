/* file: service_lapack.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Template wrappers for LAPACK functions.
//--
*/


#ifndef __SERVICE_LAPACK_H__
#define __SERVICE_LAPACK_H__

#include "daal_defines.h"
#include "service_memory.h"

#include "service_lapack_mkl.h"

namespace daal
{
namespace internal
{

/*
// Template functions definition
*/
template<typename fpType, CpuType cpu, template<typename, CpuType> class _impl=mkl::MklLapack>
struct Lapack
{
    typedef typename _impl<fpType,cpu>::SizeType SizeType;

    static void xpotrf(char *uplo, SizeType *p, fpType *ata, SizeType *ldata, SizeType *info)
    {
        _impl<fpType,cpu>::xpotrf(uplo, p, ata, ldata, info);
    }

    static void xxpotrf(char *uplo, SizeType *p, fpType *ata, SizeType *ldata, SizeType *info)
    {
        _impl<fpType,cpu>::xxpotrf(uplo, p, ata, ldata, info);
    }

    static void xpotrs(char *uplo, SizeType *p, SizeType *ny, fpType *ata, SizeType *ldata, fpType *beta, SizeType *ldaty,
                SizeType *info)
    {
        _impl<fpType,cpu>::xpotrs(uplo, p, ny, ata, ldata, beta, ldaty, info);
    }

    static void xxpotrs(char *uplo, SizeType *p, SizeType *ny, fpType *ata, SizeType *ldata, fpType *beta, SizeType *ldaty,
                SizeType *info)
    {
        _impl<fpType,cpu>::xxpotrs(uplo, p, ny, ata, ldata, beta, ldaty, info);
    }

    static void xpotri(char *uplo, SizeType *p, fpType *ata, SizeType *ldata, SizeType *info)
    {
        _impl<fpType,cpu>::xpotri(uplo, p, ata, ldata, info);
    }

    static void xxpotri(char *uplo, SizeType *p, fpType *ata, SizeType *ldata, SizeType *info)
    {
        _impl<fpType,cpu>::xxpotri(uplo, p, ata, ldata, info);
    }

    static void xgerqf(SizeType *m, SizeType *n, fpType *a, SizeType *lda, fpType *tau, fpType *work, SizeType *lwork,
                SizeType *info)
    {
        _impl<fpType,cpu>::xgerqf(m, n, a, lda, tau, work, lwork, info);
    }

    static void xxgerqf(SizeType *m, SizeType *n, fpType *a, SizeType *lda, fpType *tau, fpType *work, SizeType *lwork,
                SizeType *info)
    {
        _impl<fpType,cpu>::xxgerqf(m, n, a, lda, tau, work, lwork, info);
    }

    static void xormrq(char *side, char *trans, SizeType *m, SizeType *n, SizeType *k, fpType *a, SizeType *lda,
                fpType *tau, fpType *c, SizeType *ldc, fpType *work, SizeType *lwork, SizeType *info)
    {
        _impl<fpType,cpu>::xormrq(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
    }

    static void xxormrq(char *side, char *trans, SizeType *m, SizeType *n, SizeType *k, fpType *a, SizeType *lda,
                fpType *tau, fpType *c, SizeType *ldc, fpType *work, SizeType *lwork, SizeType *info)
    {
        _impl<fpType,cpu>::xxormrq(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
    }

    static void xtrtrs(char *uplo, char *trans, char *diag, SizeType *n, SizeType *nrhs,
                fpType *a, SizeType *lda, fpType *b, SizeType *ldb, SizeType *info)
    {
        _impl<fpType,cpu>::xtrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
    }

    static void xxtrtrs(char *uplo, char *trans, char *diag, SizeType *n, SizeType *nrhs,
                fpType *a, SizeType *lda, fpType *b, SizeType *ldb, SizeType *info)
    {
        _impl<fpType,cpu>::xxtrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info);
    }

    static void xpptrf(char *uplo, SizeType *n, fpType *ap, SizeType *info)
    {
        _impl<fpType,cpu>::xpptrf(uplo, n, ap, info);
    }

    static void xxpptrf(char *uplo, SizeType *n, fpType *ap, SizeType *info)
    {
        _impl<fpType,cpu>::xxpptrf(uplo, n, ap, info);
    }

    static void xgeqrf(SizeType m, SizeType n, fpType *a,
                SizeType lda, fpType *tau, fpType *work, SizeType lwork, SizeType *info)
    {
        _impl<fpType,cpu>::xgeqrf(m, n, a, lda, tau, work, lwork, info);
    }

    static void xxgeqrf(SizeType m, SizeType n, fpType *a,
                SizeType lda, fpType *tau, fpType *work, SizeType lwork, SizeType *info)
    {
        _impl<fpType,cpu>::xxgeqrf(m, n, a, lda, tau, work, lwork, info);
    }

    static void xgeqp3(const SizeType m, const SizeType n, fpType *a,
                const SizeType lda, SizeType *jpvt, fpType *tau, fpType *work, const SizeType lwork, SizeType *info)
    {
        _impl<fpType,cpu>::xgeqp3(m, n, a, lda, jpvt, tau, work, lwork, info);
    }

    static void xxgeqp3(SizeType m, SizeType n, fpType *a,
                SizeType lda, SizeType *jpvt, fpType *tau, fpType *work, SizeType lwork, SizeType *info)
    {
        _impl<fpType,cpu>::xxgeqp3(m, n, a, lda, jpvt, tau, work, lwork, info);
    }

    static void xorgqr(const SizeType m, const SizeType n, const SizeType k,
                fpType *a, const SizeType lda, const fpType *tau, fpType *work, const SizeType lwork, SizeType *info)
    {
        _impl<fpType,cpu>::xorgqr(m, n, k, a, lda, tau, work, lwork, info);
    }

    static void xxorgqr(SizeType m, SizeType n, SizeType k,
                fpType *a, SizeType lda, fpType *tau, fpType *work, SizeType lwork, SizeType *info)
    {
        _impl<fpType,cpu>::xxorgqr(m, n, k, a, lda, tau, work, lwork, info);
    }

    static void xgesvd(char jobu, char jobvt, SizeType m, SizeType n,
                fpType *a, SizeType lda, fpType *s, fpType *u, SizeType ldu, fpType *vt, SizeType ldvt,
                fpType *work, SizeType lwork, SizeType *info)
    {
        _impl<fpType,cpu>::xgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
    }

    static void xxgesvd(char jobu, char jobvt, SizeType m, SizeType n,
                fpType *a, SizeType lda, fpType *s, fpType *u, SizeType ldu, fpType *vt, SizeType ldvt,
                fpType *work, SizeType lwork, SizeType *info)
    {
        _impl<fpType,cpu>::xxgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
    }

    static void xsyevd(char *jobz, char *uplo, SizeType *n, fpType *a, SizeType *lda, fpType *w, fpType *work,
                SizeType *lwork, SizeType *iwork, SizeType *liwork, SizeType *info)
    {
        _impl<fpType,cpu>::xsyevd(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
    }

    static void xxsyevd(char *jobz, char *uplo, SizeType *n, fpType *a, SizeType *lda, fpType *w, fpType *work,
                SizeType *lwork, SizeType *iwork, SizeType *liwork, SizeType *info)
    {
        _impl<fpType,cpu>::xxsyevd(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
    }

    static void xormqr(char *side, char *trans, SizeType *m, SizeType *n, SizeType *k,
                fpType *a, SizeType *lda, fpType *tau, fpType *c, SizeType *ldc, fpType *work, SizeType *lwork, SizeType *info)
    {
        _impl<fpType,cpu>::xormqr(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
    }

    static void xxormqr(char *side, char *trans, SizeType *m, SizeType *n, SizeType *k,
                fpType *a, SizeType *lda, fpType *tau, fpType *c, SizeType *ldc, fpType *work, SizeType *lwork, SizeType *info)
    {
        _impl<fpType,cpu>::xxormqr(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
    }


};

} // namespace internal
} // namespace daal

#endif
