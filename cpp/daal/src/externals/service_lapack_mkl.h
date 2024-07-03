/* file: service_lapack_mkl.h */
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
//  Template wrappers for common Intel(R) MKL functions.
//--
*/

#ifndef __SERVICE_LAPACK_MKL_H__
#define __SERVICE_LAPACK_MKL_H__

#include "services/daal_defines.h"
#include <mkl.h>
#include "mkl_daal.h"

namespace daal
{
namespace internal
{
namespace mkl
{
template <typename fpType, CpuType cpu>
struct MklLapack
{};

/*
// Double precision functions definition
*/

template <CpuType cpu>
struct MklLapack<double, cpu>
{
    typedef DAAL_INT SizeType;

    static void xgetrf(DAAL_INT * m, DAAL_INT * n, double * a, DAAL_INT * lda, DAAL_INT * ipiv, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dgetrf, ((MKL_INT *)m, (MKL_INT *)n, a, (MKL_INT *)lda, (MKL_INT *)ipiv, (MKL_INT *)info));
    }

    static void xxgetrf(DAAL_INT * m, DAAL_INT * n, double * a, DAAL_INT * lda, DAAL_INT * ipiv, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dgetrf, ((MKL_INT *)m, (MKL_INT *)n, a, (MKL_INT *)lda, (MKL_INT *)ipiv, (MKL_INT *)info));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xgetrs(char * trans, DAAL_INT * n, DAAL_INT * nrhs, double * a, DAAL_INT * lda, DAAL_INT * ipiv, double * b, DAAL_INT * ldb,
                       DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dgetrs,
                          (trans, (MKL_INT *)n, (MKL_INT *)nrhs, a, (MKL_INT *)lda, (MKL_INT *)ipiv, b, (MKL_INT *)ldb, (MKL_INT *)info, 1));
    }

    static void xxgetrs(char * trans, DAAL_INT * n, DAAL_INT * nrhs, double * a, DAAL_INT * lda, DAAL_INT * ipiv, double * b, DAAL_INT * ldb,
                        DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dgetrs,
                          (trans, (MKL_INT *)n, (MKL_INT *)nrhs, a, (MKL_INT *)lda, (MKL_INT *)ipiv, b, (MKL_INT *)ldb, (MKL_INT *)info, 1));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xpotrf(char * uplo, DAAL_INT * p, double * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dpotrf, (uplo, (MKL_INT *)p, ata, (MKL_INT *)ldata, (MKL_INT *)info, 1));
    }

    static void xxpotrf(char * uplo, DAAL_INT * p, double * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dpotrf, (uplo, (MKL_INT *)p, ata, (MKL_INT *)ldata, (MKL_INT *)info, 1));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xpotrs(char * uplo, DAAL_INT * p, DAAL_INT * ny, double * ata, DAAL_INT * ldata, double * beta, DAAL_INT * ldaty, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dpotrs, (uplo, (MKL_INT *)p, (MKL_INT *)ny, ata, (MKL_INT *)ldata, beta, (MKL_INT *)ldaty, (MKL_INT *)info, 1));
    }

    static void xxpotrs(char * uplo, DAAL_INT * p, DAAL_INT * ny, double * ata, DAAL_INT * ldata, double * beta, DAAL_INT * ldaty, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dpotrs, (uplo, (MKL_INT *)p, (MKL_INT *)ny, ata, (MKL_INT *)ldata, beta, (MKL_INT *)ldaty, (MKL_INT *)info, 1));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xpotri(char * uplo, DAAL_INT * p, double * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dpotri, (uplo, (MKL_INT *)p, ata, (MKL_INT *)ldata, (MKL_INT *)info, 1));
    }

    static void xxpotri(char * uplo, DAAL_INT * p, double * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dpotri, (uplo, (MKL_INT *)p, ata, (MKL_INT *)ldata, (MKL_INT *)info, 1));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xgerqf(DAAL_INT * m, DAAL_INT * n, double * a, DAAL_INT * lda, double * tau, double * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dgerqf, ((MKL_INT *)m, (MKL_INT *)n, a, (MKL_INT *)lda, tau, work, (MKL_INT *)lwork, (MKL_INT *)info));
    }

    static void xxgerqf(DAAL_INT * m, DAAL_INT * n, double * a, DAAL_INT * lda, double * tau, double * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dgerqf, ((MKL_INT *)m, (MKL_INT *)n, a, (MKL_INT *)lda, tau, work, (MKL_INT *)lwork, (MKL_INT *)info));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xormrq(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, double * a, DAAL_INT * lda, double * tau, double * c,
                       DAAL_INT * ldc, double * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dormrq,
                          (side, trans, (MKL_INT *)m, (MKL_INT *)n, (MKL_INT *)k, a, (MKL_INT *)lda, tau, c, (MKL_INT *)ldc, work, (MKL_INT *)lwork,
                           (MKL_INT *)info, 1, 1));
    }

    static void xxormrq(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, double * a, DAAL_INT * lda, double * tau, double * c,
                        DAAL_INT * ldc, double * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dormrq,
                          (side, trans, (MKL_INT *)m, (MKL_INT *)n, (MKL_INT *)k, a, (MKL_INT *)lda, tau, c, (MKL_INT *)ldc, work, (MKL_INT *)lwork,
                           (MKL_INT *)info, 1, 1));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xtrtrs(char * uplo, char * trans, char * diag, DAAL_INT * n, DAAL_INT * nrhs, double * a, DAAL_INT * lda, double * b, DAAL_INT * ldb,
                       DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dtrtrs, (uplo, trans, diag, (MKL_INT *)n, (MKL_INT *)nrhs, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, (MKL_INT *)info, 1, 1, 1));
    }

    static void xxtrtrs(char * uplo, char * trans, char * diag, DAAL_INT * n, DAAL_INT * nrhs, double * a, DAAL_INT * lda, double * b, DAAL_INT * ldb,
                        DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dtrtrs, (uplo, trans, diag, (MKL_INT *)n, (MKL_INT *)nrhs, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, (MKL_INT *)info, 1, 1, 1));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xpptrf(char * uplo, DAAL_INT * n, double * ap, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dpptrf, (uplo, (MKL_INT *)n, ap, (MKL_INT *)info, 1));
    }

    static void xxpptrf(char * uplo, DAAL_INT * n, double * ap, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dpptrf, (uplo, (MKL_INT *)n, ap, (MKL_INT *)info, 1));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xgeqrf(DAAL_INT m, DAAL_INT n, double * a, DAAL_INT lda, double * tau, double * work, DAAL_INT lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dgeqrf, ((MKL_INT *)(&m), (MKL_INT *)(&n), a, (MKL_INT *)(&lda), tau, work, (MKL_INT *)(&lwork), (MKL_INT *)info));
    }

    static void xxgeqrf(DAAL_INT m, DAAL_INT n, double * a, DAAL_INT lda, double * tau, double * work, DAAL_INT lwork, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dgeqrf, ((MKL_INT *)(&m), (MKL_INT *)(&n), a, (MKL_INT *)(&lda), tau, work, (MKL_INT *)(&lwork), (MKL_INT *)info));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xgeqp3(const DAAL_INT m, const DAAL_INT n, double * a, const DAAL_INT lda, DAAL_INT * jpvt, double * tau, double * work,
                       const DAAL_INT lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dgeqp3,
                          ((MKL_INT *)(&m), (MKL_INT *)(&n), a, (MKL_INT *)(&lda), (MKL_INT *)jpvt, tau, work, (MKL_INT *)(&lwork), (MKL_INT *)info));
    }

    static void xxgeqp3(DAAL_INT m, DAAL_INT n, double * a, DAAL_INT lda, DAAL_INT * jpvt, double * tau, double * work, DAAL_INT lwork,
                        DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dgeqp3,
                          ((MKL_INT *)(&m), (MKL_INT *)(&n), a, (MKL_INT *)(&lda), (MKL_INT *)jpvt, tau, work, (MKL_INT *)(&lwork), (MKL_INT *)info));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xorgqr(const DAAL_INT m, const DAAL_INT n, const DAAL_INT k, double * a, const DAAL_INT lda, const double * tau, double * work,
                       const DAAL_INT lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dorgqr,
                          ((MKL_INT *)(&m), (MKL_INT *)(&n), (MKL_INT *)(&k), a, (MKL_INT *)(&lda), tau, work, (MKL_INT *)(&lwork), (MKL_INT *)info));
    }

    static void xxorgqr(DAAL_INT m, DAAL_INT n, DAAL_INT k, double * a, DAAL_INT lda, double * tau, double * work, DAAL_INT lwork, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dorgqr,
                          ((MKL_INT *)(&m), (MKL_INT *)(&n), (MKL_INT *)(&k), a, (MKL_INT *)(&lda), tau, work, (MKL_INT *)(&lwork), (MKL_INT *)info));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xgesvd(char jobu, char jobvt, DAAL_INT m, DAAL_INT n, double * a, DAAL_INT lda, double * s, double * u, DAAL_INT ldu, double * vt,
                       DAAL_INT ldvt, double * work, DAAL_INT lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dgesvd,
                          (&jobu, &jobvt, (MKL_INT *)(&m), (MKL_INT *)(&n), a, (MKL_INT *)(&lda), s, u, (MKL_INT *)(&ldu), vt, (MKL_INT *)(&ldvt),
                           work, (MKL_INT *)(&lwork), (MKL_INT *)info, 1, 1));
    }

    static void xxgesvd(char jobu, char jobvt, DAAL_INT m, DAAL_INT n, double * a, DAAL_INT lda, double * s, double * u, DAAL_INT ldu, double * vt,
                        DAAL_INT ldvt, double * work, DAAL_INT lwork, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dgesvd,
                          (&jobu, &jobvt, (MKL_INT *)(&m), (MKL_INT *)(&n), a, (MKL_INT *)(&lda), s, u, (MKL_INT *)(&ldu), vt, (MKL_INT *)(&ldvt),
                           work, (MKL_INT *)(&lwork), (MKL_INT *)info, 1, 1));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xsyevd(char * jobz, char * uplo, DAAL_INT * n, double * a, DAAL_INT * lda, double * w, double * work, DAAL_INT * lwork,
                       DAAL_INT * iwork, DAAL_INT * liwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(
            lapack_, dsyevd,
            (jobz, uplo, (MKL_INT *)n, a, (MKL_INT *)lda, w, work, (MKL_INT *)lwork, (MKL_INT *)iwork, (MKL_INT *)liwork, (MKL_INT *)info, 1, 1));
    }

    static void xxsyevd(char * jobz, char * uplo, DAAL_INT * n, double * a, DAAL_INT * lda, double * w, double * work, DAAL_INT * lwork,
                        DAAL_INT * iwork, DAAL_INT * liwork, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(
            lapack_, dsyevd,
            (jobz, uplo, (MKL_INT *)n, a, (MKL_INT *)lda, w, work, (MKL_INT *)lwork, (MKL_INT *)iwork, (MKL_INT *)liwork, (MKL_INT *)info, 1, 1));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xormqr(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, double * a, DAAL_INT * lda, double * tau, double * c,
                       DAAL_INT * ldc, double * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dormqr,
                          (side, trans, (MKL_INT *)m, (MKL_INT *)n, (MKL_INT *)k, a, (MKL_INT *)lda, tau, c, (MKL_INT *)ldc, work, (MKL_INT *)lwork,
                           (MKL_INT *)info, 1, 1));
    }

    static void xxormqr(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, double * a, DAAL_INT * lda, double * tau, double * c,
                        DAAL_INT * ldc, double * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dormqr,
                          (side, trans, (MKL_INT *)m, (MKL_INT *)n, (MKL_INT *)k, a, (MKL_INT *)lda, tau, c, (MKL_INT *)ldc, work, (MKL_INT *)lwork,
                           (MKL_INT *)info, 1, 1));
        mkl_serv_set_num_threads_local(old_threads);
    }
};

/*
// Single precision functions definition
*/

template <CpuType cpu>
struct MklLapack<float, cpu>
{
    typedef DAAL_INT SizeType;

    static void xgetrf(DAAL_INT * m, DAAL_INT * n, float * a, DAAL_INT * lda, DAAL_INT * ipiv, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, sgetrf, ((MKL_INT *)m, (MKL_INT *)n, a, (MKL_INT *)lda, (MKL_INT *)ipiv, (MKL_INT *)info));
    }

    static void xxgetrf(DAAL_INT * m, DAAL_INT * n, float * a, DAAL_INT * lda, DAAL_INT * ipiv, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, sgetrf, ((MKL_INT *)m, (MKL_INT *)n, a, (MKL_INT *)lda, (MKL_INT *)ipiv, (MKL_INT *)info));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xgetrs(char * trans, DAAL_INT * n, DAAL_INT * nrhs, float * a, DAAL_INT * lda, DAAL_INT * ipiv, float * b, DAAL_INT * ldb,
                       DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, sgetrs,
                          (trans, (MKL_INT *)n, (MKL_INT *)nrhs, a, (MKL_INT *)lda, (MKL_INT *)ipiv, b, (MKL_INT *)ldb, (MKL_INT *)info, 1));
    }

    static void xxgetrs(char * trans, DAAL_INT * n, DAAL_INT * nrhs, float * a, DAAL_INT * lda, DAAL_INT * ipiv, float * b, DAAL_INT * ldb,
                        DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, sgetrs,
                          (trans, (MKL_INT *)n, (MKL_INT *)nrhs, a, (MKL_INT *)lda, (MKL_INT *)ipiv, b, (MKL_INT *)ldb, (MKL_INT *)info, 1));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xpotrf(char * uplo, DAAL_INT * p, float * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, spotrf, (uplo, (MKL_INT *)p, ata, (MKL_INT *)ldata, (MKL_INT *)info, 1));
    }

    static void xxpotrf(char * uplo, DAAL_INT * p, float * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, spotrf, (uplo, (MKL_INT *)p, ata, (MKL_INT *)ldata, (MKL_INT *)info, 1));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xpotrs(char * uplo, DAAL_INT * p, DAAL_INT * ny, float * ata, DAAL_INT * ldata, float * beta, DAAL_INT * ldaty, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, spotrs, (uplo, (MKL_INT *)p, (MKL_INT *)ny, ata, (MKL_INT *)ldata, beta, (MKL_INT *)ldaty, (MKL_INT *)info, 1));
    }

    static void xxpotrs(char * uplo, DAAL_INT * p, DAAL_INT * ny, float * ata, DAAL_INT * ldata, float * beta, DAAL_INT * ldaty, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, spotrs, (uplo, (MKL_INT *)p, (MKL_INT *)ny, ata, (MKL_INT *)ldata, beta, (MKL_INT *)ldaty, (MKL_INT *)info, 1));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xpotri(char * uplo, DAAL_INT * p, float * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, spotri, (uplo, (MKL_INT *)p, ata, (MKL_INT *)ldata, (MKL_INT *)info, 1));
    }

    static void xxpotri(char * uplo, DAAL_INT * p, float * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, spotri, (uplo, (MKL_INT *)p, ata, (MKL_INT *)ldata, (MKL_INT *)info, 1));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xgerqf(DAAL_INT * m, DAAL_INT * n, float * a, DAAL_INT * lda, float * tau, float * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, sgerqf, ((MKL_INT *)m, (MKL_INT *)n, a, (MKL_INT *)lda, tau, work, (MKL_INT *)lwork, (MKL_INT *)info));
    }

    static void xxgerqf(DAAL_INT * m, DAAL_INT * n, float * a, DAAL_INT * lda, float * tau, float * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, sgerqf, ((MKL_INT *)m, (MKL_INT *)n, a, (MKL_INT *)lda, tau, work, (MKL_INT *)lwork, (MKL_INT *)info));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xormrq(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, float * a, DAAL_INT * lda, float * tau, float * c,
                       DAAL_INT * ldc, float * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, sormrq,
                          (side, trans, (MKL_INT *)m, (MKL_INT *)n, (MKL_INT *)k, a, (MKL_INT *)lda, tau, c, (MKL_INT *)ldc, work, (MKL_INT *)lwork,
                           (MKL_INT *)info, 1, 1));
    }

    static void xxormrq(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, float * a, DAAL_INT * lda, float * tau, float * c,
                        DAAL_INT * ldc, float * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, sormrq,
                          (side, trans, (MKL_INT *)m, (MKL_INT *)n, (MKL_INT *)k, a, (MKL_INT *)lda, tau, c, (MKL_INT *)ldc, work, (MKL_INT *)lwork,
                           (MKL_INT *)info, 1, 1));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xtrtrs(char * uplo, char * trans, char * diag, DAAL_INT * n, DAAL_INT * nrhs, float * a, DAAL_INT * lda, float * b, DAAL_INT * ldb,
                       DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, strtrs, (uplo, trans, diag, (MKL_INT *)n, (MKL_INT *)nrhs, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, (MKL_INT *)info, 1, 1, 1));
    }

    static void xxtrtrs(char * uplo, char * trans, char * diag, DAAL_INT * n, DAAL_INT * nrhs, float * a, DAAL_INT * lda, float * b, DAAL_INT * ldb,
                        DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, strtrs, (uplo, trans, diag, (MKL_INT *)n, (MKL_INT *)nrhs, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, (MKL_INT *)info, 1, 1, 1));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xpptrf(char * uplo, DAAL_INT * n, float * ap, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, spptrf, (uplo, (MKL_INT *)n, ap, (MKL_INT *)info, 1));
    }

    static void xxpptrf(char * uplo, DAAL_INT * n, float * ap, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, spptrf, (uplo, (MKL_INT *)n, ap, (MKL_INT *)info, 1));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xgeqrf(DAAL_INT m, DAAL_INT n, float * a, DAAL_INT lda, float * tau, float * work, DAAL_INT lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, sgeqrf, ((MKL_INT *)(&m), (MKL_INT *)(&n), a, (MKL_INT *)(&lda), tau, work, (MKL_INT *)(&lwork), (MKL_INT *)info));
    }

    static void xxgeqrf(DAAL_INT m, DAAL_INT n, float * a, DAAL_INT lda, float * tau, float * work, DAAL_INT lwork, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, sgeqrf, ((MKL_INT *)(&m), (MKL_INT *)(&n), a, (MKL_INT *)(&lda), tau, work, (MKL_INT *)(&lwork), (MKL_INT *)info));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xgeqp3(const DAAL_INT m, const DAAL_INT n, float * a, const DAAL_INT lda, DAAL_INT * jpvt, float * tau, float * work,
                       const DAAL_INT lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, sgeqp3,
                          ((MKL_INT *)(&m), (MKL_INT *)(&n), a, (MKL_INT *)(&lda), (MKL_INT *)jpvt, tau, work, (MKL_INT *)(&lwork), (MKL_INT *)info));
    }

    static void xxgeqp3(DAAL_INT m, DAAL_INT n, float * a, DAAL_INT lda, DAAL_INT * jpvt, float * tau, float * work, DAAL_INT lwork, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, sgeqp3,
                          ((MKL_INT *)(&m), (MKL_INT *)(&n), a, (MKL_INT *)(&lda), (MKL_INT *)jpvt, tau, work, (MKL_INT *)(&lwork), (MKL_INT *)info));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xorgqr(const DAAL_INT m, const DAAL_INT n, const DAAL_INT k, float * a, const DAAL_INT lda, const float * tau, float * work,
                       const DAAL_INT lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, sorgqr,
                          ((MKL_INT *)(&m), (MKL_INT *)(&n), (MKL_INT *)(&k), a, (MKL_INT *)(&lda), tau, work, (MKL_INT *)(&lwork), (MKL_INT *)info));
    }

    static void xxorgqr(DAAL_INT m, DAAL_INT n, DAAL_INT k, float * a, DAAL_INT lda, float * tau, float * work, DAAL_INT lwork, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, sorgqr,
                          ((MKL_INT *)(&m), (MKL_INT *)(&n), (MKL_INT *)(&k), a, (MKL_INT *)(&lda), tau, work, (MKL_INT *)(&lwork), (MKL_INT *)info));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xgesvd(char jobu, char jobvt, DAAL_INT m, DAAL_INT n, float * a, DAAL_INT lda, float * s, float * u, DAAL_INT ldu, float * vt,
                       DAAL_INT ldvt, float * work, DAAL_INT lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, sgesvd,
                          (&jobu, &jobvt, (MKL_INT *)(&m), (MKL_INT *)(&n), a, (MKL_INT *)(&lda), s, u, (MKL_INT *)(&ldu), vt, (MKL_INT *)(&ldvt),
                           work, (MKL_INT *)(&lwork), (MKL_INT *)info, 1, 1));
    }

    static void xxgesvd(char jobu, char jobvt, DAAL_INT m, DAAL_INT n, float * a, DAAL_INT lda, float * s, float * u, DAAL_INT ldu, float * vt,
                        DAAL_INT ldvt, float * work, DAAL_INT lwork, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, sgesvd,
                          (&jobu, &jobvt, (MKL_INT *)(&m), (MKL_INT *)(&n), a, (MKL_INT *)(&lda), s, u, (MKL_INT *)(&ldu), vt, (MKL_INT *)(&ldvt),
                           work, (MKL_INT *)(&lwork), (MKL_INT *)info, 1, 1));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xsyevd(char * jobz, char * uplo, DAAL_INT * n, float * a, DAAL_INT * lda, float * w, float * work, DAAL_INT * lwork, DAAL_INT * iwork,
                       DAAL_INT * liwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(
            lapack_, ssyevd,
            (jobz, uplo, (MKL_INT *)n, a, (MKL_INT *)lda, w, work, (MKL_INT *)lwork, (MKL_INT *)iwork, (MKL_INT *)liwork, (MKL_INT *)info, 1, 1));
    }

    static void xxsyevd(char * jobz, char * uplo, DAAL_INT * n, float * a, DAAL_INT * lda, float * w, float * work, DAAL_INT * lwork,
                        DAAL_INT * iwork, DAAL_INT * liwork, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(
            lapack_, ssyevd,
            (jobz, uplo, (MKL_INT *)n, a, (MKL_INT *)lda, w, work, (MKL_INT *)lwork, (MKL_INT *)iwork, (MKL_INT *)liwork, (MKL_INT *)info, 1, 1));
        mkl_serv_set_num_threads_local(old_threads);
    }

    static void xormqr(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, float * a, DAAL_INT * lda, float * tau, float * c,
                       DAAL_INT * ldc, float * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, sormqr,
                          (side, trans, (MKL_INT *)m, (MKL_INT *)n, (MKL_INT *)k, a, (MKL_INT *)lda, tau, c, (MKL_INT *)ldc, work, (MKL_INT *)lwork,
                           (MKL_INT *)info, 1, 1));
    }

    static void xxormqr(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, float * a, DAAL_INT * lda, float * tau, float * c,
                        DAAL_INT * ldc, float * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        int old_threads = mkl_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, sormqr,
                          (side, trans, (MKL_INT *)m, (MKL_INT *)n, (MKL_INT *)k, a, (MKL_INT *)lda, tau, c, (MKL_INT *)ldc, work, (MKL_INT *)lwork,
                           (MKL_INT *)info, 1, 1));
        mkl_serv_set_num_threads_local(old_threads);
    }
};

} // namespace mkl
} // namespace internal
} // namespace daal

#endif
