/* file: service_lapack_mkl.h */
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
//  Template wrappers for common Intel(R) MKL functions.
//--
*/

#ifndef __SERVICE_LAPACK_MKL_H__
#define __SERVICE_LAPACK_MKL_H__

#include "daal_defines.h"
#include "mkl_daal.h"

#if !defined(__DAAL_CONCAT4)
    #define __DAAL_CONCAT4(a, b, c, d)  __DAAL_CONCAT41(a, b, c, d)
    #define __DAAL_CONCAT41(a, b, c, d) a##b##c##d
#endif

#if !defined(__DAAL_CONCAT5)
    #define __DAAL_CONCAT5(a, b, c, d, e)  __DAAL_CONCAT51(a, b, c, d, e)
    #define __DAAL_CONCAT51(a, b, c, d, e) a##b##c##d##e
#endif

#if defined(__APPLE__)
    #define __DAAL_MKL_SSE2  ssse3_
    #define __DAAL_MKL_SSSE3 ssse3_
#else
    #define __DAAL_MKL_SSE2  sse2_
    #define __DAAL_MKL_SSSE3 ssse3_
#endif

#define __DAAL_MKLFN(f_cpu, f_pref, f_name)              __DAAL_CONCAT4(fpk_, f_pref, f_cpu, f_name)
#define __DAAL_MKLFN_CALL(f_pref, f_name, f_args)        __DAAL_MKLFN_CALL1(f_pref, f_name, f_args)
#define __DAAL_MKLFN_CALL_RETURN(f_pref, f_name, f_args) __DAAL_MKLFN_CALL2(f_pref, f_name, f_args)

#if (defined(__x86_64__) && !defined(__APPLE__)) || defined(_WIN64)
    #define __DAAL_MKLFPK_KNL avx512_mic_
#else
    #define __DAAL_MKLFPK_KNL avx2_
#endif

#define __DAAL_MKLFN_CALL1(f_pref, f_name, f_args)              \
    if (avx512 == cpu)                                          \
    {                                                           \
        __DAAL_MKLFN(avx512_, f_pref, f_name) f_args;           \
    }                                                           \
    if (avx512_mic == cpu)                                      \
    {                                                           \
        __DAAL_MKLFN(__DAAL_MKLFPK_KNL, f_pref, f_name) f_args; \
    }                                                           \
    if (avx2 == cpu)                                            \
    {                                                           \
        __DAAL_MKLFN(avx2_, f_pref, f_name) f_args;             \
    }                                                           \
    if (avx == cpu)                                             \
    {                                                           \
        __DAAL_MKLFN(avx_, f_pref, f_name) f_args;              \
    }                                                           \
    if (sse42 == cpu)                                           \
    {                                                           \
        __DAAL_MKLFN(sse42_, f_pref, f_name) f_args;            \
    }                                                           \
    if (ssse3 == cpu)                                           \
    {                                                           \
        __DAAL_MKLFN(__DAAL_MKL_SSSE3, f_pref, f_name) f_args;  \
    }                                                           \
    if (sse2 == cpu)                                            \
    {                                                           \
        __DAAL_MKLFN(__DAAL_MKL_SSE2, f_pref, f_name) f_args;   \
    }

#define __DAAL_MKLFN_CALL2(f_pref, f_name, f_args)                     \
    if (avx512 == cpu)                                                 \
    {                                                                  \
        return __DAAL_MKLFN(avx512_, f_pref, f_name) f_args;           \
    }                                                                  \
    if (avx512_mic == cpu)                                             \
    {                                                                  \
        return __DAAL_MKLFN(__DAAL_MKLFPK_KNL, f_pref, f_name) f_args; \
    }                                                                  \
    if (avx2 == cpu)                                                   \
    {                                                                  \
        return __DAAL_MKLFN(avx2_, f_pref, f_name) f_args;             \
    }                                                                  \
    if (avx == cpu)                                                    \
    {                                                                  \
        return __DAAL_MKLFN(avx_, f_pref, f_name) f_args;              \
    }                                                                  \
    if (sse42 == cpu)                                                  \
    {                                                                  \
        return __DAAL_MKLFN(sse42_, f_pref, f_name) f_args;            \
    }                                                                  \
    if (ssse3 == cpu)                                                  \
    {                                                                  \
        return __DAAL_MKLFN(__DAAL_MKL_SSSE3, f_pref, f_name) f_args;  \
    }                                                                  \
    if (sse2 == cpu)                                                   \
    {                                                                  \
        return __DAAL_MKLFN(__DAAL_MKL_SSE2, f_pref, f_name) f_args;   \
    }

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

    static void xpotrf(char * uplo, DAAL_INT * p, double * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dpotrf, (uplo, p, ata, ldata, info, 1));
    }

    static void xxpotrf(char * uplo, DAAL_INT * p, double * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dpotrf, (uplo, p, ata, ldata, info, 1));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xpotrs(char * uplo, DAAL_INT * p, DAAL_INT * ny, double * ata, DAAL_INT * ldata, double * beta, DAAL_INT * ldaty, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dpotrs, (uplo, p, ny, ata, ldata, beta, ldaty, info, 1));
    }

    static void xxpotrs(char * uplo, DAAL_INT * p, DAAL_INT * ny, double * ata, DAAL_INT * ldata, double * beta, DAAL_INT * ldaty, DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dpotrs, (uplo, p, ny, ata, ldata, beta, ldaty, info, 1));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xpotri(char * uplo, DAAL_INT * p, double * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dpotri, (uplo, p, ata, ldata, info, 1));
    }

    static void xxpotri(char * uplo, DAAL_INT * p, double * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dpotri, (uplo, p, ata, ldata, info, 1));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xgerqf(DAAL_INT * m, DAAL_INT * n, double * a, DAAL_INT * lda, double * tau, double * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dgerqf, (m, n, a, lda, tau, work, lwork, info));
    }

    static void xxgerqf(DAAL_INT * m, DAAL_INT * n, double * a, DAAL_INT * lda, double * tau, double * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dgerqf, (m, n, a, lda, tau, work, lwork, info));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xormrq(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, double * a, DAAL_INT * lda, double * tau, double * c,
                       DAAL_INT * ldc, double * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dormrq, (side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info, 1, 1));
    }

    static void xxormrq(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, double * a, DAAL_INT * lda, double * tau, double * c,
                        DAAL_INT * ldc, double * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dormrq, (side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info, 1, 1));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xtrtrs(char * uplo, char * trans, char * diag, DAAL_INT * n, DAAL_INT * nrhs, double * a, DAAL_INT * lda, double * b, DAAL_INT * ldb,
                       DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dtrtrs, (uplo, trans, diag, n, nrhs, a, lda, b, ldb, info, 1, 1, 1));
    }

    static void xxtrtrs(char * uplo, char * trans, char * diag, DAAL_INT * n, DAAL_INT * nrhs, double * a, DAAL_INT * lda, double * b, DAAL_INT * ldb,
                        DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dtrtrs, (uplo, trans, diag, n, nrhs, a, lda, b, ldb, info, 1, 1, 1));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xpptrf(char * uplo, DAAL_INT * n, double * ap, DAAL_INT * info) { __DAAL_MKLFN_CALL(lapack_, dpptrf, (uplo, n, ap, info, 1)); }

    static void xxpptrf(char * uplo, DAAL_INT * n, double * ap, DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dpptrf, (uplo, n, ap, info, 1));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xgeqrf(DAAL_INT m, DAAL_INT n, double * a, DAAL_INT lda, double * tau, double * work, DAAL_INT lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dgeqrf, (&m, &n, a, &lda, tau, work, &lwork, info));
    }

    static void xxgeqrf(DAAL_INT m, DAAL_INT n, double * a, DAAL_INT lda, double * tau, double * work, DAAL_INT lwork, DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dgeqrf, (&m, &n, a, &lda, tau, work, &lwork, info));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xgeqp3(const DAAL_INT m, const DAAL_INT n, double * a, const DAAL_INT lda, DAAL_INT * jpvt, double * tau, double * work,
                       const DAAL_INT lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dgeqp3, (&m, &n, a, &lda, jpvt, tau, work, &lwork, info));
    }

    static void xxgeqp3(DAAL_INT m, DAAL_INT n, double * a, DAAL_INT lda, DAAL_INT * jpvt, double * tau, double * work, DAAL_INT lwork,
                        DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dgeqp3, (&m, &n, a, &lda, jpvt, tau, work, &lwork, info));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xorgqr(const DAAL_INT m, const DAAL_INT n, const DAAL_INT k, double * a, const DAAL_INT lda, const double * tau, double * work,
                       const DAAL_INT lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dorgqr, (&m, &n, &k, a, &lda, tau, work, &lwork, info));
    }

    static void xxorgqr(DAAL_INT m, DAAL_INT n, DAAL_INT k, double * a, DAAL_INT lda, double * tau, double * work, DAAL_INT lwork, DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dorgqr, (&m, &n, &k, a, &lda, tau, work, &lwork, info));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xgesvd(char jobu, char jobvt, DAAL_INT m, DAAL_INT n, double * a, DAAL_INT lda, double * s, double * u, DAAL_INT ldu, double * vt,
                       DAAL_INT ldvt, double * work, DAAL_INT lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dgesvd, (&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, info, 1, 1));
    }

    static void xxgesvd(char jobu, char jobvt, DAAL_INT m, DAAL_INT n, double * a, DAAL_INT lda, double * s, double * u, DAAL_INT ldu, double * vt,
                        DAAL_INT ldvt, double * work, DAAL_INT lwork, DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dgesvd, (&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, info, 1, 1));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xsyevd(char * jobz, char * uplo, DAAL_INT * n, double * a, DAAL_INT * lda, double * w, double * work, DAAL_INT * lwork,
                       DAAL_INT * iwork, DAAL_INT * liwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dsyevd, (jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info, 1, 1));
    }

    static void xxsyevd(char * jobz, char * uplo, DAAL_INT * n, double * a, DAAL_INT * lda, double * w, double * work, DAAL_INT * lwork,
                        DAAL_INT * iwork, DAAL_INT * liwork, DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dsyevd, (jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info, 1, 1));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xormqr(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, double * a, DAAL_INT * lda, double * tau, double * c,
                       DAAL_INT * ldc, double * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, dormqr, (side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info, 1, 1));
    }

    static void xxormqr(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, double * a, DAAL_INT * lda, double * tau, double * c,
                        DAAL_INT * ldc, double * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, dormqr, (side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info, 1, 1));
        fpk_serv_set_num_threads_local(old_threads);
    }
};

/*
// Single precision functions definition
*/

template <CpuType cpu>
struct MklLapack<float, cpu>
{
    typedef DAAL_INT SizeType;

    static void xpotrf(char * uplo, DAAL_INT * p, float * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, spotrf, (uplo, p, ata, ldata, info, 1));
    }

    static void xxpotrf(char * uplo, DAAL_INT * p, float * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, spotrf, (uplo, p, ata, ldata, info, 1));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xpotrs(char * uplo, DAAL_INT * p, DAAL_INT * ny, float * ata, DAAL_INT * ldata, float * beta, DAAL_INT * ldaty, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, spotrs, (uplo, p, ny, ata, ldata, beta, ldaty, info, 1));
    }

    static void xxpotrs(char * uplo, DAAL_INT * p, DAAL_INT * ny, float * ata, DAAL_INT * ldata, float * beta, DAAL_INT * ldaty, DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, spotrs, (uplo, p, ny, ata, ldata, beta, ldaty, info, 1));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xpotri(char * uplo, DAAL_INT * p, float * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, spotri, (uplo, p, ata, ldata, info, 1));
    }

    static void xxpotri(char * uplo, DAAL_INT * p, float * ata, DAAL_INT * ldata, DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, spotri, (uplo, p, ata, ldata, info, 1));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xgerqf(DAAL_INT * m, DAAL_INT * n, float * a, DAAL_INT * lda, float * tau, float * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, sgerqf, (m, n, a, lda, tau, work, lwork, info));
    }

    static void xxgerqf(DAAL_INT * m, DAAL_INT * n, float * a, DAAL_INT * lda, float * tau, float * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, sgerqf, (m, n, a, lda, tau, work, lwork, info));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xormrq(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, float * a, DAAL_INT * lda, float * tau, float * c,
                       DAAL_INT * ldc, float * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, sormrq, (side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info, 1, 1));
    }

    static void xxormrq(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, float * a, DAAL_INT * lda, float * tau, float * c,
                        DAAL_INT * ldc, float * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, sormrq, (side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info, 1, 1));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xtrtrs(char * uplo, char * trans, char * diag, DAAL_INT * n, DAAL_INT * nrhs, float * a, DAAL_INT * lda, float * b, DAAL_INT * ldb,
                       DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, strtrs, (uplo, trans, diag, n, nrhs, a, lda, b, ldb, info, 1, 1, 1));
    }

    static void xxtrtrs(char * uplo, char * trans, char * diag, DAAL_INT * n, DAAL_INT * nrhs, float * a, DAAL_INT * lda, float * b, DAAL_INT * ldb,
                        DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, strtrs, (uplo, trans, diag, n, nrhs, a, lda, b, ldb, info, 1, 1, 1));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xpptrf(char * uplo, DAAL_INT * n, float * ap, DAAL_INT * info) { __DAAL_MKLFN_CALL(lapack_, spptrf, (uplo, n, ap, info, 1)); }

    static void xxpptrf(char * uplo, DAAL_INT * n, float * ap, DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, spptrf, (uplo, n, ap, info, 1));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xgeqrf(DAAL_INT m, DAAL_INT n, float * a, DAAL_INT lda, float * tau, float * work, DAAL_INT lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, sgeqrf, (&m, &n, a, &lda, tau, work, &lwork, info));
    }

    static void xxgeqrf(DAAL_INT m, DAAL_INT n, float * a, DAAL_INT lda, float * tau, float * work, DAAL_INT lwork, DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, sgeqrf, (&m, &n, a, &lda, tau, work, &lwork, info));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xgeqp3(const DAAL_INT m, const DAAL_INT n, float * a, const DAAL_INT lda, DAAL_INT * jpvt, float * tau, float * work,
                       const DAAL_INT lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, sgeqp3, (&m, &n, a, &lda, jpvt, tau, work, &lwork, info));
    }

    static void xxgeqp3(DAAL_INT m, DAAL_INT n, float * a, DAAL_INT lda, DAAL_INT * jpvt, float * tau, float * work, DAAL_INT lwork, DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, sgeqp3, (&m, &n, a, &lda, jpvt, tau, work, &lwork, info));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xorgqr(const DAAL_INT m, const DAAL_INT n, const DAAL_INT k, float * a, const DAAL_INT lda, const float * tau, float * work,
                       const DAAL_INT lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, sorgqr, (&m, &n, &k, a, &lda, tau, work, &lwork, info));
    }

    static void xxorgqr(DAAL_INT m, DAAL_INT n, DAAL_INT k, float * a, DAAL_INT lda, float * tau, float * work, DAAL_INT lwork, DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, sorgqr, (&m, &n, &k, a, &lda, tau, work, &lwork, info));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xgesvd(char jobu, char jobvt, DAAL_INT m, DAAL_INT n, float * a, DAAL_INT lda, float * s, float * u, DAAL_INT ldu, float * vt,
                       DAAL_INT ldvt, float * work, DAAL_INT lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, sgesvd, (&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, info, 1, 1));
    }

    static void xxgesvd(char jobu, char jobvt, DAAL_INT m, DAAL_INT n, float * a, DAAL_INT lda, float * s, float * u, DAAL_INT ldu, float * vt,
                        DAAL_INT ldvt, float * work, DAAL_INT lwork, DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, sgesvd, (&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, info, 1, 1));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xsyevd(char * jobz, char * uplo, DAAL_INT * n, float * a, DAAL_INT * lda, float * w, float * work, DAAL_INT * lwork, DAAL_INT * iwork,
                       DAAL_INT * liwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, ssyevd, (jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info, 1, 1));
    }

    static void xxsyevd(char * jobz, char * uplo, DAAL_INT * n, float * a, DAAL_INT * lda, float * w, float * work, DAAL_INT * lwork,
                        DAAL_INT * iwork, DAAL_INT * liwork, DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, ssyevd, (jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info, 1, 1));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xormqr(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, float * a, DAAL_INT * lda, float * tau, float * c,
                       DAAL_INT * ldc, float * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        __DAAL_MKLFN_CALL(lapack_, sormqr, (side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info, 1, 1));
    }

    static void xxormqr(char * side, char * trans, DAAL_INT * m, DAAL_INT * n, DAAL_INT * k, float * a, DAAL_INT * lda, float * tau, float * c,
                        DAAL_INT * ldc, float * work, DAAL_INT * lwork, DAAL_INT * info)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(lapack_, sormqr, (side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info, 1, 1));
        fpk_serv_set_num_threads_local(old_threads);
    }
};

} // namespace mkl
} // namespace internal
} // namespace daal

#endif
