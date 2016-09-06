/* file: service_blas_mkl.h */
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
//  Template wrappers for common MKL functions.
//--
*/


#ifndef __SERVICE_BLAS_MKL_H__
#define __SERVICE_BLAS_MKL_H__

#include "daal_defines.h"
#include "mkl_daal.h"

#if !defined(__DAAL_CONCAT4)
    #define __DAAL_CONCAT4(a,b,c,d) __DAAL_CONCAT41(a,b,c,d)
    #define __DAAL_CONCAT41(a,b,c,d) a##b##c##d
#endif

#if !defined(__DAAL_CONCAT5)
    #define __DAAL_CONCAT5(a,b,c,d,e) __DAAL_CONCAT51(a,b,c,d,e)
    #define __DAAL_CONCAT51(a,b,c,d,e) a##b##c##d##e
#endif

#if defined(__APPLE__)
    #define __DAAL_MKL_SSE2  ssse3_
    #define __DAAL_MKL_SSSE3 ssse3_
#else
    #define __DAAL_MKL_SSE2  sse2_
    #define __DAAL_MKL_SSSE3 ssse3_
#endif

#define __DAAL_MKLFN(f_cpu,f_pref,f_name)        __DAAL_CONCAT4(fpk_,f_pref,f_cpu,f_name)
#define __DAAL_MKLFN_CALL(f_pref,f_name,f_args)  __DAAL_MKLFN_CALL1(f_pref,f_name,f_args)
#define __DAAL_MKLFN_CALL_RETURN(f_pref,f_name,f_args)  __DAAL_MKLFN_CALL2(f_pref,f_name,f_args)

#if (defined(__x86_64__) && !defined(__APPLE__)) || defined(_WIN64)
    #define __DAAL_MKLFPK_KNL   avx512_mic_
#else
    #define __DAAL_MKLFPK_KNL   avx2_
#endif


#define __DAAL_MKLFN_CALL1(f_pref,f_name,f_args)                    \
    if(avx512 == cpu)                                               \
    {                                                               \
        __DAAL_MKLFN(avx512_,f_pref,f_name) f_args;                 \
    }                                                               \
    if(avx512_mic == cpu)                                           \
    {                                                               \
        __DAAL_MKLFN(__DAAL_MKLFPK_KNL,f_pref,f_name) f_args;       \
    }                                                               \
    if(avx2 == cpu)                                                 \
    {                                                               \
        __DAAL_MKLFN(avx2_,f_pref,f_name) f_args;                   \
    }                                                               \
    if(avx == cpu)                                                  \
    {                                                               \
        __DAAL_MKLFN(avx_,f_pref,f_name) f_args;                    \
    }                                                               \
    if(sse42 == cpu)                                                \
    {                                                               \
        __DAAL_MKLFN(sse42_,f_pref,f_name) f_args;                  \
    }                                                               \
    if(ssse3 == cpu)                                                \
    {                                                               \
        __DAAL_MKLFN(__DAAL_MKL_SSSE3,f_pref,f_name) f_args;        \
    }                                                               \
    if(sse2 == cpu)                                                 \
    {                                                               \
        __DAAL_MKLFN(__DAAL_MKL_SSE2,f_pref,f_name) f_args;         \
    }

#define __DAAL_MKLFN_CALL2(f_pref,f_name,f_args)                    \
    if(avx512 == cpu)                                               \
    {                                                               \
        return __DAAL_MKLFN(avx512_,f_pref,f_name) f_args;          \
    }                                                               \
    if(avx512_mic == cpu)                                           \
    {                                                               \
        return __DAAL_MKLFN(__DAAL_MKLFPK_KNL,f_pref,f_name) f_args;\
    }                                                               \
    if(avx2 == cpu)                                                 \
    {                                                               \
        return __DAAL_MKLFN(avx2_,f_pref,f_name) f_args;            \
    }                                                               \
    if(avx == cpu)                                                  \
    {                                                               \
        return __DAAL_MKLFN(avx_,f_pref,f_name) f_args;             \
    }                                                               \
    if(sse42 == cpu)                                                \
    {                                                               \
        return __DAAL_MKLFN(sse42_,f_pref,f_name) f_args;           \
    }                                                               \
    if(ssse3 == cpu)                                                \
    {                                                               \
        return __DAAL_MKLFN(__DAAL_MKL_SSSE3,f_pref,f_name) f_args; \
    }                                                               \
    if(sse2 == cpu)                                                 \
    {                                                               \
        return __DAAL_MKLFN(__DAAL_MKL_SSE2,f_pref,f_name) f_args;  \
    }

namespace daal
{
namespace internal
{
namespace mkl
{

template<typename fpType, CpuType cpu>
struct MklBlas {};

/*
// Double precision functions definition
*/

template<CpuType cpu>
struct MklBlas<double, cpu>
{
    typedef MKL_INT SizeType;

    static void xsyrk(char *uplo, char *trans, MKL_INT *p, MKL_INT *n, double *alpha, double *a, MKL_INT *lda,
               double *beta, double *ata, MKL_INT *ldata)
    {
        __DAAL_MKLFN_CALL(blas_, dsyrk, (uplo, trans, p, n, alpha, a, lda, beta, ata, ldata));
    }

    static void xxsyrk(char *uplo, char *trans, MKL_INT *p, MKL_INT *n, double *alpha, double *a, MKL_INT *lda,
               double *beta, double *ata, MKL_INT *ldata)
    {
        __DAAL_MKLFN_CALL(blas_, xdsyrk, (uplo, trans, p, n, alpha, a, lda, beta, ata, ldata));
    }

    static void xsyr(const char *uplo, const MKL_INT *n, const double *alpha,
              const double *x, const MKL_INT *incx, double *a, const MKL_INT *lda)
    {
        __DAAL_MKLFN_CALL(blas_, dsyr, (uplo, n, alpha, x, incx , a, lda));
    }

    static void xxsyr(const char *uplo, const MKL_INT *n, const double *alpha,
              const double *x, const MKL_INT *incx, double *a, const MKL_INT *lda)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(blas_, dsyr, (uplo, n, alpha, x, incx , a, lda));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xgemm(char *transa, char *transb, MKL_INT *p, MKL_INT *ny, MKL_INT *n, double *alpha, double *a,
               MKL_INT *lda, double *y, MKL_INT *ldy, double *beta, double *aty, MKL_INT *ldaty)
    {
        __DAAL_MKLFN_CALL(blas_, dgemm, (transa, transb, p, ny, n, alpha, a, lda, y, ldy, beta, aty, ldaty));
    }

    static void xxgemm(char *transa, char *transb, MKL_INT *p, MKL_INT *ny, MKL_INT *n, double *alpha, double *a,
               MKL_INT *lda, double *y, MKL_INT *ldy, double *beta, double *aty, MKL_INT *ldaty)
    {
        __DAAL_MKLFN_CALL(blas_, xdgemm, (transa, transb, p, ny, n, alpha, a, lda, y, ldy, beta, aty, ldaty));
    }

    static void xsymm(char *side, char *uplo, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *lda, double *b,
               MKL_INT *ldb, double *beta, double *c, MKL_INT *ldc)
    {
        __DAAL_MKLFN_CALL(blas_, dsymm, (side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc));
    }

    static void xxsymm(char *side, char *uplo, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *lda, double *b,
               MKL_INT *ldb, double *beta, double *c, MKL_INT *ldc)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(blas_, dsymm, (side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xgemv(char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *lda, double *x,
               MKL_INT *incx, double *beta, double *y, MKL_INT *incy)
    {
        __DAAL_MKLFN_CALL(blas_, dgemv, (trans, m, n, alpha, a, lda, x, incx, beta, y, incy));
    }

    static void xxgemv(char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *lda, double *x,
               MKL_INT *incx, double *beta, double *y, MKL_INT *incy)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(blas_, dgemv, (trans, m, n, alpha, a, lda, x, incx, beta, y, incy));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xaxpy(MKL_INT *n, double *a, double *x, MKL_INT *incx, double *y, MKL_INT *incy)
    {
        __DAAL_MKLFN_CALL(blas_, daxpy, (n, a, x, incx, y, incy));
    }

    static void xxaxpy(MKL_INT *n, double *a, double *x, MKL_INT *incx, double *y, MKL_INT *incy)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(blas_, daxpy, (n, a, x, incx, y, incy));
        fpk_serv_set_num_threads_local(old_threads);
    }

};

/*
// Single precision functions definition
*/

template<CpuType cpu>
struct MklBlas<float, cpu>
{
    typedef MKL_INT SizeType;

    static void xsyrk(char *uplo, char *trans, MKL_INT *p, MKL_INT *n, float *alpha, float *a, MKL_INT *lda, float *beta,
               float *ata, MKL_INT *ldata)
    {
        __DAAL_MKLFN_CALL(blas_, ssyrk, (uplo, trans, p, n, alpha, a, lda, beta, ata, ldata));
    }

    static void xxsyrk(char *uplo, char *trans, MKL_INT *p, MKL_INT *n, float *alpha, float *a, MKL_INT *lda, float *beta,
               float *ata, MKL_INT *ldata)
    {
        __DAAL_MKLFN_CALL(blas_, xssyrk, (uplo, trans, p, n, alpha, a, lda, beta, ata, ldata));
    }

    static void xsyr(const char *uplo, const MKL_INT *n, const float *alpha,
              const float *x, const MKL_INT *incx, float *a, const MKL_INT *lda)
    {
        __DAAL_MKLFN_CALL(blas_, ssyr, (uplo, n, alpha, x, incx , a, lda));
    }

    static void xxsyr(const char *uplo, const MKL_INT *n, const float *alpha,
              const float *x, const MKL_INT *incx, float *a, const MKL_INT *lda)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(blas_, ssyr, (uplo, n, alpha, x, incx , a, lda));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xgemm(char *transa, char *transb, MKL_INT *p, MKL_INT *ny, MKL_INT *n, float *alpha, float *a,
               MKL_INT *lda, float *y, MKL_INT *ldy, float *beta, float *aty, MKL_INT *ldaty)
    {
        __DAAL_MKLFN_CALL(blas_, sgemm, (transa, transb, p, ny, n, alpha, a, lda, y, ldy, beta, aty, ldaty));
    }

    static void xxgemm(char *transa, char *transb, MKL_INT *p, MKL_INT *ny, MKL_INT *n, float *alpha, float *a,
               MKL_INT *lda, float *y, MKL_INT *ldy, float *beta, float *aty, MKL_INT *ldaty)
    {
        __DAAL_MKLFN_CALL(blas_, xsgemm, (transa, transb, p, ny, n, alpha, a, lda, y, ldy, beta, aty, ldaty));
    }

    static void xsymm(char *side, char *uplo, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *lda, float *b,
               MKL_INT *ldb, float *beta, float *c, MKL_INT *ldc)
    {
        __DAAL_MKLFN_CALL(blas_, ssymm, (side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc));
    }

    static void xxsymm(char *side, char *uplo, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *lda, float *b,
               MKL_INT *ldb, float *beta, float *c, MKL_INT *ldc)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(blas_, ssymm, (side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xgemv(char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *lda, float *x, MKL_INT *incx,
               float *beta, float *y, MKL_INT *incy)
    {
        __DAAL_MKLFN_CALL(blas_, sgemv, (trans, m, n, alpha, a, lda, x, incx, beta, y, incy));
    }

    static void xxgemv(char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *lda, float *x, MKL_INT *incx,
               float *beta, float *y, MKL_INT *incy)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(blas_, sgemv, (trans, m, n, alpha, a, lda, x, incx, beta, y, incy));
        fpk_serv_set_num_threads_local(old_threads);
    }

    static void xaxpy(MKL_INT *n, float *a, float *x, MKL_INT *incx, float *y, MKL_INT *incy)
    {
        __DAAL_MKLFN_CALL(blas_, saxpy, (n, a, x, incx, y, incy));
    }

    static void xxaxpy(MKL_INT *n, float *a, float *x, MKL_INT *incx, float *y, MKL_INT *incy)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(blas_, saxpy, (n, a, x, incx, y, incy));
        fpk_serv_set_num_threads_local(old_threads);
    }

};



} // namespace mkl
} // namespace internal
} // namespace daal

#endif
