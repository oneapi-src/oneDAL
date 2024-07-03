/*******************************************************************************
* Copyright 2014-2023 Intel Corporation.
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


#ifndef MKL_DAL_H
#define MKL_DAL_H

#include <stddef.h>

#ifdef __cplusplus
#if __cplusplus > 199711L
#define NOTHROW noexcept
#else
#define NOTHROW throw()
#endif
#else
#define NOTHROW
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#if !defined(MKL_INT)
#if defined(_WIN64) || defined(__x86_64__)
#define MKL_INT __int64
#else
#define MKL_INT __int32
#endif
#endif

typedef void *             _MKL_DSS_HANDLE_t;

enum PARDISO_ENV_PARAM {
       PARDISO_OOC_FILE_NAME = 1
};

#define MKL_MEM_MCDRAM 1

#define MKL_ENABLE_AVX512_MIC_E1 5

#if !defined(__DAAL_CONCAT4)
    #define __DAAL_CONCAT4(a, b, c, d)  __DAAL_CONCAT41(a, b, c, d)
    #define __DAAL_CONCAT41(a, b, c, d) a##b##c##d
#endif

#if !defined(__DAAL_CONCAT5)
    #define __DAAL_CONCAT5(a, b, c, d, e)  __DAAL_CONCAT51(a, b, c, d, e)
    #define __DAAL_CONCAT51(a, b, c, d, e) a##b##c##d##e
#endif

#if defined(__APPLE__)
    #define __DAAL_MKL_SSE2  avx_
    #define __DAAL_MKL_SSE42 avx_
#else
    #define __DAAL_MKL_SSE2  sse2_
    #define __DAAL_MKL_SSE42 sse42_
#endif

#define __DAAL_MKLFN(f_cpu, f_pref, f_name)              __DAAL_CONCAT4(mkl_, f_pref, f_cpu, f_name)
#define __DAAL_MKLFN_(f_cpu, f_pref, f_name)              f_name
#define __DAAL_MKLFN_CALL_(f_pref, f_name, f_args)        __DAAL_MKLFN_CALL1(f_pref, f_name, f_args)
#define __DAAL_MKLFN_CALL(f_pref, f_name, f_args)        __DAAL_MKLFN_CALL1(f_pref, f_name, f_args)
#define __DAAL_MKLFN_CALL_RETURN(f_pref, f_name, f_args) __DAAL_MKLFN_CALL2(f_pref, f_name, f_args)

#define __DAAL_MKLFN_CALL1(f_pref, f_name, f_args)             \
    if (avx512 == cpu)                                         \
    {                                                          \
        __DAAL_MKLFN(avx2_, f_pref, f_name) f_args;          \
    }                                                          \
    if (avx2 == cpu)                                           \
    {                                                          \
        __DAAL_MKLFN(avx2_, f_pref, f_name) f_args;            \
    }                                                          \
    if (sse42 == cpu)                                          \
    {                                                          \
        __DAAL_MKLFN(__DAAL_MKL_SSE42, f_pref, f_name) f_args; \
    }                                                          \
    if (sse2 == cpu)                                           \
    {                                                          \
        __DAAL_MKLFN(__DAAL_MKL_SSE42, f_pref, f_name) f_args;  \
    }

#define __DAAL_MKLFN_CALL1_(f_pref, f_name, f_args)             \
    if (avx512 == cpu)                                         \
    {                                                          \
        __DAAL_MKLFN_(avx2_, f_pref, f_name) f_args;          \
    }                                                          \
    if (avx2 == cpu)                                           \
    {                                                          \
        __DAAL_MKLFN_(avx2_, f_pref, f_name) f_args;            \
    }                                                          \
    if (sse42 == cpu)                                          \
    {                                                          \
        __DAAL_MKLFN_(__DAAL_MKL_SSE42, f_pref, f_name) f_args; \
    }                                                          \
    if (sse2 == cpu)                                           \
    {                                                          \
        __DAAL_MKLFN_(__DAAL_MKL_SSE42, f_pref, f_name) f_args;  \
    }

#define __DAAL_MKLFN_CALL2(f_pref, f_name, f_args)                    \
    if (avx512 == cpu)                                                \
    {                                                                 \
        return __DAAL_MKLFN(avx2_, f_pref, f_name) f_args;          \
    }                                                                 \
    if (avx2 == cpu)                                                  \
    {                                                                 \
        return __DAAL_MKLFN(avx2_, f_pref, f_name) f_args;            \
    }                                                                 \
    if (sse42 == cpu)                                                 \
    {                                                                 \
        return __DAAL_MKLFN(__DAAL_MKL_SSE42, f_pref, f_name) f_args; \
    }                                                                 \
    if (sse2 == cpu)                                                  \
    {                                                                 \
        return __DAAL_MKLFN(__DAAL_MKL_SSE42, f_pref, f_name) f_args;  \
    }


typedef int             IppStatus;
typedef unsigned char   Ipp8u;
typedef unsigned short  Ipp16u;
typedef unsigned int    Ipp32u;
typedef signed short    Ipp16s;
typedef signed int      Ipp32s;
typedef float           Ipp32f;
typedef double          Ipp64f;

typedef void          (*func_type)(DAAL_INT , DAAL_INT , DAAL_INT , void *);
void mkl_vsl_serv_threader_for(DAAL_INT n, DAAL_INT threads_request, void* a, func_type func);
void mkl_vsl_serv_threader_for_ordered(DAAL_INT n, DAAL_INT threads_request, void* a, func_type func);
void mkl_vsl_serv_threader_sections(DAAL_INT threads_request, void* a, func_type func);
void mkl_vsl_serv_threader_ordered(DAAL_INT i, DAAL_INT th_idx, DAAL_INT th_num, void* a, func_type func);
DAAL_INT mkl_vsl_serv_threader_get_num_threads_limit(void);

void mkl_blas_sse2_daxpy(const MKL_INT *n, const double *alpha, const double *x,
    const MKL_INT *incx, double *y, const MKL_INT *incy);
void mkl_blas_ssse3_daxpy(const MKL_INT *n, const double *alpha, const double *x,
    const MKL_INT *incx, double *y, const MKL_INT *incy);
void mkl_blas_sse42_daxpy(const MKL_INT *n, const double *alpha, const double *x,
    const MKL_INT *incx, double *y, const MKL_INT *incy);
void mkl_blas_avx_daxpy(const MKL_INT *n, const double *alpha, const double *x,
    const MKL_INT *incx, double *y, const MKL_INT *incy);
void mkl_blas_avx2_daxpy(const MKL_INT *n, const double *alpha, const double *x,
    const MKL_INT *incx, double *y, const MKL_INT *incy);
void mkl_blas_avx512_daxpy(const MKL_INT *n, const double *alpha,
    const double *x, const MKL_INT *incx, double *y, const MKL_INT *incy);

void mkl_blas_sse2_dcopy(const MKL_INT *n, const double *x, const MKL_INT *incx,
    double *y, const MKL_INT *incy);
void mkl_blas_ssse3_dcopy(const MKL_INT *n, const double *x, const MKL_INT *incx,
    double *y, const MKL_INT *incy);
void mkl_blas_sse42_dcopy(const MKL_INT *n, const double *x, const MKL_INT *incx,
    double *y, const MKL_INT *incy);
void mkl_blas_avx_dcopy(const MKL_INT *n, const double *x, const MKL_INT *incx,
    double *y, const MKL_INT *incy);
void mkl_blas_avx2_dcopy(const MKL_INT *n, const double *x, const MKL_INT *incx,
    double *y, const MKL_INT *incy);
void mkl_blas_avx512_dcopy(const MKL_INT *n, const double *x,
    const MKL_INT *incx, double *y, const MKL_INT *incy);

double mkl_blas_sse2_ddot(const MKL_INT *n, const double *x, const MKL_INT *incx,
    const double *y, const MKL_INT *incy);
double mkl_blas_ssse3_ddot(const MKL_INT *n, const double *x,
    const MKL_INT *incx, const double *y, const MKL_INT *incy);
double mkl_blas_sse42_ddot(const MKL_INT *n, const double *x,
    const MKL_INT *incx, const double *y, const MKL_INT *incy);
double mkl_blas_avx_ddot(const MKL_INT *n, const double *x, const MKL_INT *incx,
    const double *y, const MKL_INT *incy);
double mkl_blas_avx2_ddot(const MKL_INT *n, const double *x, const MKL_INT *incx,
    const double *y, const MKL_INT *incy);
double mkl_blas_avx512_ddot(const MKL_INT *n, const double *x,
    const MKL_INT *incx, const double *y, const MKL_INT *incy);

void mkl_blas_sse2_dgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_ssse3_dgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_sse42_dgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_avx_dgemm(const char *transa, const char *transb, const MKL_INT *m,
    const MKL_INT *n, const MKL_INT *k, const double *alpha, const double *a,
    const MKL_INT *lda, const double *b, const MKL_INT *ldb, const double *beta,
    double *c, const MKL_INT *ldc);
void mkl_blas_avx2_dgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_avx512_dgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);

void mkl_blas_sse2_dgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_ssse3_dgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_sse42_dgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_avx_dgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_avx2_dgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_avx512_dgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);

void mkl_blas_sse2_dgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, const double *x,
    const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void mkl_blas_ssse3_dgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, const double *x,
    const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void mkl_blas_sse42_dgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, const double *x,
    const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void mkl_blas_avx_dgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, const double *x,
    const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void mkl_blas_avx2_dgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, const double *x,
    const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void mkl_blas_avx512_dgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, const double *x,
    const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);

void mkl_blas_sse2_dsymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
    const double *b, const MKL_INT *ldb, const double *beta, double *c,
    const MKL_INT *ldc);
void mkl_blas_ssse3_dsymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
    const double *b, const MKL_INT *ldb, const double *beta, double *c,
    const MKL_INT *ldc);
void mkl_blas_sse42_dsymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
    const double *b, const MKL_INT *ldb, const double *beta, double *c,
    const MKL_INT *ldc);
void mkl_blas_avx_dsymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
    const double *b, const MKL_INT *ldb, const double *beta, double *c,
    const MKL_INT *ldc);
void mkl_blas_avx2_dsymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
    const double *b, const MKL_INT *ldb, const double *beta, double *c,
    const MKL_INT *ldc);
void mkl_blas_avx512_dsymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
    const double *b, const MKL_INT *ldb, const double *beta, double *c,
    const MKL_INT *ldc);

void mkl_blas_sse2_dsyr(const char *uplo, const MKL_INT *n, const double *alpha,
    const double *x, const MKL_INT *incx, double *a, const MKL_INT *lda);
void mkl_blas_ssse3_dsyr(const char *uplo, const MKL_INT *n, const double *alpha,
    const double *x, const MKL_INT *incx, double *a, const MKL_INT *lda);
void mkl_blas_sse42_dsyr(const char *uplo, const MKL_INT *n, const double *alpha,
    const double *x, const MKL_INT *incx, double *a, const MKL_INT *lda);
void mkl_blas_avx_dsyr(const char *uplo, const MKL_INT *n, const double *alpha,
    const double *x, const MKL_INT *incx, double *a, const MKL_INT *lda);
void mkl_blas_avx2_dsyr(const char *uplo, const MKL_INT *n, const double *alpha,
    const double *x, const MKL_INT *incx, double *a, const MKL_INT *lda);
void mkl_blas_avx512_dsyr(const char *uplo, const MKL_INT *n,
    const double *alpha, const double *x, const MKL_INT *incx, double *a,
    const MKL_INT *lda);

void mkl_blas_sse2_dsyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *lda,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_ssse3_dsyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *lda,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_sse42_dsyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *lda,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_avx_dsyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *lda,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_avx2_dsyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *lda,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_avx512_dsyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *lda,
    const double *beta, double *c, const MKL_INT *ldc);

void mkl_blas_sse2_dtrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const double *alpha,
    const double *a, const MKL_INT *lda, double *b, const MKL_INT *ldb);
void mkl_blas_ssse3_dtrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const double *alpha,
    const double *a, const MKL_INT *lda, double *b, const MKL_INT *ldb);
void mkl_blas_sse42_dtrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const double *alpha,
    const double *a, const MKL_INT *lda, double *b, const MKL_INT *ldb);
void mkl_blas_avx_dtrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const double *alpha,
    const double *a, const MKL_INT *lda, double *b, const MKL_INT *ldb);
void mkl_blas_avx2_dtrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const double *alpha,
    const double *a, const MKL_INT *lda, double *b, const MKL_INT *ldb);
void mkl_blas_avx512_dtrmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, double *b,
    const MKL_INT *ldb);

void mkl_blas_sse2_saxpy(const MKL_INT *n, const float *alpha, const float *x,
    const MKL_INT *incx, float *y, const MKL_INT *incy);
void mkl_blas_ssse3_saxpy(const MKL_INT *n, const float *alpha, const float *x,
    const MKL_INT *incx, float *y, const MKL_INT *incy);
void mkl_blas_sse42_saxpy(const MKL_INT *n, const float *alpha, const float *x,
    const MKL_INT *incx, float *y, const MKL_INT *incy);
void mkl_blas_avx_saxpy(const MKL_INT *n, const float *alpha, const float *x,
    const MKL_INT *incx, float *y, const MKL_INT *incy);
void mkl_blas_avx2_saxpy(const MKL_INT *n, const float *alpha, const float *x,
    const MKL_INT *incx, float *y, const MKL_INT *incy);
void mkl_blas_avx512_saxpy(const MKL_INT *n, const float *alpha, const float *x,
    const MKL_INT *incx, float *y, const MKL_INT *incy);

void mkl_blas_sse2_scopy(const MKL_INT *n, const float *x, const MKL_INT *incx,
    float *y, const MKL_INT *incy);
void mkl_blas_ssse3_scopy(const MKL_INT *n, const float *x, const MKL_INT *incx,
    float *y, const MKL_INT *incy);
void mkl_blas_sse42_scopy(const MKL_INT *n, const float *x, const MKL_INT *incx,
    float *y, const MKL_INT *incy);
void mkl_blas_avx_scopy(const MKL_INT *n, const float *x, const MKL_INT *incx,
    float *y, const MKL_INT *incy);
void mkl_blas_avx2_scopy(const MKL_INT *n, const float *x, const MKL_INT *incx,
    float *y, const MKL_INT *incy);
void mkl_blas_avx512_scopy(const MKL_INT *n, const float *x, const MKL_INT *incx,
    float *y, const MKL_INT *incy);

float mkl_blas_sse2_sdot(const MKL_INT *n, const float *x, const MKL_INT *incx,
    const float *y, const MKL_INT *incy);
float mkl_blas_ssse3_sdot(const MKL_INT *n, const float *x, const MKL_INT *incx,
    const float *y, const MKL_INT *incy);
float mkl_blas_sse42_sdot(const MKL_INT *n, const float *x, const MKL_INT *incx,
    const float *y, const MKL_INT *incy);
float mkl_blas_avx_sdot(const MKL_INT *n, const float *x, const MKL_INT *incx,
    const float *y, const MKL_INT *incy);
float mkl_blas_avx2_sdot(const MKL_INT *n, const float *x, const MKL_INT *incx,
    const float *y, const MKL_INT *incy);
float mkl_blas_avx512_sdot(const MKL_INT *n, const float *x, const MKL_INT *incx,
    const float *y, const MKL_INT *incy);

void mkl_blas_sse2_sgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_ssse3_sgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_sse42_sgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_avx_sgemm(const char *transa, const char *transb, const MKL_INT *m,
    const MKL_INT *n, const MKL_INT *k, const float *alpha, const float *a,
    const MKL_INT *lda, const float *b, const MKL_INT *ldb, const float *beta,
    float *c, const MKL_INT *ldc);
void mkl_blas_avx2_sgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_avx512_sgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);

void mkl_blas_sse2_sgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_ssse3_sgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_sse42_sgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_avx_sgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_avx2_sgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_avx512_sgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);

void mkl_blas_sse2_sgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, const float *x,
    const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void mkl_blas_ssse3_sgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, const float *x,
    const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void mkl_blas_sse42_sgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, const float *x,
    const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void mkl_blas_avx_sgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, const float *x,
    const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void mkl_blas_avx2_sgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, const float *x,
    const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void mkl_blas_avx512_sgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, const float *x,
    const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);

void mkl_blas_sse2_ssymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
    const float *b, const MKL_INT *ldb, const float *beta, float *c,
    const MKL_INT *ldc);
void mkl_blas_ssse3_ssymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
    const float *b, const MKL_INT *ldb, const float *beta, float *c,
    const MKL_INT *ldc);
void mkl_blas_sse42_ssymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
    const float *b, const MKL_INT *ldb, const float *beta, float *c,
    const MKL_INT *ldc);
void mkl_blas_avx_ssymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
    const float *b, const MKL_INT *ldb, const float *beta, float *c,
    const MKL_INT *ldc);
void mkl_blas_avx2_ssymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
    const float *b, const MKL_INT *ldb, const float *beta, float *c,
    const MKL_INT *ldc);
void mkl_blas_avx512_ssymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
    const float *b, const MKL_INT *ldb, const float *beta, float *c,
    const MKL_INT *ldc);

void mkl_blas_sse2_ssyr(const char *uplo, const MKL_INT *n, const float *alpha,
    const float *x, const MKL_INT *incx, float *a, const MKL_INT *lda);
void mkl_blas_ssse3_ssyr(const char *uplo, const MKL_INT *n, const float *alpha,
    const float *x, const MKL_INT *incx, float *a, const MKL_INT *lda);
void mkl_blas_sse42_ssyr(const char *uplo, const MKL_INT *n, const float *alpha,
    const float *x, const MKL_INT *incx, float *a, const MKL_INT *lda);
void mkl_blas_avx_ssyr(const char *uplo, const MKL_INT *n, const float *alpha,
    const float *x, const MKL_INT *incx, float *a, const MKL_INT *lda);
void mkl_blas_avx2_ssyr(const char *uplo, const MKL_INT *n, const float *alpha,
    const float *x, const MKL_INT *incx, float *a, const MKL_INT *lda);
void mkl_blas_avx512_ssyr(const char *uplo, const MKL_INT *n, const float *alpha,
    const float *x, const MKL_INT *incx, float *a, const MKL_INT *lda);

void mkl_blas_sse2_ssyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *lda,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_ssse3_ssyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *lda,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_sse42_ssyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *lda,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_avx_ssyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *lda,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_avx2_ssyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *lda,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_avx512_ssyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *lda,
    const float *beta, float *c, const MKL_INT *ldc);

void mkl_blas_sse2_strmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const float *alpha,
    const float *a, const MKL_INT *lda, float *b, const MKL_INT *ldb);
void mkl_blas_ssse3_strmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const float *alpha,
    const float *a, const MKL_INT *lda, float *b, const MKL_INT *ldb);
void mkl_blas_sse42_strmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const float *alpha,
    const float *a, const MKL_INT *lda, float *b, const MKL_INT *ldb);
void mkl_blas_avx_strmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const float *alpha,
    const float *a, const MKL_INT *lda, float *b, const MKL_INT *ldb);
void mkl_blas_avx2_strmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const float *alpha,
    const float *a, const MKL_INT *lda, float *b, const MKL_INT *ldb);
void mkl_blas_avx512_strmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, float *b,
    const MKL_INT *ldb);

void mkl_blas_sse2_xdaxpy(const MKL_INT *n, const double *alpha, const double *x,
    const MKL_INT *incx, double *y, const MKL_INT *incy);
void mkl_blas_ssse3_xdaxpy(const MKL_INT *n, const double *alpha,
    const double *x, const MKL_INT *incx, double *y, const MKL_INT *incy);
void mkl_blas_sse42_xdaxpy(const MKL_INT *n, const double *alpha,
    const double *x, const MKL_INT *incx, double *y, const MKL_INT *incy);
void mkl_blas_avx_xdaxpy(const MKL_INT *n, const double *alpha, const double *x,
    const MKL_INT *incx, double *y, const MKL_INT *incy);
void mkl_blas_avx2_xdaxpy(const MKL_INT *n, const double *alpha, const double *x,
    const MKL_INT *incx, double *y, const MKL_INT *incy);
void mkl_blas_avx512_xdaxpy(const MKL_INT *n, const double *alpha,
    const double *x, const MKL_INT *incx, double *y, const MKL_INT *incy);

void mkl_blas_sse2_xdcopy(const MKL_INT *n, const double *x, const MKL_INT *incx,
    double *y, const MKL_INT *incy);
void mkl_blas_ssse3_xdcopy(const MKL_INT *n, const double *x,
    const MKL_INT *incx, double *y, const MKL_INT *incy);
void mkl_blas_sse42_xdcopy(const MKL_INT *n, const double *x,
    const MKL_INT *incx, double *y, const MKL_INT *incy);
void mkl_blas_avx_xdcopy(const MKL_INT *n, const double *x, const MKL_INT *incx,
    double *y, const MKL_INT *incy);
void mkl_blas_avx2_xdcopy(const MKL_INT *n, const double *x, const MKL_INT *incx,
    double *y, const MKL_INT *incy);
void mkl_blas_avx512_xdcopy(const MKL_INT *n, const double *x,
    const MKL_INT *incx, double *y, const MKL_INT *incy);

double mkl_blas_sse2_xddot(const MKL_INT *n, const double *x,
    const MKL_INT *incx, const double *y, const MKL_INT *incy);
double mkl_blas_ssse3_xddot(const MKL_INT *n, const double *x,
    const MKL_INT *incx, const double *y, const MKL_INT *incy);
double mkl_blas_sse42_xddot(const MKL_INT *n, const double *x,
    const MKL_INT *incx, const double *y, const MKL_INT *incy);
double mkl_blas_avx_xddot(const MKL_INT *n, const double *x, const MKL_INT *incx,
    const double *y, const MKL_INT *incy);
double mkl_blas_avx2_xddot(const MKL_INT *n, const double *x,
    const MKL_INT *incx, const double *y, const MKL_INT *incy);
double mkl_blas_avx512_xddot(const MKL_INT *n, const double *x,
    const MKL_INT *incx, const double *y, const MKL_INT *incy);

void mkl_blas_sse2_xdgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_ssse3_xdgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_sse42_xdgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_avx_xdgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_avx2_xdgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_avx512_xdgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);

void mkl_blas_sse2_xdgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_ssse3_xdgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_sse42_xdgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_avx_xdgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_avx2_xdgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_avx512_xdgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);

void mkl_blas_sse2_xdgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, const double *x,
    const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void mkl_blas_ssse3_xdgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, const double *x,
    const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void mkl_blas_sse42_xdgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, const double *x,
    const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void mkl_blas_avx_xdgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, const double *x,
    const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void mkl_blas_avx2_xdgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, const double *x,
    const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void mkl_blas_avx512_xdgemv(const char *trans, const MKL_INT *m,
    const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
    const double *x, const MKL_INT *incx, const double *beta, double *y,
    const MKL_INT *incy);

void mkl_blas_sse2_xdsymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
    const double *b, const MKL_INT *ldb, const double *beta, double *c,
    const MKL_INT *ldc);
void mkl_blas_ssse3_xdsymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
    const double *b, const MKL_INT *ldb, const double *beta, double *c,
    const MKL_INT *ldc);
void mkl_blas_sse42_xdsymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
    const double *b, const MKL_INT *ldb, const double *beta, double *c,
    const MKL_INT *ldc);
void mkl_blas_avx_xdsymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
    const double *b, const MKL_INT *ldb, const double *beta, double *c,
    const MKL_INT *ldc);
void mkl_blas_avx2_xdsymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
    const double *b, const MKL_INT *ldb, const double *beta, double *c,
    const MKL_INT *ldc);
void mkl_blas_avx512_xdsymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
    const double *b, const MKL_INT *ldb, const double *beta, double *c,
    const MKL_INT *ldc);

void mkl_blas_sse2_xdsyr(const char *uplo, const MKL_INT *n, const double *alpha,
    const double *x, const MKL_INT *incx, double *a, const MKL_INT *lda);
void mkl_blas_ssse3_xdsyr(const char *uplo, const MKL_INT *n,
    const double *alpha, const double *x, const MKL_INT *incx, double *a,
    const MKL_INT *lda);
void mkl_blas_sse42_xdsyr(const char *uplo, const MKL_INT *n,
    const double *alpha, const double *x, const MKL_INT *incx, double *a,
    const MKL_INT *lda);
void mkl_blas_avx_xdsyr(const char *uplo, const MKL_INT *n, const double *alpha,
    const double *x, const MKL_INT *incx, double *a, const MKL_INT *lda);
void mkl_blas_avx2_xdsyr(const char *uplo, const MKL_INT *n, const double *alpha,
    const double *x, const MKL_INT *incx, double *a, const MKL_INT *lda);
void mkl_blas_avx512_xdsyr(const char *uplo, const MKL_INT *n,
    const double *alpha, const double *x, const MKL_INT *incx, double *a,
    const MKL_INT *lda);

void mkl_blas_sse2_xdsyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *lda,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_ssse3_xdsyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *lda,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_sse42_xdsyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *lda,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_avx_xdsyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *lda,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_avx2_xdsyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *lda,
    const double *beta, double *c, const MKL_INT *ldc);
void mkl_blas_avx512_xdsyrk(const char *uplo, const char *trans,
    const MKL_INT *n, const MKL_INT *k, const double *alpha, const double *a,
    const MKL_INT *lda, const double *beta, double *c, const MKL_INT *ldc);

void mkl_blas_sse2_xdtrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const double *alpha,
    const double *a, const MKL_INT *lda, double *b, const MKL_INT *ldb);
void mkl_blas_ssse3_xdtrmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, double *b,
    const MKL_INT *ldb);
void mkl_blas_sse42_xdtrmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, double *b,
    const MKL_INT *ldb);
void mkl_blas_avx_xdtrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const double *alpha,
    const double *a, const MKL_INT *lda, double *b, const MKL_INT *ldb);
void mkl_blas_avx2_xdtrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const double *alpha,
    const double *a, const MKL_INT *lda, double *b, const MKL_INT *ldb);
void mkl_blas_avx512_xdtrmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, double *b,
    const MKL_INT *ldb);

void mkl_blas_sse2_xsaxpy(const MKL_INT *n, const float *alpha, const float *x,
    const MKL_INT *incx, float *y, const MKL_INT *incy);
void mkl_blas_ssse3_xsaxpy(const MKL_INT *n, const float *alpha, const float *x,
    const MKL_INT *incx, float *y, const MKL_INT *incy);
void mkl_blas_sse42_xsaxpy(const MKL_INT *n, const float *alpha, const float *x,
    const MKL_INT *incx, float *y, const MKL_INT *incy);
void mkl_blas_avx_xsaxpy(const MKL_INT *n, const float *alpha, const float *x,
    const MKL_INT *incx, float *y, const MKL_INT *incy);
void mkl_blas_avx2_xsaxpy(const MKL_INT *n, const float *alpha, const float *x,
    const MKL_INT *incx, float *y, const MKL_INT *incy);
void mkl_blas_avx512_xsaxpy(const MKL_INT *n, const float *alpha, const float *x,
    const MKL_INT *incx, float *y, const MKL_INT *incy);

void mkl_blas_sse2_xscopy(const MKL_INT *n, const float *x, const MKL_INT *incx,
    float *y, const MKL_INT *incy);
void mkl_blas_ssse3_xscopy(const MKL_INT *n, const float *x, const MKL_INT *incx,
    float *y, const MKL_INT *incy);
void mkl_blas_sse42_xscopy(const MKL_INT *n, const float *x, const MKL_INT *incx,
    float *y, const MKL_INT *incy);
void mkl_blas_avx_xscopy(const MKL_INT *n, const float *x, const MKL_INT *incx,
    float *y, const MKL_INT *incy);
void mkl_blas_avx2_xscopy(const MKL_INT *n, const float *x, const MKL_INT *incx,
    float *y, const MKL_INT *incy);
void mkl_blas_avx512_xscopy(const MKL_INT *n, const float *x,
    const MKL_INT *incx, float *y, const MKL_INT *incy);

float mkl_blas_sse2_xsdot(const MKL_INT *n, const float *x, const MKL_INT *incx,
    const float *y, const MKL_INT *incy);
float mkl_blas_ssse3_xsdot(const MKL_INT *n, const float *x, const MKL_INT *incx,
    const float *y, const MKL_INT *incy);
float mkl_blas_sse42_xsdot(const MKL_INT *n, const float *x, const MKL_INT *incx,
    const float *y, const MKL_INT *incy);
float mkl_blas_avx_xsdot(const MKL_INT *n, const float *x, const MKL_INT *incx,
    const float *y, const MKL_INT *incy);
float mkl_blas_avx2_xsdot(const MKL_INT *n, const float *x, const MKL_INT *incx,
    const float *y, const MKL_INT *incy);
float mkl_blas_avx512_xsdot(const MKL_INT *n, const float *x,
    const MKL_INT *incx, const float *y, const MKL_INT *incy);

void mkl_blas_sse2_xsgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_ssse3_xsgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_sse42_xsgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_avx_xsgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_avx2_xsgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_avx512_xsgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);

void mkl_blas_sse2_xsgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_ssse3_xsgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_sse42_xsgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_avx_xsgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_avx2_xsgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_avx512_xsgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);

void mkl_blas_sse2_xsgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, const float *x,
    const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void mkl_blas_ssse3_xsgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, const float *x,
    const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void mkl_blas_sse42_xsgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, const float *x,
    const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void mkl_blas_avx_xsgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, const float *x,
    const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void mkl_blas_avx2_xsgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, const float *x,
    const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void mkl_blas_avx512_xsgemv(const char *trans, const MKL_INT *m,
    const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
    const float *x, const MKL_INT *incx, const float *beta, float *y,
    const MKL_INT *incy);

void mkl_blas_sse2_xssymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
    const float *b, const MKL_INT *ldb, const float *beta, float *c,
    const MKL_INT *ldc);
void mkl_blas_ssse3_xssymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
    const float *b, const MKL_INT *ldb, const float *beta, float *c,
    const MKL_INT *ldc);
void mkl_blas_sse42_xssymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
    const float *b, const MKL_INT *ldb, const float *beta, float *c,
    const MKL_INT *ldc);
void mkl_blas_avx_xssymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
    const float *b, const MKL_INT *ldb, const float *beta, float *c,
    const MKL_INT *ldc);
void mkl_blas_avx2_xssymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
    const float *b, const MKL_INT *ldb, const float *beta, float *c,
    const MKL_INT *ldc);
void mkl_blas_avx512_xssymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
    const float *b, const MKL_INT *ldb, const float *beta, float *c,
    const MKL_INT *ldc);

void mkl_blas_sse2_xssyr(const char *uplo, const MKL_INT *n, const float *alpha,
    const float *x, const MKL_INT *incx, float *a, const MKL_INT *lda);
void mkl_blas_ssse3_xssyr(const char *uplo, const MKL_INT *n, const float *alpha,
    const float *x, const MKL_INT *incx, float *a, const MKL_INT *lda);
void mkl_blas_sse42_xssyr(const char *uplo, const MKL_INT *n, const float *alpha,
    const float *x, const MKL_INT *incx, float *a, const MKL_INT *lda);
void mkl_blas_avx_xssyr(const char *uplo, const MKL_INT *n, const float *alpha,
    const float *x, const MKL_INT *incx, float *a, const MKL_INT *lda);
void mkl_blas_avx2_xssyr(const char *uplo, const MKL_INT *n, const float *alpha,
    const float *x, const MKL_INT *incx, float *a, const MKL_INT *lda);
void mkl_blas_avx512_xssyr(const char *uplo, const MKL_INT *n,
    const float *alpha, const float *x, const MKL_INT *incx, float *a,
    const MKL_INT *lda);

void mkl_blas_sse2_xssyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *lda,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_ssse3_xssyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *lda,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_sse42_xssyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *lda,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_avx_xssyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *lda,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_avx2_xssyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *lda,
    const float *beta, float *c, const MKL_INT *ldc);
void mkl_blas_avx512_xssyrk(const char *uplo, const char *trans,
    const MKL_INT *n, const MKL_INT *k, const float *alpha, const float *a,
    const MKL_INT *lda, const float *beta, float *c, const MKL_INT *ldc);

void mkl_blas_sse2_xstrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const float *alpha,
    const float *a, const MKL_INT *lda, float *b, const MKL_INT *ldb);
void mkl_blas_ssse3_xstrmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, float *b,
    const MKL_INT *ldb);
void mkl_blas_sse42_xstrmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, float *b,
    const MKL_INT *ldb);
void mkl_blas_avx_xstrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const float *alpha,
    const float *a, const MKL_INT *lda, float *b, const MKL_INT *ldb);
void mkl_blas_avx2_xstrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const float *alpha,
    const float *a, const MKL_INT *lda, float *b, const MKL_INT *ldb);
void mkl_blas_avx512_xstrmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, float *b,
    const MKL_INT *ldb);




IppStatus mkl_dft_sse2_ippsSortRadixAscend_16s_I(Ipp16s *pSrcDst, Ipp16s *pTmp,
    Ipp32s len);
IppStatus mkl_dft_ssse3_ippsSortRadixAscend_16s_I(Ipp16s *pSrcDst, Ipp16s *pTmp,
    Ipp32s len);
IppStatus mkl_dft_sse42_ippsSortRadixAscend_16s_I(Ipp16s *pSrcDst, Ipp16s *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx_ippsSortRadixAscend_16s_I(Ipp16s *pSrcDst, Ipp16s *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx2_ippsSortRadixAscend_16s_I(Ipp16s *pSrcDst, Ipp16s *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx512_ippsSortRadixAscend_16s_I(Ipp16s *pSrcDst, Ipp16s *pTmp,
    Ipp32s len);

IppStatus mkl_dft_sse2_ippsSortRadixAscend_16u_I(Ipp16u *pSrcDst, Ipp16u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_ssse3_ippsSortRadixAscend_16u_I(Ipp16u *pSrcDst, Ipp16u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_sse42_ippsSortRadixAscend_16u_I(Ipp16u *pSrcDst, Ipp16u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx_ippsSortRadixAscend_16u_I(Ipp16u *pSrcDst, Ipp16u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx2_ippsSortRadixAscend_16u_I(Ipp16u *pSrcDst, Ipp16u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx512_ippsSortRadixAscend_16u_I(Ipp16u *pSrcDst, Ipp16u *pTmp,
    Ipp32s len);

IppStatus mkl_dft_sse2_ippsSortRadixAscend_32f_I(Ipp32f *pSrcDst, Ipp32f *pTmp,
    Ipp32s len);
IppStatus mkl_dft_ssse3_ippsSortRadixAscend_32f_I(Ipp32f *pSrcDst, Ipp32f *pTmp,
    Ipp32s len);
IppStatus mkl_dft_sse42_ippsSortRadixAscend_32f_I(Ipp32f *pSrcDst, Ipp32f *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx_ippsSortRadixAscend_32f_I(Ipp32f *pSrcDst, Ipp32f *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx2_ippsSortRadixAscend_32f_I(Ipp32f *pSrcDst, Ipp32f *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx512_ippsSortRadixAscend_32f_I(Ipp32f *pSrcDst, Ipp32f *pTmp,
    Ipp32s len);

IppStatus mkl_dft_sse2_ippsSortRadixAscend_32s_I(Ipp32s *pSrcDst, Ipp32s *pTmp,
    Ipp32s len);
IppStatus mkl_dft_ssse3_ippsSortRadixAscend_32s_I(Ipp32s *pSrcDst, Ipp32s *pTmp,
    Ipp32s len);
IppStatus mkl_dft_sse42_ippsSortRadixAscend_32s_I(Ipp32s *pSrcDst, Ipp32s *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx_ippsSortRadixAscend_32s_I(Ipp32s *pSrcDst, Ipp32s *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx2_ippsSortRadixAscend_32s_I(Ipp32s *pSrcDst, Ipp32s *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx512_ippsSortRadixAscend_32s_I(Ipp32s *pSrcDst, Ipp32s *pTmp,
    Ipp32s len);

IppStatus mkl_dft_sse2_ippsSortRadixAscend_32u_I(Ipp32u *pSrcDst, Ipp32u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_ssse3_ippsSortRadixAscend_32u_I(Ipp32u *pSrcDst, Ipp32u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_sse42_ippsSortRadixAscend_32u_I(Ipp32u *pSrcDst, Ipp32u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx_ippsSortRadixAscend_32u_I(Ipp32u *pSrcDst, Ipp32u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx2_ippsSortRadixAscend_32u_I(Ipp32u *pSrcDst, Ipp32u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx512_ippsSortRadixAscend_32u_I(Ipp32u *pSrcDst, Ipp32u *pTmp,
    Ipp32s len);

IppStatus mkl_dft_sse2_ippsSortRadixAscend_64f_I(Ipp64f *pSrcDst, Ipp64f *pTmp,
    Ipp32s len);
IppStatus mkl_dft_ssse3_ippsSortRadixAscend_64f_I(Ipp64f *pSrcDst, Ipp64f *pTmp,
    Ipp32s len);
IppStatus mkl_dft_sse42_ippsSortRadixAscend_64f_I(Ipp64f *pSrcDst, Ipp64f *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx_ippsSortRadixAscend_64f_I(Ipp64f *pSrcDst, Ipp64f *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx2_ippsSortRadixAscend_64f_I(Ipp64f *pSrcDst, Ipp64f *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx512_ippsSortRadixAscend_64f_I(Ipp64f *pSrcDst, Ipp64f *pTmp,
    Ipp32s len);

IppStatus mkl_dft_sse2_ippsSortRadixAscend_8u_I(Ipp8u *pSrcDst, Ipp8u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_ssse3_ippsSortRadixAscend_8u_I(Ipp8u *pSrcDst, Ipp8u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_sse42_ippsSortRadixAscend_8u_I(Ipp8u *pSrcDst, Ipp8u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx_ippsSortRadixAscend_8u_I(Ipp8u *pSrcDst, Ipp8u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx2_ippsSortRadixAscend_8u_I(Ipp8u *pSrcDst, Ipp8u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx512_ippsSortRadixAscend_8u_I(Ipp8u *pSrcDst, Ipp8u *pTmp,
    Ipp32s len);

IppStatus mkl_dft_sse2_ippsSortRadixDescend_16s_I(Ipp16s *pSrcDst, Ipp16s *pTmp,
    Ipp32s len);
IppStatus mkl_dft_ssse3_ippsSortRadixDescend_16s_I(Ipp16s *pSrcDst, Ipp16s *pTmp,
    Ipp32s len);
IppStatus mkl_dft_sse42_ippsSortRadixDescend_16s_I(Ipp16s *pSrcDst, Ipp16s *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx_ippsSortRadixDescend_16s_I(Ipp16s *pSrcDst, Ipp16s *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx2_ippsSortRadixDescend_16s_I(Ipp16s *pSrcDst, Ipp16s *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx512_ippsSortRadixDescend_16s_I(Ipp16s *pSrcDst,
    Ipp16s *pTmp, Ipp32s len);

IppStatus mkl_dft_sse2_ippsSortRadixDescend_16u_I(Ipp16u *pSrcDst, Ipp16u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_ssse3_ippsSortRadixDescend_16u_I(Ipp16u *pSrcDst, Ipp16u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_sse42_ippsSortRadixDescend_16u_I(Ipp16u *pSrcDst, Ipp16u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx_ippsSortRadixDescend_16u_I(Ipp16u *pSrcDst, Ipp16u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx2_ippsSortRadixDescend_16u_I(Ipp16u *pSrcDst, Ipp16u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx512_ippsSortRadixDescend_16u_I(Ipp16u *pSrcDst,
    Ipp16u *pTmp, Ipp32s len);

IppStatus mkl_dft_sse2_ippsSortRadixDescend_32f_I(Ipp32f *pSrcDst, Ipp32f *pTmp,
    Ipp32s len);
IppStatus mkl_dft_ssse3_ippsSortRadixDescend_32f_I(Ipp32f *pSrcDst, Ipp32f *pTmp,
    Ipp32s len);
IppStatus mkl_dft_sse42_ippsSortRadixDescend_32f_I(Ipp32f *pSrcDst, Ipp32f *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx_ippsSortRadixDescend_32f_I(Ipp32f *pSrcDst, Ipp32f *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx2_ippsSortRadixDescend_32f_I(Ipp32f *pSrcDst, Ipp32f *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx512_ippsSortRadixDescend_32f_I(Ipp32f *pSrcDst,
    Ipp32f *pTmp, Ipp32s len);

IppStatus mkl_dft_sse2_ippsSortRadixDescend_32s_I(Ipp32s *pSrcDst, Ipp32s *pTmp,
    Ipp32s len);
IppStatus mkl_dft_ssse3_ippsSortRadixDescend_32s_I(Ipp32s *pSrcDst, Ipp32s *pTmp,
    Ipp32s len);
IppStatus mkl_dft_sse42_ippsSortRadixDescend_32s_I(Ipp32s *pSrcDst, Ipp32s *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx_ippsSortRadixDescend_32s_I(Ipp32s *pSrcDst, Ipp32s *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx2_ippsSortRadixDescend_32s_I(Ipp32s *pSrcDst, Ipp32s *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx512_ippsSortRadixDescend_32s_I(Ipp32s *pSrcDst,
    Ipp32s *pTmp, Ipp32s len);

IppStatus mkl_dft_sse2_ippsSortRadixDescend_32u_I(Ipp32u *pSrcDst, Ipp32u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_ssse3_ippsSortRadixDescend_32u_I(Ipp32u *pSrcDst, Ipp32u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_sse42_ippsSortRadixDescend_32u_I(Ipp32u *pSrcDst, Ipp32u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx_ippsSortRadixDescend_32u_I(Ipp32u *pSrcDst, Ipp32u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx2_ippsSortRadixDescend_32u_I(Ipp32u *pSrcDst, Ipp32u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx512_ippsSortRadixDescend_32u_I(Ipp32u *pSrcDst,
    Ipp32u *pTmp, Ipp32s len);

IppStatus mkl_dft_sse2_ippsSortRadixDescend_64f_I(Ipp64f *pSrcDst, Ipp64f *pTmp,
    Ipp32s len);
IppStatus mkl_dft_ssse3_ippsSortRadixDescend_64f_I(Ipp64f *pSrcDst, Ipp64f *pTmp,
    Ipp32s len);
IppStatus mkl_dft_sse42_ippsSortRadixDescend_64f_I(Ipp64f *pSrcDst, Ipp64f *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx_ippsSortRadixDescend_64f_I(Ipp64f *pSrcDst, Ipp64f *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx2_ippsSortRadixDescend_64f_I(Ipp64f *pSrcDst, Ipp64f *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx512_ippsSortRadixDescend_64f_I(Ipp64f *pSrcDst,
    Ipp64f *pTmp, Ipp32s len);

IppStatus mkl_dft_sse2_ippsSortRadixDescend_8u_I(Ipp8u *pSrcDst, Ipp8u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_ssse3_ippsSortRadixDescend_8u_I(Ipp8u *pSrcDst, Ipp8u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_sse42_ippsSortRadixDescend_8u_I(Ipp8u *pSrcDst, Ipp8u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx_ippsSortRadixDescend_8u_I(Ipp8u *pSrcDst, Ipp8u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx2_ippsSortRadixDescend_8u_I(Ipp8u *pSrcDst, Ipp8u *pTmp,
    Ipp32s len);
IppStatus mkl_dft_avx512_ippsSortRadixDescend_8u_I(Ipp8u *pSrcDst, Ipp8u *pTmp,
    Ipp32s len);



void mkl_lapack_sse2_dgels(const char* trans, const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* nrhs, double* a, const MKL_INT* lda, double* b,
    const MKL_INT* ldb, double* work, const MKL_INT* lwork, MKL_INT* info ,
    int itrans);
void mkl_lapack_ssse3_dgels(const char* trans, const MKL_INT* m,
    const MKL_INT* n, const MKL_INT* nrhs, double* a, const MKL_INT* lda,
    double* b, const MKL_INT* ldb, double* work, const MKL_INT* lwork,
    MKL_INT* info , int itrans);
void mkl_lapack_sse42_dgels(const char* trans, const MKL_INT* m,
    const MKL_INT* n, const MKL_INT* nrhs, double* a, const MKL_INT* lda,
    double* b, const MKL_INT* ldb, double* work, const MKL_INT* lwork,
    MKL_INT* info , int itrans);
void mkl_lapack_avx_dgels(const char* trans, const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* nrhs, double* a, const MKL_INT* lda, double* b,
    const MKL_INT* ldb, double* work, const MKL_INT* lwork, MKL_INT* info ,
    int itrans);
void mkl_lapack_avx2_dgels(const char* trans, const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* nrhs, double* a, const MKL_INT* lda, double* b,
    const MKL_INT* ldb, double* work, const MKL_INT* lwork, MKL_INT* info ,
    int itrans);
void mkl_lapack_avx512_dgels(const char* trans, const MKL_INT* m,
    const MKL_INT* n, const MKL_INT* nrhs, double* a, const MKL_INT* lda,
    double* b, const MKL_INT* ldb, double* work, const MKL_INT* lwork,
    MKL_INT* info , int itrans);

void mkl_lapack_sse2_dgeqp3(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* jpvt, double* tau, double* work,
    const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_ssse3_dgeqp3(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* jpvt, double* tau, double* work,
    const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_sse42_dgeqp3(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* jpvt, double* tau, double* work,
    const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_avx_dgeqp3(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* jpvt, double* tau, double* work,
    const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_avx2_dgeqp3(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* jpvt, double* tau, double* work,
    const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_avx512_dgeqp3(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* jpvt, double* tau, double* work,
    const MKL_INT* lwork, MKL_INT* info );

void mkl_lapack_sse2_dgeqrf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, double* tau, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void mkl_lapack_ssse3_dgeqrf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, double* tau, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void mkl_lapack_sse42_dgeqrf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, double* tau, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void mkl_lapack_avx_dgeqrf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, double* tau, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void mkl_lapack_avx2_dgeqrf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, double* tau, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void mkl_lapack_avx512_dgeqrf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, double* tau, double* work, const MKL_INT* lwork,
    MKL_INT* info );

void mkl_lapack_sse2_dgerqf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, double* tau, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void mkl_lapack_ssse3_dgerqf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, double* tau, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void mkl_lapack_sse42_dgerqf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, double* tau, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void mkl_lapack_avx_dgerqf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, double* tau, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void mkl_lapack_avx2_dgerqf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, double* tau, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void mkl_lapack_avx512_dgerqf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, double* tau, double* work, const MKL_INT* lwork,
    MKL_INT* info );

void mkl_lapack_sse2_dgesvd(const char* jobu, const char* jobvt,
    const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, double* s,
    double* u, const MKL_INT* ldu, double* vt, const MKL_INT* ldvt, double* work,
    const MKL_INT* lwork, MKL_INT* info , int ijobu, int ijobvt);
void mkl_lapack_ssse3_dgesvd(const char* jobu, const char* jobvt,
    const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, double* s,
    double* u, const MKL_INT* ldu, double* vt, const MKL_INT* ldvt, double* work,
    const MKL_INT* lwork, MKL_INT* info , int ijobu, int ijobvt);
void mkl_lapack_sse42_dgesvd(const char* jobu, const char* jobvt,
    const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, double* s,
    double* u, const MKL_INT* ldu, double* vt, const MKL_INT* ldvt, double* work,
    const MKL_INT* lwork, MKL_INT* info , int ijobu, int ijobvt);
void mkl_lapack_avx_dgesvd(const char* jobu, const char* jobvt, const MKL_INT* m,
    const MKL_INT* n, double* a, const MKL_INT* lda, double* s, double* u,
    const MKL_INT* ldu, double* vt, const MKL_INT* ldvt, double* work,
    const MKL_INT* lwork, MKL_INT* info , int ijobu, int ijobvt);
void mkl_lapack_avx2_dgesvd(const char* jobu, const char* jobvt,
    const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, double* s,
    double* u, const MKL_INT* ldu, double* vt, const MKL_INT* ldvt, double* work,
    const MKL_INT* lwork, MKL_INT* info , int ijobu, int ijobvt);
void mkl_lapack_avx512_dgesvd(const char* jobu, const char* jobvt,
    const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, double* s,
    double* u, const MKL_INT* ldu, double* vt, const MKL_INT* ldvt, double* work,
    const MKL_INT* lwork, MKL_INT* info , int ijobu, int ijobvt);

void mkl_lapack_sse2_dgetrf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* ipiv, MKL_INT* info );
void mkl_lapack_ssse3_dgetrf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* ipiv, MKL_INT* info );
void mkl_lapack_sse42_dgetrf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* ipiv, MKL_INT* info );
void mkl_lapack_avx_dgetrf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* ipiv, MKL_INT* info );
void mkl_lapack_avx2_dgetrf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* ipiv, MKL_INT* info );
void mkl_lapack_avx512_dgetrf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* ipiv, MKL_INT* info );

void mkl_lapack_sse2_dgetrs(const char* trans, const MKL_INT* n,
    const MKL_INT* nrhs, const double* a, const MKL_INT* lda,
    const MKL_INT* ipiv, double* b, const MKL_INT* ldb, MKL_INT* info ,
    int itrans);
void mkl_lapack_ssse3_dgetrs(const char* trans, const MKL_INT* n,
    const MKL_INT* nrhs, const double* a, const MKL_INT* lda,
    const MKL_INT* ipiv, double* b, const MKL_INT* ldb, MKL_INT* info ,
    int itrans);
void mkl_lapack_sse42_dgetrs(const char* trans, const MKL_INT* n,
    const MKL_INT* nrhs, const double* a, const MKL_INT* lda,
    const MKL_INT* ipiv, double* b, const MKL_INT* ldb, MKL_INT* info ,
    int itrans);
void mkl_lapack_avx_dgetrs(const char* trans, const MKL_INT* n,
    const MKL_INT* nrhs, const double* a, const MKL_INT* lda,
    const MKL_INT* ipiv, double* b, const MKL_INT* ldb, MKL_INT* info ,
    int itrans);
void mkl_lapack_avx2_dgetrs(const char* trans, const MKL_INT* n,
    const MKL_INT* nrhs, const double* a, const MKL_INT* lda,
    const MKL_INT* ipiv, double* b, const MKL_INT* ldb, MKL_INT* info ,
    int itrans);
void mkl_lapack_avx512_dgetrs(const char* trans, const MKL_INT* n,
    const MKL_INT* nrhs, const double* a, const MKL_INT* lda,
    const MKL_INT* ipiv, double* b, const MKL_INT* ldb, MKL_INT* info ,
    int itrans);

double mkl_lapack_sse2_dlange(const char* norm, const MKL_INT* m,
    const MKL_INT* n, const double* a, const MKL_INT* lda, double* work ,
    int inorm);
double mkl_lapack_ssse3_dlange(const char* norm, const MKL_INT* m,
    const MKL_INT* n, const double* a, const MKL_INT* lda, double* work ,
    int inorm);
double mkl_lapack_sse42_dlange(const char* norm, const MKL_INT* m,
    const MKL_INT* n, const double* a, const MKL_INT* lda, double* work ,
    int inorm);
double mkl_lapack_avx_dlange(const char* norm, const MKL_INT* m,
    const MKL_INT* n, const double* a, const MKL_INT* lda, double* work ,
    int inorm);
double mkl_lapack_avx2_dlange(const char* norm, const MKL_INT* m,
    const MKL_INT* n, const double* a, const MKL_INT* lda, double* work ,
    int inorm);
double mkl_lapack_avx512_dlange(const char* norm, const MKL_INT* m,
    const MKL_INT* n, const double* a, const MKL_INT* lda, double* work ,
    int inorm);

void mkl_lapack_sse2_dlarnv(const MKL_INT* idist, MKL_INT* iseed,
    const MKL_INT* n, double* x );
void mkl_lapack_ssse3_dlarnv(const MKL_INT* idist, MKL_INT* iseed,
    const MKL_INT* n, double* x );
void mkl_lapack_sse42_dlarnv(const MKL_INT* idist, MKL_INT* iseed,
    const MKL_INT* n, double* x );
void mkl_lapack_avx_dlarnv(const MKL_INT* idist, MKL_INT* iseed,
    const MKL_INT* n, double* x );
void mkl_lapack_avx2_dlarnv(const MKL_INT* idist, MKL_INT* iseed,
    const MKL_INT* n, double* x );
void mkl_lapack_avx512_dlarnv(const MKL_INT* idist, MKL_INT* iseed,
    const MKL_INT* n, double* x );

void mkl_lapack_sse2_dorgqr(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
    double* a, const MKL_INT* lda, const double* tau, double* work,
    const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_ssse3_dorgqr(const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* k, double* a, const MKL_INT* lda, const double* tau,
    double* work, const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_sse42_dorgqr(const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* k, double* a, const MKL_INT* lda, const double* tau,
    double* work, const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_avx_dorgqr(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
    double* a, const MKL_INT* lda, const double* tau, double* work,
    const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_avx2_dorgqr(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
    double* a, const MKL_INT* lda, const double* tau, double* work,
    const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_avx512_dorgqr(const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* k, double* a, const MKL_INT* lda, const double* tau,
    double* work, const MKL_INT* lwork, MKL_INT* info );

void mkl_lapack_sse2_dorgrq(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
    double* a, const MKL_INT* lda, const double* tau, double* work,
    const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_ssse3_dorgrq(const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* k, double* a, const MKL_INT* lda, const double* tau,
    double* work, const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_sse42_dorgrq(const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* k, double* a, const MKL_INT* lda, const double* tau,
    double* work, const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_avx_dorgrq(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
    double* a, const MKL_INT* lda, const double* tau, double* work,
    const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_avx2_dorgrq(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
    double* a, const MKL_INT* lda, const double* tau, double* work,
    const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_avx512_dorgrq(const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* k, double* a, const MKL_INT* lda, const double* tau,
    double* work, const MKL_INT* lwork, MKL_INT* info );

void mkl_lapack_sse2_dormqr(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const double* a,
    const MKL_INT* lda, const double* tau, double* c, const MKL_INT* ldc,
    double* work, const MKL_INT* lwork, MKL_INT* info , int iside, int itrans);
void mkl_lapack_ssse3_dormqr(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const double* a,
    const MKL_INT* lda, const double* tau, double* c, const MKL_INT* ldc,
    double* work, const MKL_INT* lwork, MKL_INT* info , int iside, int itrans);
void mkl_lapack_sse42_dormqr(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const double* a,
    const MKL_INT* lda, const double* tau, double* c, const MKL_INT* ldc,
    double* work, const MKL_INT* lwork, MKL_INT* info , int iside, int itrans);
void mkl_lapack_avx_dormqr(const char* side, const char* trans, const MKL_INT* m,
    const MKL_INT* n, const MKL_INT* k, const double* a, const MKL_INT* lda,
    const double* tau, double* c, const MKL_INT* ldc, double* work,
    const MKL_INT* lwork, MKL_INT* info , int iside, int itrans);
void mkl_lapack_avx2_dormqr(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const double* a,
    const MKL_INT* lda, const double* tau, double* c, const MKL_INT* ldc,
    double* work, const MKL_INT* lwork, MKL_INT* info , int iside, int itrans);
void mkl_lapack_avx512_dormqr(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const double* a,
    const MKL_INT* lda, const double* tau, double* c, const MKL_INT* ldc,
    double* work, const MKL_INT* lwork, MKL_INT* info , int iside, int itrans);

void mkl_lapack_sse2_dormrq(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const double* a,
    const MKL_INT* lda, const double* tau, double* c, const MKL_INT* ldc,
    double* work, const MKL_INT* lwork, MKL_INT* info , int iside, int itrans);
void mkl_lapack_ssse3_dormrq(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const double* a,
    const MKL_INT* lda, const double* tau, double* c, const MKL_INT* ldc,
    double* work, const MKL_INT* lwork, MKL_INT* info , int iside, int itrans);
void mkl_lapack_sse42_dormrq(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const double* a,
    const MKL_INT* lda, const double* tau, double* c, const MKL_INT* ldc,
    double* work, const MKL_INT* lwork, MKL_INT* info , int iside, int itrans);
void mkl_lapack_avx_dormrq(const char* side, const char* trans, const MKL_INT* m,
    const MKL_INT* n, const MKL_INT* k, const double* a, const MKL_INT* lda,
    const double* tau, double* c, const MKL_INT* ldc, double* work,
    const MKL_INT* lwork, MKL_INT* info , int iside, int itrans);
void mkl_lapack_avx2_dormrq(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const double* a,
    const MKL_INT* lda, const double* tau, double* c, const MKL_INT* ldc,
    double* work, const MKL_INT* lwork, MKL_INT* info , int iside, int itrans);
void mkl_lapack_avx512_dormrq(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const double* a,
    const MKL_INT* lda, const double* tau, double* c, const MKL_INT* ldc,
    double* work, const MKL_INT* lwork, MKL_INT* info , int iside, int itrans);

void mkl_lapack_sse2_dpftrf(const char* transr, const char* uplo,
    const MKL_INT* n, double* a, MKL_INT* info , int itransr, int iuplo);
void mkl_lapack_ssse3_dpftrf(const char* transr, const char* uplo,
    const MKL_INT* n, double* a, MKL_INT* info , int itransr, int iuplo);
void mkl_lapack_sse42_dpftrf(const char* transr, const char* uplo,
    const MKL_INT* n, double* a, MKL_INT* info , int itransr, int iuplo);
void mkl_lapack_avx_dpftrf(const char* transr, const char* uplo,
    const MKL_INT* n, double* a, MKL_INT* info , int itransr, int iuplo);
void mkl_lapack_avx2_dpftrf(const char* transr, const char* uplo,
    const MKL_INT* n, double* a, MKL_INT* info , int itransr, int iuplo);
void mkl_lapack_avx512_dpftrf(const char* transr, const char* uplo,
    const MKL_INT* n, double* a, MKL_INT* info , int itransr, int iuplo);

void mkl_lapack_sse2_dpotrf(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* info , int iuplo);
void mkl_lapack_ssse3_dpotrf(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* info , int iuplo);
void mkl_lapack_sse42_dpotrf(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* info , int iuplo);
void mkl_lapack_avx_dpotrf(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* info , int iuplo);
void mkl_lapack_avx2_dpotrf(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* info , int iuplo);
void mkl_lapack_avx512_dpotrf(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* info , int iuplo);

void mkl_lapack_sse2_dpotri(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* info , int iuplo);
void mkl_lapack_ssse3_dpotri(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* info , int iuplo);
void mkl_lapack_sse42_dpotri(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* info , int iuplo);
void mkl_lapack_avx_dpotri(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* info , int iuplo);
void mkl_lapack_avx2_dpotri(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* info , int iuplo);
void mkl_lapack_avx512_dpotri(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* info , int iuplo);

void mkl_lapack_sse2_dpotrs(const char* uplo, const MKL_INT* n,
    const MKL_INT* nrhs, const double* a, const MKL_INT* lda, double* b,
    const MKL_INT* ldb, MKL_INT* info , int iuplo);
void mkl_lapack_ssse3_dpotrs(const char* uplo, const MKL_INT* n,
    const MKL_INT* nrhs, const double* a, const MKL_INT* lda, double* b,
    const MKL_INT* ldb, MKL_INT* info , int iuplo);
void mkl_lapack_sse42_dpotrs(const char* uplo, const MKL_INT* n,
    const MKL_INT* nrhs, const double* a, const MKL_INT* lda, double* b,
    const MKL_INT* ldb, MKL_INT* info , int iuplo);
void mkl_lapack_avx_dpotrs(const char* uplo, const MKL_INT* n,
    const MKL_INT* nrhs, const double* a, const MKL_INT* lda, double* b,
    const MKL_INT* ldb, MKL_INT* info , int iuplo);
void mkl_lapack_avx2_dpotrs(const char* uplo, const MKL_INT* n,
    const MKL_INT* nrhs, const double* a, const MKL_INT* lda, double* b,
    const MKL_INT* ldb, MKL_INT* info , int iuplo);
void mkl_lapack_avx512_dpotrs(const char* uplo, const MKL_INT* n,
    const MKL_INT* nrhs, const double* a, const MKL_INT* lda, double* b,
    const MKL_INT* ldb, MKL_INT* info , int iuplo);

void mkl_lapack_sse2_dpptrf(const char* uplo, const MKL_INT* n, double* ap,
    MKL_INT* info , int iuplo);
void mkl_lapack_ssse3_dpptrf(const char* uplo, const MKL_INT* n, double* ap,
    MKL_INT* info , int iuplo);
void mkl_lapack_sse42_dpptrf(const char* uplo, const MKL_INT* n, double* ap,
    MKL_INT* info , int iuplo);
void mkl_lapack_avx_dpptrf(const char* uplo, const MKL_INT* n, double* ap,
    MKL_INT* info , int iuplo);
void mkl_lapack_avx2_dpptrf(const char* uplo, const MKL_INT* n, double* ap,
    MKL_INT* info , int iuplo);
void mkl_lapack_avx512_dpptrf(const char* uplo, const MKL_INT* n, double* ap,
    MKL_INT* info , int iuplo);

void mkl_lapack_sse2_dpstrf(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* piv, MKL_INT* rank, const double* tol,
    double* work, MKL_INT* info , int iuplo);
void mkl_lapack_ssse3_dpstrf(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* piv, MKL_INT* rank, const double* tol,
    double* work, MKL_INT* info , int iuplo);
void mkl_lapack_sse42_dpstrf(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* piv, MKL_INT* rank, const double* tol,
    double* work, MKL_INT* info , int iuplo);
void mkl_lapack_avx_dpstrf(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* piv, MKL_INT* rank, const double* tol,
    double* work, MKL_INT* info , int iuplo);
void mkl_lapack_avx2_dpstrf(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* piv, MKL_INT* rank, const double* tol,
    double* work, MKL_INT* info , int iuplo);
void mkl_lapack_avx512_dpstrf(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* piv, MKL_INT* rank, const double* tol,
    double* work, MKL_INT* info , int iuplo);

void mkl_lapack_sse2_dspevd(const char* jobz, const char* uplo, const MKL_INT* n,
    double* ap, double* w, double* z, const MKL_INT* ldz, double* work,
    const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info ,
    int ijobz, int iuplo);
void mkl_lapack_ssse3_dspevd(const char* jobz, const char* uplo,
    const MKL_INT* n, double* ap, double* w, double* z, const MKL_INT* ldz,
    double* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork,
    MKL_INT* info , int ijobz, int iuplo);
void mkl_lapack_sse42_dspevd(const char* jobz, const char* uplo,
    const MKL_INT* n, double* ap, double* w, double* z, const MKL_INT* ldz,
    double* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork,
    MKL_INT* info , int ijobz, int iuplo);
void mkl_lapack_avx_dspevd(const char* jobz, const char* uplo, const MKL_INT* n,
    double* ap, double* w, double* z, const MKL_INT* ldz, double* work,
    const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info ,
    int ijobz, int iuplo);
void mkl_lapack_avx2_dspevd(const char* jobz, const char* uplo, const MKL_INT* n,
    double* ap, double* w, double* z, const MKL_INT* ldz, double* work,
    const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info ,
    int ijobz, int iuplo);
void mkl_lapack_avx512_dspevd(const char* jobz, const char* uplo,
    const MKL_INT* n, double* ap, double* w, double* z, const MKL_INT* ldz,
    double* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork,
    MKL_INT* info , int ijobz, int iuplo);

void mkl_lapack_sse2_dsyev(const char* jobz, const char* uplo, const MKL_INT* n,
    double* a, const MKL_INT* lda, double* w, double* work, const MKL_INT* lwork,
    MKL_INT* info , int ijobz, int iuplo);
void mkl_lapack_ssse3_dsyev(const char* jobz, const char* uplo, const MKL_INT* n,
    double* a, const MKL_INT* lda, double* w, double* work, const MKL_INT* lwork,
    MKL_INT* info , int ijobz, int iuplo);
void mkl_lapack_sse42_dsyev(const char* jobz, const char* uplo, const MKL_INT* n,
    double* a, const MKL_INT* lda, double* w, double* work, const MKL_INT* lwork,
    MKL_INT* info , int ijobz, int iuplo);
void mkl_lapack_avx_dsyev(const char* jobz, const char* uplo, const MKL_INT* n,
    double* a, const MKL_INT* lda, double* w, double* work, const MKL_INT* lwork,
    MKL_INT* info , int ijobz, int iuplo);
void mkl_lapack_avx2_dsyev(const char* jobz, const char* uplo, const MKL_INT* n,
    double* a, const MKL_INT* lda, double* w, double* work, const MKL_INT* lwork,
    MKL_INT* info , int ijobz, int iuplo);
void mkl_lapack_avx512_dsyev(const char* jobz, const char* uplo,
    const MKL_INT* n, double* a, const MKL_INT* lda, double* w, double* work,
    const MKL_INT* lwork, MKL_INT* info , int ijobz, int iuplo);

void mkl_lapack_sse2_dsyevd(const char* jobz, const char* uplo, const MKL_INT* n,
    double* a, const MKL_INT* lda, double* w, double* work, const MKL_INT* lwork,
    MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info , int ijobz, int iuplo);
void mkl_lapack_ssse3_dsyevd(const char* jobz, const char* uplo,
    const MKL_INT* n, double* a, const MKL_INT* lda, double* w, double* work,
    const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info ,
    int ijobz, int iuplo);
void mkl_lapack_sse42_dsyevd(const char* jobz, const char* uplo,
    const MKL_INT* n, double* a, const MKL_INT* lda, double* w, double* work,
    const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info ,
    int ijobz, int iuplo);
void mkl_lapack_avx_dsyevd(const char* jobz, const char* uplo, const MKL_INT* n,
    double* a, const MKL_INT* lda, double* w, double* work, const MKL_INT* lwork,
    MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info , int ijobz, int iuplo);
void mkl_lapack_avx2_dsyevd(const char* jobz, const char* uplo, const MKL_INT* n,
    double* a, const MKL_INT* lda, double* w, double* work, const MKL_INT* lwork,
    MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info , int ijobz, int iuplo);
void mkl_lapack_avx512_dsyevd(const char* jobz, const char* uplo,
    const MKL_INT* n, double* a, const MKL_INT* lda, double* w, double* work,
    const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info ,
    int ijobz, int iuplo);

void mkl_lapack_sse2_dsyevr(const char* jobz, const char* range,
    const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda,
    const double* vl, const double* vu, const MKL_INT* il, const MKL_INT* iu,
    const double* abstol, MKL_INT* m, double* w, double* z, const MKL_INT* ldz,
    MKL_INT* isuppz, double* work, const MKL_INT* lwork, MKL_INT* iwork,
    const MKL_INT* liwork, MKL_INT* info , int ijobz, int irange, int iuplo);
void mkl_lapack_ssse3_dsyevr(const char* jobz, const char* range,
    const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda,
    const double* vl, const double* vu, const MKL_INT* il, const MKL_INT* iu,
    const double* abstol, MKL_INT* m, double* w, double* z, const MKL_INT* ldz,
    MKL_INT* isuppz, double* work, const MKL_INT* lwork, MKL_INT* iwork,
    const MKL_INT* liwork, MKL_INT* info , int ijobz, int irange, int iuplo);
void mkl_lapack_sse42_dsyevr(const char* jobz, const char* range,
    const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda,
    const double* vl, const double* vu, const MKL_INT* il, const MKL_INT* iu,
    const double* abstol, MKL_INT* m, double* w, double* z, const MKL_INT* ldz,
    MKL_INT* isuppz, double* work, const MKL_INT* lwork, MKL_INT* iwork,
    const MKL_INT* liwork, MKL_INT* info , int ijobz, int irange, int iuplo);
void mkl_lapack_avx_dsyevr(const char* jobz, const char* range, const char* uplo,
    const MKL_INT* n, double* a, const MKL_INT* lda, const double* vl,
    const double* vu, const MKL_INT* il, const MKL_INT* iu, const double* abstol,
    MKL_INT* m, double* w, double* z, const MKL_INT* ldz, MKL_INT* isuppz,
    double* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork,
    MKL_INT* info , int ijobz, int irange, int iuplo);
void mkl_lapack_avx2_dsyevr(const char* jobz, const char* range,
    const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda,
    const double* vl, const double* vu, const MKL_INT* il, const MKL_INT* iu,
    const double* abstol, MKL_INT* m, double* w, double* z, const MKL_INT* ldz,
    MKL_INT* isuppz, double* work, const MKL_INT* lwork, MKL_INT* iwork,
    const MKL_INT* liwork, MKL_INT* info , int ijobz, int irange, int iuplo);
void mkl_lapack_avx512_dsyevr(const char* jobz, const char* range,
    const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda,
    const double* vl, const double* vu, const MKL_INT* il, const MKL_INT* iu,
    const double* abstol, MKL_INT* m, double* w, double* z, const MKL_INT* ldz,
    MKL_INT* isuppz, double* work, const MKL_INT* lwork, MKL_INT* iwork,
    const MKL_INT* liwork, MKL_INT* info , int ijobz, int irange, int iuplo);

void mkl_lapack_sse2_dtrtrs(const char* uplo, const char* trans,
    const char* diag, const MKL_INT* n, const MKL_INT* nrhs, const double* a,
    const MKL_INT* lda, double* b, const MKL_INT* ldb, MKL_INT* info , int iuplo,
    int itrans, int idiag);
void mkl_lapack_ssse3_dtrtrs(const char* uplo, const char* trans,
    const char* diag, const MKL_INT* n, const MKL_INT* nrhs, const double* a,
    const MKL_INT* lda, double* b, const MKL_INT* ldb, MKL_INT* info , int iuplo,
    int itrans, int idiag);
void mkl_lapack_sse42_dtrtrs(const char* uplo, const char* trans,
    const char* diag, const MKL_INT* n, const MKL_INT* nrhs, const double* a,
    const MKL_INT* lda, double* b, const MKL_INT* ldb, MKL_INT* info , int iuplo,
    int itrans, int idiag);
void mkl_lapack_avx_dtrtrs(const char* uplo, const char* trans, const char* diag,
    const MKL_INT* n, const MKL_INT* nrhs, const double* a, const MKL_INT* lda,
    double* b, const MKL_INT* ldb, MKL_INT* info , int iuplo, int itrans,
    int idiag);
void mkl_lapack_avx2_dtrtrs(const char* uplo, const char* trans,
    const char* diag, const MKL_INT* n, const MKL_INT* nrhs, const double* a,
    const MKL_INT* lda, double* b, const MKL_INT* ldb, MKL_INT* info , int iuplo,
    int itrans, int idiag);
void mkl_lapack_avx512_dtrtrs(const char* uplo, const char* trans,
    const char* diag, const MKL_INT* n, const MKL_INT* nrhs, const double* a,
    const MKL_INT* lda, double* b, const MKL_INT* ldb, MKL_INT* info , int iuplo,
    int itrans, int idiag);

void mkl_lapack_sse2_sgels(const char* trans, const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* nrhs, float* a, const MKL_INT* lda, float* b,
    const MKL_INT* ldb, float* work, const MKL_INT* lwork, MKL_INT* info ,
    int itrans);
void mkl_lapack_ssse3_sgels(const char* trans, const MKL_INT* m,
    const MKL_INT* n, const MKL_INT* nrhs, float* a, const MKL_INT* lda,
    float* b, const MKL_INT* ldb, float* work, const MKL_INT* lwork,
    MKL_INT* info , int itrans);
void mkl_lapack_sse42_sgels(const char* trans, const MKL_INT* m,
    const MKL_INT* n, const MKL_INT* nrhs, float* a, const MKL_INT* lda,
    float* b, const MKL_INT* ldb, float* work, const MKL_INT* lwork,
    MKL_INT* info , int itrans);
void mkl_lapack_avx_sgels(const char* trans, const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* nrhs, float* a, const MKL_INT* lda, float* b,
    const MKL_INT* ldb, float* work, const MKL_INT* lwork, MKL_INT* info ,
    int itrans);
void mkl_lapack_avx2_sgels(const char* trans, const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* nrhs, float* a, const MKL_INT* lda, float* b,
    const MKL_INT* ldb, float* work, const MKL_INT* lwork, MKL_INT* info ,
    int itrans);
void mkl_lapack_avx512_sgels(const char* trans, const MKL_INT* m,
    const MKL_INT* n, const MKL_INT* nrhs, float* a, const MKL_INT* lda,
    float* b, const MKL_INT* ldb, float* work, const MKL_INT* lwork,
    MKL_INT* info , int itrans);

void mkl_lapack_sse2_sgeqp3(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* jpvt, float* tau, float* work,
    const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_ssse3_sgeqp3(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* jpvt, float* tau, float* work,
    const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_sse42_sgeqp3(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* jpvt, float* tau, float* work,
    const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_avx_sgeqp3(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* jpvt, float* tau, float* work,
    const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_avx2_sgeqp3(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* jpvt, float* tau, float* work,
    const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_avx512_sgeqp3(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* jpvt, float* tau, float* work,
    const MKL_INT* lwork, MKL_INT* info );

void mkl_lapack_sse2_sgeqrf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, float* tau, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void mkl_lapack_ssse3_sgeqrf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, float* tau, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void mkl_lapack_sse42_sgeqrf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, float* tau, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void mkl_lapack_avx_sgeqrf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, float* tau, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void mkl_lapack_avx2_sgeqrf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, float* tau, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void mkl_lapack_avx512_sgeqrf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, float* tau, float* work, const MKL_INT* lwork,
    MKL_INT* info );

void mkl_lapack_sse2_sgerqf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, float* tau, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void mkl_lapack_ssse3_sgerqf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, float* tau, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void mkl_lapack_sse42_sgerqf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, float* tau, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void mkl_lapack_avx_sgerqf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, float* tau, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void mkl_lapack_avx2_sgerqf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, float* tau, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void mkl_lapack_avx512_sgerqf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, float* tau, float* work, const MKL_INT* lwork,
    MKL_INT* info );

void mkl_lapack_sse2_sgesvd(const char* jobu, const char* jobvt,
    const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda, float* s,
    float* u, const MKL_INT* ldu, float* vt, const MKL_INT* ldvt, float* work,
    const MKL_INT* lwork, MKL_INT* info , int ijobu, int ijobvt);
void mkl_lapack_ssse3_sgesvd(const char* jobu, const char* jobvt,
    const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda, float* s,
    float* u, const MKL_INT* ldu, float* vt, const MKL_INT* ldvt, float* work,
    const MKL_INT* lwork, MKL_INT* info , int ijobu, int ijobvt);
void mkl_lapack_sse42_sgesvd(const char* jobu, const char* jobvt,
    const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda, float* s,
    float* u, const MKL_INT* ldu, float* vt, const MKL_INT* ldvt, float* work,
    const MKL_INT* lwork, MKL_INT* info , int ijobu, int ijobvt);
void mkl_lapack_avx_sgesvd(const char* jobu, const char* jobvt, const MKL_INT* m,
    const MKL_INT* n, float* a, const MKL_INT* lda, float* s, float* u,
    const MKL_INT* ldu, float* vt, const MKL_INT* ldvt, float* work,
    const MKL_INT* lwork, MKL_INT* info , int ijobu, int ijobvt);
void mkl_lapack_avx2_sgesvd(const char* jobu, const char* jobvt,
    const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda, float* s,
    float* u, const MKL_INT* ldu, float* vt, const MKL_INT* ldvt, float* work,
    const MKL_INT* lwork, MKL_INT* info , int ijobu, int ijobvt);
void mkl_lapack_avx512_sgesvd(const char* jobu, const char* jobvt,
    const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda, float* s,
    float* u, const MKL_INT* ldu, float* vt, const MKL_INT* ldvt, float* work,
    const MKL_INT* lwork, MKL_INT* info , int ijobu, int ijobvt);

void mkl_lapack_sse2_sgetrf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* ipiv, MKL_INT* info );
void mkl_lapack_ssse3_sgetrf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* ipiv, MKL_INT* info );
void mkl_lapack_sse42_sgetrf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* ipiv, MKL_INT* info );
void mkl_lapack_avx_sgetrf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* ipiv, MKL_INT* info );
void mkl_lapack_avx2_sgetrf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* ipiv, MKL_INT* info );
void mkl_lapack_avx512_sgetrf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* ipiv, MKL_INT* info );

void mkl_lapack_sse2_sgetrs(const char* trans, const MKL_INT* n,
    const MKL_INT* nrhs, const float* a, const MKL_INT* lda, const MKL_INT* ipiv,
    float* b, const MKL_INT* ldb, MKL_INT* info , int itrans);
void mkl_lapack_ssse3_sgetrs(const char* trans, const MKL_INT* n,
    const MKL_INT* nrhs, const float* a, const MKL_INT* lda, const MKL_INT* ipiv,
    float* b, const MKL_INT* ldb, MKL_INT* info , int itrans);
void mkl_lapack_sse42_sgetrs(const char* trans, const MKL_INT* n,
    const MKL_INT* nrhs, const float* a, const MKL_INT* lda, const MKL_INT* ipiv,
    float* b, const MKL_INT* ldb, MKL_INT* info , int itrans);
void mkl_lapack_avx_sgetrs(const char* trans, const MKL_INT* n,
    const MKL_INT* nrhs, const float* a, const MKL_INT* lda, const MKL_INT* ipiv,
    float* b, const MKL_INT* ldb, MKL_INT* info , int itrans);
void mkl_lapack_avx2_sgetrs(const char* trans, const MKL_INT* n,
    const MKL_INT* nrhs, const float* a, const MKL_INT* lda, const MKL_INT* ipiv,
    float* b, const MKL_INT* ldb, MKL_INT* info , int itrans);
void mkl_lapack_avx512_sgetrs(const char* trans, const MKL_INT* n,
    const MKL_INT* nrhs, const float* a, const MKL_INT* lda, const MKL_INT* ipiv,
    float* b, const MKL_INT* ldb, MKL_INT* info , int itrans);

float mkl_lapack_sse2_slange(const char* norm, const MKL_INT* m,
    const MKL_INT* n, const float* a, const MKL_INT* lda, float* work ,
    int inorm);
float mkl_lapack_ssse3_slange(const char* norm, const MKL_INT* m,
    const MKL_INT* n, const float* a, const MKL_INT* lda, float* work ,
    int inorm);
float mkl_lapack_sse42_slange(const char* norm, const MKL_INT* m,
    const MKL_INT* n, const float* a, const MKL_INT* lda, float* work ,
    int inorm);
float mkl_lapack_avx_slange(const char* norm, const MKL_INT* m, const MKL_INT* n,
    const float* a, const MKL_INT* lda, float* work , int inorm);
float mkl_lapack_avx2_slange(const char* norm, const MKL_INT* m,
    const MKL_INT* n, const float* a, const MKL_INT* lda, float* work ,
    int inorm);
float mkl_lapack_avx512_slange(const char* norm, const MKL_INT* m,
    const MKL_INT* n, const float* a, const MKL_INT* lda, float* work ,
    int inorm);

void mkl_lapack_sse2_slarnv(const MKL_INT* idist, MKL_INT* iseed,
    const MKL_INT* n, float* x );
void mkl_lapack_ssse3_slarnv(const MKL_INT* idist, MKL_INT* iseed,
    const MKL_INT* n, float* x );
void mkl_lapack_sse42_slarnv(const MKL_INT* idist, MKL_INT* iseed,
    const MKL_INT* n, float* x );
void mkl_lapack_avx_slarnv(const MKL_INT* idist, MKL_INT* iseed,
    const MKL_INT* n, float* x );
void mkl_lapack_avx2_slarnv(const MKL_INT* idist, MKL_INT* iseed,
    const MKL_INT* n, float* x );
void mkl_lapack_avx512_slarnv(const MKL_INT* idist, MKL_INT* iseed,
    const MKL_INT* n, float* x );

void mkl_lapack_sse2_sorgqr(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
    float* a, const MKL_INT* lda, const float* tau, float* work,
    const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_ssse3_sorgqr(const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* k, float* a, const MKL_INT* lda, const float* tau,
    float* work, const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_sse42_sorgqr(const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* k, float* a, const MKL_INT* lda, const float* tau,
    float* work, const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_avx_sorgqr(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
    float* a, const MKL_INT* lda, const float* tau, float* work,
    const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_avx2_sorgqr(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
    float* a, const MKL_INT* lda, const float* tau, float* work,
    const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_avx512_sorgqr(const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* k, float* a, const MKL_INT* lda, const float* tau,
    float* work, const MKL_INT* lwork, MKL_INT* info );

void mkl_lapack_sse2_sorgrq(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
    float* a, const MKL_INT* lda, const float* tau, float* work,
    const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_ssse3_sorgrq(const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* k, float* a, const MKL_INT* lda, const float* tau,
    float* work, const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_sse42_sorgrq(const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* k, float* a, const MKL_INT* lda, const float* tau,
    float* work, const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_avx_sorgrq(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
    float* a, const MKL_INT* lda, const float* tau, float* work,
    const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_avx2_sorgrq(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
    float* a, const MKL_INT* lda, const float* tau, float* work,
    const MKL_INT* lwork, MKL_INT* info );
void mkl_lapack_avx512_sorgrq(const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* k, float* a, const MKL_INT* lda, const float* tau,
    float* work, const MKL_INT* lwork, MKL_INT* info );

void mkl_lapack_sse2_sormqr(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const float* a,
    const MKL_INT* lda, const float* tau, float* c, const MKL_INT* ldc,
    float* work, const MKL_INT* lwork, MKL_INT* info , int iside, int itrans);
void mkl_lapack_ssse3_sormqr(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const float* a,
    const MKL_INT* lda, const float* tau, float* c, const MKL_INT* ldc,
    float* work, const MKL_INT* lwork, MKL_INT* info , int iside, int itrans);
void mkl_lapack_sse42_sormqr(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const float* a,
    const MKL_INT* lda, const float* tau, float* c, const MKL_INT* ldc,
    float* work, const MKL_INT* lwork, MKL_INT* info , int iside, int itrans);
void mkl_lapack_avx_sormqr(const char* side, const char* trans, const MKL_INT* m,
    const MKL_INT* n, const MKL_INT* k, const float* a, const MKL_INT* lda,
    const float* tau, float* c, const MKL_INT* ldc, float* work,
    const MKL_INT* lwork, MKL_INT* info , int iside, int itrans);
void mkl_lapack_avx2_sormqr(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const float* a,
    const MKL_INT* lda, const float* tau, float* c, const MKL_INT* ldc,
    float* work, const MKL_INT* lwork, MKL_INT* info , int iside, int itrans);
void mkl_lapack_avx512_sormqr(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const float* a,
    const MKL_INT* lda, const float* tau, float* c, const MKL_INT* ldc,
    float* work, const MKL_INT* lwork, MKL_INT* info , int iside, int itrans);

void mkl_lapack_sse2_sormrq(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const float* a,
    const MKL_INT* lda, const float* tau, float* c, const MKL_INT* ldc,
    float* work, const MKL_INT* lwork, MKL_INT* info , int iside, int itrans);
void mkl_lapack_ssse3_sormrq(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const float* a,
    const MKL_INT* lda, const float* tau, float* c, const MKL_INT* ldc,
    float* work, const MKL_INT* lwork, MKL_INT* info , int iside, int itrans);
void mkl_lapack_sse42_sormrq(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const float* a,
    const MKL_INT* lda, const float* tau, float* c, const MKL_INT* ldc,
    float* work, const MKL_INT* lwork, MKL_INT* info , int iside, int itrans);
void mkl_lapack_avx_sormrq(const char* side, const char* trans, const MKL_INT* m,
    const MKL_INT* n, const MKL_INT* k, const float* a, const MKL_INT* lda,
    const float* tau, float* c, const MKL_INT* ldc, float* work,
    const MKL_INT* lwork, MKL_INT* info , int iside, int itrans);
void mkl_lapack_avx2_sormrq(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const float* a,
    const MKL_INT* lda, const float* tau, float* c, const MKL_INT* ldc,
    float* work, const MKL_INT* lwork, MKL_INT* info , int iside, int itrans);
void mkl_lapack_avx512_sormrq(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const float* a,
    const MKL_INT* lda, const float* tau, float* c, const MKL_INT* ldc,
    float* work, const MKL_INT* lwork, MKL_INT* info , int iside, int itrans);

void mkl_lapack_sse2_spftrf(const char* transr, const char* uplo,
    const MKL_INT* n, float* a, MKL_INT* info , int itransr, int iuplo);
void mkl_lapack_ssse3_spftrf(const char* transr, const char* uplo,
    const MKL_INT* n, float* a, MKL_INT* info , int itransr, int iuplo);
void mkl_lapack_sse42_spftrf(const char* transr, const char* uplo,
    const MKL_INT* n, float* a, MKL_INT* info , int itransr, int iuplo);
void mkl_lapack_avx_spftrf(const char* transr, const char* uplo,
    const MKL_INT* n, float* a, MKL_INT* info , int itransr, int iuplo);
void mkl_lapack_avx2_spftrf(const char* transr, const char* uplo,
    const MKL_INT* n, float* a, MKL_INT* info , int itransr, int iuplo);
void mkl_lapack_avx512_spftrf(const char* transr, const char* uplo,
    const MKL_INT* n, float* a, MKL_INT* info , int itransr, int iuplo);

void mkl_lapack_sse2_spotrf(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* info , int iuplo);
void mkl_lapack_ssse3_spotrf(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* info , int iuplo);
void mkl_lapack_sse42_spotrf(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* info , int iuplo);
void mkl_lapack_avx_spotrf(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* info , int iuplo);
void mkl_lapack_avx2_spotrf(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* info , int iuplo);
void mkl_lapack_avx512_spotrf(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* info , int iuplo);

void mkl_lapack_sse2_spotri(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* info , int iuplo);
void mkl_lapack_ssse3_spotri(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* info , int iuplo);
void mkl_lapack_sse42_spotri(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* info , int iuplo);
void mkl_lapack_avx_spotri(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* info , int iuplo);
void mkl_lapack_avx2_spotri(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* info , int iuplo);
void mkl_lapack_avx512_spotri(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* info , int iuplo);

void mkl_lapack_sse2_spotrs(const char* uplo, const MKL_INT* n,
    const MKL_INT* nrhs, const float* a, const MKL_INT* lda, float* b,
    const MKL_INT* ldb, MKL_INT* info , int iuplo);
void mkl_lapack_ssse3_spotrs(const char* uplo, const MKL_INT* n,
    const MKL_INT* nrhs, const float* a, const MKL_INT* lda, float* b,
    const MKL_INT* ldb, MKL_INT* info , int iuplo);
void mkl_lapack_sse42_spotrs(const char* uplo, const MKL_INT* n,
    const MKL_INT* nrhs, const float* a, const MKL_INT* lda, float* b,
    const MKL_INT* ldb, MKL_INT* info , int iuplo);
void mkl_lapack_avx_spotrs(const char* uplo, const MKL_INT* n,
    const MKL_INT* nrhs, const float* a, const MKL_INT* lda, float* b,
    const MKL_INT* ldb, MKL_INT* info , int iuplo);
void mkl_lapack_avx2_spotrs(const char* uplo, const MKL_INT* n,
    const MKL_INT* nrhs, const float* a, const MKL_INT* lda, float* b,
    const MKL_INT* ldb, MKL_INT* info , int iuplo);
void mkl_lapack_avx512_spotrs(const char* uplo, const MKL_INT* n,
    const MKL_INT* nrhs, const float* a, const MKL_INT* lda, float* b,
    const MKL_INT* ldb, MKL_INT* info , int iuplo);

void mkl_lapack_sse2_spptrf(const char* uplo, const MKL_INT* n, float* ap,
    MKL_INT* info , int iuplo);
void mkl_lapack_ssse3_spptrf(const char* uplo, const MKL_INT* n, float* ap,
    MKL_INT* info , int iuplo);
void mkl_lapack_sse42_spptrf(const char* uplo, const MKL_INT* n, float* ap,
    MKL_INT* info , int iuplo);
void mkl_lapack_avx_spptrf(const char* uplo, const MKL_INT* n, float* ap,
    MKL_INT* info , int iuplo);
void mkl_lapack_avx2_spptrf(const char* uplo, const MKL_INT* n, float* ap,
    MKL_INT* info , int iuplo);
void mkl_lapack_avx512_spptrf(const char* uplo, const MKL_INT* n, float* ap,
    MKL_INT* info , int iuplo);

void mkl_lapack_sse2_spstrf(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* piv, MKL_INT* rank, const float* tol,
    float* work, MKL_INT* info , int iuplo);
void mkl_lapack_ssse3_spstrf(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* piv, MKL_INT* rank, const float* tol,
    float* work, MKL_INT* info , int iuplo);
void mkl_lapack_sse42_spstrf(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* piv, MKL_INT* rank, const float* tol,
    float* work, MKL_INT* info , int iuplo);
void mkl_lapack_avx_spstrf(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* piv, MKL_INT* rank, const float* tol,
    float* work, MKL_INT* info , int iuplo);
void mkl_lapack_avx2_spstrf(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* piv, MKL_INT* rank, const float* tol,
    float* work, MKL_INT* info , int iuplo);
void mkl_lapack_avx512_spstrf(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* piv, MKL_INT* rank, const float* tol,
    float* work, MKL_INT* info , int iuplo);

void mkl_lapack_sse2_sspevd(const char* jobz, const char* uplo, const MKL_INT* n,
    float* ap, float* w, float* z, const MKL_INT* ldz, float* work,
    const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info ,
    int ijobz, int iuplo);
void mkl_lapack_ssse3_sspevd(const char* jobz, const char* uplo,
    const MKL_INT* n, float* ap, float* w, float* z, const MKL_INT* ldz,
    float* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork,
    MKL_INT* info , int ijobz, int iuplo);
void mkl_lapack_sse42_sspevd(const char* jobz, const char* uplo,
    const MKL_INT* n, float* ap, float* w, float* z, const MKL_INT* ldz,
    float* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork,
    MKL_INT* info , int ijobz, int iuplo);
void mkl_lapack_avx_sspevd(const char* jobz, const char* uplo, const MKL_INT* n,
    float* ap, float* w, float* z, const MKL_INT* ldz, float* work,
    const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info ,
    int ijobz, int iuplo);
void mkl_lapack_avx2_sspevd(const char* jobz, const char* uplo, const MKL_INT* n,
    float* ap, float* w, float* z, const MKL_INT* ldz, float* work,
    const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info ,
    int ijobz, int iuplo);
void mkl_lapack_avx512_sspevd(const char* jobz, const char* uplo,
    const MKL_INT* n, float* ap, float* w, float* z, const MKL_INT* ldz,
    float* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork,
    MKL_INT* info , int ijobz, int iuplo);

void mkl_lapack_sse2_ssyev(const char* jobz, const char* uplo, const MKL_INT* n,
    float* a, const MKL_INT* lda, float* w, float* work, const MKL_INT* lwork,
    MKL_INT* info , int ijobz, int iuplo);
void mkl_lapack_ssse3_ssyev(const char* jobz, const char* uplo, const MKL_INT* n,
    float* a, const MKL_INT* lda, float* w, float* work, const MKL_INT* lwork,
    MKL_INT* info , int ijobz, int iuplo);
void mkl_lapack_sse42_ssyev(const char* jobz, const char* uplo, const MKL_INT* n,
    float* a, const MKL_INT* lda, float* w, float* work, const MKL_INT* lwork,
    MKL_INT* info , int ijobz, int iuplo);
void mkl_lapack_avx_ssyev(const char* jobz, const char* uplo, const MKL_INT* n,
    float* a, const MKL_INT* lda, float* w, float* work, const MKL_INT* lwork,
    MKL_INT* info , int ijobz, int iuplo);
void mkl_lapack_avx2_ssyev(const char* jobz, const char* uplo, const MKL_INT* n,
    float* a, const MKL_INT* lda, float* w, float* work, const MKL_INT* lwork,
    MKL_INT* info , int ijobz, int iuplo);
void mkl_lapack_avx512_ssyev(const char* jobz, const char* uplo,
    const MKL_INT* n, float* a, const MKL_INT* lda, float* w, float* work,
    const MKL_INT* lwork, MKL_INT* info , int ijobz, int iuplo);

void mkl_lapack_sse2_ssyevd(const char* jobz, const char* uplo, const MKL_INT* n,
    float* a, const MKL_INT* lda, float* w, float* work, const MKL_INT* lwork,
    MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info , int ijobz, int iuplo);
void mkl_lapack_ssse3_ssyevd(const char* jobz, const char* uplo,
    const MKL_INT* n, float* a, const MKL_INT* lda, float* w, float* work,
    const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info ,
    int ijobz, int iuplo);
void mkl_lapack_sse42_ssyevd(const char* jobz, const char* uplo,
    const MKL_INT* n, float* a, const MKL_INT* lda, float* w, float* work,
    const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info ,
    int ijobz, int iuplo);
void mkl_lapack_avx_ssyevd(const char* jobz, const char* uplo, const MKL_INT* n,
    float* a, const MKL_INT* lda, float* w, float* work, const MKL_INT* lwork,
    MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info , int ijobz, int iuplo);
void mkl_lapack_avx2_ssyevd(const char* jobz, const char* uplo, const MKL_INT* n,
    float* a, const MKL_INT* lda, float* w, float* work, const MKL_INT* lwork,
    MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info , int ijobz, int iuplo);
void mkl_lapack_avx512_ssyevd(const char* jobz, const char* uplo,
    const MKL_INT* n, float* a, const MKL_INT* lda, float* w, float* work,
    const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info ,
    int ijobz, int iuplo);

void mkl_lapack_sse2_ssyevr(const char* jobz, const char* range,
    const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda,
    const float* vl, const float* vu, const MKL_INT* il, const MKL_INT* iu,
    const float* abstol, MKL_INT* m, float* w, float* z, const MKL_INT* ldz,
    MKL_INT* isuppz, float* work, const MKL_INT* lwork, MKL_INT* iwork,
    const MKL_INT* liwork, MKL_INT* info , int ijobz, int irange, int iuplo);
void mkl_lapack_ssse3_ssyevr(const char* jobz, const char* range,
    const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda,
    const float* vl, const float* vu, const MKL_INT* il, const MKL_INT* iu,
    const float* abstol, MKL_INT* m, float* w, float* z, const MKL_INT* ldz,
    MKL_INT* isuppz, float* work, const MKL_INT* lwork, MKL_INT* iwork,
    const MKL_INT* liwork, MKL_INT* info , int ijobz, int irange, int iuplo);
void mkl_lapack_sse42_ssyevr(const char* jobz, const char* range,
    const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda,
    const float* vl, const float* vu, const MKL_INT* il, const MKL_INT* iu,
    const float* abstol, MKL_INT* m, float* w, float* z, const MKL_INT* ldz,
    MKL_INT* isuppz, float* work, const MKL_INT* lwork, MKL_INT* iwork,
    const MKL_INT* liwork, MKL_INT* info , int ijobz, int irange, int iuplo);
void mkl_lapack_avx_ssyevr(const char* jobz, const char* range, const char* uplo,
    const MKL_INT* n, float* a, const MKL_INT* lda, const float* vl,
    const float* vu, const MKL_INT* il, const MKL_INT* iu, const float* abstol,
    MKL_INT* m, float* w, float* z, const MKL_INT* ldz, MKL_INT* isuppz,
    float* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork,
    MKL_INT* info , int ijobz, int irange, int iuplo);
void mkl_lapack_avx2_ssyevr(const char* jobz, const char* range,
    const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda,
    const float* vl, const float* vu, const MKL_INT* il, const MKL_INT* iu,
    const float* abstol, MKL_INT* m, float* w, float* z, const MKL_INT* ldz,
    MKL_INT* isuppz, float* work, const MKL_INT* lwork, MKL_INT* iwork,
    const MKL_INT* liwork, MKL_INT* info , int ijobz, int irange, int iuplo);
void mkl_lapack_avx512_ssyevr(const char* jobz, const char* range,
    const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda,
    const float* vl, const float* vu, const MKL_INT* il, const MKL_INT* iu,
    const float* abstol, MKL_INT* m, float* w, float* z, const MKL_INT* ldz,
    MKL_INT* isuppz, float* work, const MKL_INT* lwork, MKL_INT* iwork,
    const MKL_INT* liwork, MKL_INT* info , int ijobz, int irange, int iuplo);

void mkl_lapack_sse2_strtrs(const char* uplo, const char* trans,
    const char* diag, const MKL_INT* n, const MKL_INT* nrhs, const float* a,
    const MKL_INT* lda, float* b, const MKL_INT* ldb, MKL_INT* info , int iuplo,
    int itrans, int idiag);
void mkl_lapack_ssse3_strtrs(const char* uplo, const char* trans,
    const char* diag, const MKL_INT* n, const MKL_INT* nrhs, const float* a,
    const MKL_INT* lda, float* b, const MKL_INT* ldb, MKL_INT* info , int iuplo,
    int itrans, int idiag);
void mkl_lapack_sse42_strtrs(const char* uplo, const char* trans,
    const char* diag, const MKL_INT* n, const MKL_INT* nrhs, const float* a,
    const MKL_INT* lda, float* b, const MKL_INT* ldb, MKL_INT* info , int iuplo,
    int itrans, int idiag);
void mkl_lapack_avx_strtrs(const char* uplo, const char* trans, const char* diag,
    const MKL_INT* n, const MKL_INT* nrhs, const float* a, const MKL_INT* lda,
    float* b, const MKL_INT* ldb, MKL_INT* info , int iuplo, int itrans,
    int idiag);
void mkl_lapack_avx2_strtrs(const char* uplo, const char* trans,
    const char* diag, const MKL_INT* n, const MKL_INT* nrhs, const float* a,
    const MKL_INT* lda, float* b, const MKL_INT* ldb, MKL_INT* info , int iuplo,
    int itrans, int idiag);
void mkl_lapack_avx512_strtrs(const char* uplo, const char* trans,
    const char* diag, const MKL_INT* n, const MKL_INT* nrhs, const float* a,
    const MKL_INT* lda, float* b, const MKL_INT* ldb, MKL_INT* info , int iuplo,
    int itrans, int idiag);




void* mkl_serv_allocate(size_t size, int alignment);

int mkl_serv_cpuisclx(void);

int mkl_serv_cpuiscpx(void);

int mkl_serv_cpuisicx(void);

int mkl_serv_cpuisknm(void);

void mkl_serv_deallocate(void *ptr);


int mkl_serv_enable_instructions(int);

void mkl_serv_free(void *ptr);

void mkl_serv_free_buffers(void);


int mkl_serv_get_ht(void);


int mkl_serv_get_max_threads(void);


int mkl_serv_get_ncorespercpu(void);

int mkl_serv_get_ncpus(void);

int mkl_serv_get_nlogicalcores(void);


void* mkl_serv_malloc(size_t size, int align);


int mkl_serv_memcpy_s(void *dest, size_t dmax, const void *src, size_t slen);

int mkl_serv_memmove_s(void *dest, size_t dmax, const void *src, size_t slen);


int mkl_serv_register_jit_function(void *addr, size_t size, const char *name);


int mkl_serv_set_memory_limit(int mem_type, size_t limit);

void mkl_serv_set_num_threads(int nth);

int mkl_serv_set_num_threads_local(int nth);


int mkl_serv_strncat_s(char *dest, size_t dmax, const char *src, size_t slen);

int mkl_serv_strncpy_s(char *dest, size_t dmax, const char *src, size_t slen);

size_t mkl_serv_strnlen_s(const char *s, size_t smax);

void mkl_trans_sse2_mkl_domatcopy(char ordering, char trans, size_t rows,
    size_t cols, const double alpha, const double * A, size_t lda, double * B,
    size_t ldb);
void mkl_trans_ssse3_mkl_domatcopy(char ordering, char trans, size_t rows,
    size_t cols, const double alpha, const double * A, size_t lda, double * B,
    size_t ldb);
void mkl_trans_sse42_mkl_domatcopy(char ordering, char trans, size_t rows,
    size_t cols, const double alpha, const double * A, size_t lda, double * B,
    size_t ldb);
void mkl_trans_avx_mkl_domatcopy(char ordering, char trans, size_t rows,
    size_t cols, const double alpha, const double * A, size_t lda, double * B,
    size_t ldb);
void mkl_trans_avx2_mkl_domatcopy(char ordering, char trans, size_t rows,
    size_t cols, const double alpha, const double * A, size_t lda, double * B,
    size_t ldb);
void mkl_trans_avx512_mkl_domatcopy(char ordering, char trans, size_t rows,
    size_t cols, const double alpha, const double * A, size_t lda, double * B,
    size_t ldb);

void mkl_trans_sse2_mkl_somatcopy(char ordering, char trans, size_t rows,
    size_t cols, const float alpha, const float * A, size_t lda, float * B,
    size_t ldb);
void mkl_trans_ssse3_mkl_somatcopy(char ordering, char trans, size_t rows,
    size_t cols, const float alpha, const float * A, size_t lda, float * B,
    size_t ldb);
void mkl_trans_sse42_mkl_somatcopy(char ordering, char trans, size_t rows,
    size_t cols, const float alpha, const float * A, size_t lda, float * B,
    size_t ldb);
void mkl_trans_avx_mkl_somatcopy(char ordering, char trans, size_t rows,
    size_t cols, const float alpha, const float * A, size_t lda, float * B,
    size_t ldb);
void mkl_trans_avx2_mkl_somatcopy(char ordering, char trans, size_t rows,
    size_t cols, const float alpha, const float * A, size_t lda, float * B,
    size_t ldb);
void mkl_trans_avx512_mkl_somatcopy(char ordering, char trans, size_t rows,
    size_t cols, const float alpha, const float * A, size_t lda, float * B,
    size_t ldb);


#if defined(__cplusplus)
}
#endif

#endif /*MKL_DAL_H*/