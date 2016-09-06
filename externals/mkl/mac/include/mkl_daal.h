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

#ifndef MKL_DAL_H
#define MKL_DAL_H

#include <stddef.h>
#include "mkl_dnn_types.h"

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

#ifndef MKL_Complex8
typedef
struct _MKL_Complex8 {
    float real;
    float imag;
} MKL_Complex8;
#endif

#ifndef MKL_Complex16
typedef
struct _MKL_Complex16 {
    double real;
    double imag;
} MKL_Complex16;
#endif

typedef void *             _MKL_DSS_HANDLE_t;

enum PARDISO_ENV_PARAM {
       PARDISO_OOC_FILE_NAME = 1
};

typedef int             IppStatus;
typedef unsigned char   Ipp8u;
typedef unsigned short  Ipp16u;
typedef unsigned int    Ipp32u;
typedef signed short    Ipp16s;
typedef signed int      Ipp32s;
typedef float           Ipp32f;
typedef double          Ipp64f;

void fpk_blas_ssse3_daxpy(const MKL_INT *n, const double *alpha, const double *x,
    const MKL_INT *incx, double *y, const MKL_INT *incy);
void fpk_blas_ssse3_dcopy(const MKL_INT *n, const double *x, const MKL_INT *incx,
    double *y, const MKL_INT *incy);
double fpk_blas_ssse3_ddot(const MKL_INT *n, const double *x,
    const MKL_INT *incx, const double *y, const MKL_INT *incy);
void fpk_blas_ssse3_dgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_ssse3_dgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_ssse3_dgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, const double *x,
    const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void fpk_blas_ssse3_dsymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
    const double *b, const MKL_INT *ldb, const double *beta, double *c,
    const MKL_INT *ldc);
void fpk_blas_ssse3_dsyr(const char *uplo, const MKL_INT *n, const double *alpha,
    const double *x, const MKL_INT *incx, double *a, const MKL_INT *lda);
void fpk_blas_ssse3_dsyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *lda,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_ssse3_dtrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const double *alpha,
    const double *a, const MKL_INT *lda, double *b, const MKL_INT *ldb);
void fpk_blas_ssse3_saxpy(const MKL_INT *n, const float *alpha, const float *x,
    const MKL_INT *incx, float *y, const MKL_INT *incy);
void fpk_blas_ssse3_scopy(const MKL_INT *n, const float *x, const MKL_INT *incx,
    float *y, const MKL_INT *incy);
float fpk_blas_ssse3_sdot(const MKL_INT *n, const float *x, const MKL_INT *incx,
    const float *y, const MKL_INT *incy);
void fpk_blas_ssse3_sgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_ssse3_sgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_ssse3_sgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, const float *x,
    const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void fpk_blas_ssse3_ssymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
    const float *b, const MKL_INT *ldb, const float *beta, float *c,
    const MKL_INT *ldc);
void fpk_blas_ssse3_ssyr(const char *uplo, const MKL_INT *n, const float *alpha,
    const float *x, const MKL_INT *incx, float *a, const MKL_INT *lda);
void fpk_blas_ssse3_ssyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *lda,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_ssse3_strmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const float *alpha,
    const float *a, const MKL_INT *lda, float *b, const MKL_INT *ldb);
void fpk_blas_ssse3_xdaxpy(const MKL_INT *n, const double *alpha,
    const double *x, const MKL_INT *incx, double *y, const MKL_INT *incy);
void fpk_blas_ssse3_xdcopy(const MKL_INT *n, const double *x,
    const MKL_INT *incx, double *y, const MKL_INT *incy);
double fpk_blas_ssse3_xddot(const MKL_INT *n, const double *x,
    const MKL_INT *incx, const double *y, const MKL_INT *incy);
void fpk_blas_ssse3_xdgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_ssse3_xdgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_ssse3_xdgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, const double *x,
    const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void fpk_blas_ssse3_xdsymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
    const double *b, const MKL_INT *ldb, const double *beta, double *c,
    const MKL_INT *ldc);
void fpk_blas_ssse3_xdsyr(const char *uplo, const MKL_INT *n,
    const double *alpha, const double *x, const MKL_INT *incx, double *a,
    const MKL_INT *lda);
void fpk_blas_ssse3_xdsyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *lda,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_ssse3_xdtrmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, double *b,
    const MKL_INT *ldb);
void fpk_blas_ssse3_xsaxpy(const MKL_INT *n, const float *alpha, const float *x,
    const MKL_INT *incx, float *y, const MKL_INT *incy);
void fpk_blas_ssse3_xscopy(const MKL_INT *n, const float *x, const MKL_INT *incx,
    float *y, const MKL_INT *incy);
float fpk_blas_ssse3_xsdot(const MKL_INT *n, const float *x, const MKL_INT *incx,
    const float *y, const MKL_INT *incy);
void fpk_blas_ssse3_xsgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_ssse3_xsgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_ssse3_xsgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, const float *x,
    const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void fpk_blas_ssse3_xssymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
    const float *b, const MKL_INT *ldb, const float *beta, float *c,
    const MKL_INT *ldc);
void fpk_blas_ssse3_xssyr(const char *uplo, const MKL_INT *n, const float *alpha,
    const float *x, const MKL_INT *incx, float *a, const MKL_INT *lda);
void fpk_blas_ssse3_xssyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *lda,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_ssse3_xstrmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, float *b,
    const MKL_INT *ldb);

void fpk_blas_sse42_daxpy(const MKL_INT *n, const double *alpha, const double *x,
    const MKL_INT *incx, double *y, const MKL_INT *incy);
void fpk_blas_sse42_dcopy(const MKL_INT *n, const double *x, const MKL_INT *incx,
    double *y, const MKL_INT *incy);
double fpk_blas_sse42_ddot(const MKL_INT *n, const double *x,
    const MKL_INT *incx, const double *y, const MKL_INT *incy);
void fpk_blas_sse42_dgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_sse42_dgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_sse42_dgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, const double *x,
    const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void fpk_blas_sse42_dsymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
    const double *b, const MKL_INT *ldb, const double *beta, double *c,
    const MKL_INT *ldc);
void fpk_blas_sse42_dsyr(const char *uplo, const MKL_INT *n, const double *alpha,
    const double *x, const MKL_INT *incx, double *a, const MKL_INT *lda);
void fpk_blas_sse42_dsyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *lda,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_sse42_dtrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const double *alpha,
    const double *a, const MKL_INT *lda, double *b, const MKL_INT *ldb);
void fpk_blas_sse42_saxpy(const MKL_INT *n, const float *alpha, const float *x,
    const MKL_INT *incx, float *y, const MKL_INT *incy);
void fpk_blas_sse42_scopy(const MKL_INT *n, const float *x, const MKL_INT *incx,
    float *y, const MKL_INT *incy);
float fpk_blas_sse42_sdot(const MKL_INT *n, const float *x, const MKL_INT *incx,
    const float *y, const MKL_INT *incy);
void fpk_blas_sse42_sgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_sse42_sgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_sse42_sgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, const float *x,
    const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void fpk_blas_sse42_ssymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
    const float *b, const MKL_INT *ldb, const float *beta, float *c,
    const MKL_INT *ldc);
void fpk_blas_sse42_ssyr(const char *uplo, const MKL_INT *n, const float *alpha,
    const float *x, const MKL_INT *incx, float *a, const MKL_INT *lda);
void fpk_blas_sse42_ssyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *lda,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_sse42_strmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const float *alpha,
    const float *a, const MKL_INT *lda, float *b, const MKL_INT *ldb);
void fpk_blas_sse42_xdaxpy(const MKL_INT *n, const double *alpha,
    const double *x, const MKL_INT *incx, double *y, const MKL_INT *incy);
void fpk_blas_sse42_xdcopy(const MKL_INT *n, const double *x,
    const MKL_INT *incx, double *y, const MKL_INT *incy);
double fpk_blas_sse42_xddot(const MKL_INT *n, const double *x,
    const MKL_INT *incx, const double *y, const MKL_INT *incy);
void fpk_blas_sse42_xdgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_sse42_xdgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_sse42_xdgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, const double *x,
    const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void fpk_blas_sse42_xdsymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
    const double *b, const MKL_INT *ldb, const double *beta, double *c,
    const MKL_INT *ldc);
void fpk_blas_sse42_xdsyr(const char *uplo, const MKL_INT *n,
    const double *alpha, const double *x, const MKL_INT *incx, double *a,
    const MKL_INT *lda);
void fpk_blas_sse42_xdsyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *lda,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_sse42_xdtrmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, double *b,
    const MKL_INT *ldb);
void fpk_blas_sse42_xsaxpy(const MKL_INT *n, const float *alpha, const float *x,
    const MKL_INT *incx, float *y, const MKL_INT *incy);
void fpk_blas_sse42_xscopy(const MKL_INT *n, const float *x, const MKL_INT *incx,
    float *y, const MKL_INT *incy);
float fpk_blas_sse42_xsdot(const MKL_INT *n, const float *x, const MKL_INT *incx,
    const float *y, const MKL_INT *incy);
void fpk_blas_sse42_xsgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_sse42_xsgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_sse42_xsgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, const float *x,
    const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void fpk_blas_sse42_xssymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
    const float *b, const MKL_INT *ldb, const float *beta, float *c,
    const MKL_INT *ldc);
void fpk_blas_sse42_xssyr(const char *uplo, const MKL_INT *n, const float *alpha,
    const float *x, const MKL_INT *incx, float *a, const MKL_INT *lda);
void fpk_blas_sse42_xssyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *lda,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_sse42_xstrmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, float *b,
    const MKL_INT *ldb);

void fpk_blas_avx_daxpy(const MKL_INT *n, const double *alpha, const double *x,
    const MKL_INT *incx, double *y, const MKL_INT *incy);
void fpk_blas_avx_dcopy(const MKL_INT *n, const double *x, const MKL_INT *incx,
    double *y, const MKL_INT *incy);
double fpk_blas_avx_ddot(const MKL_INT *n, const double *x, const MKL_INT *incx,
    const double *y, const MKL_INT *incy);
void fpk_blas_avx_dgemm(const char *transa, const char *transb, const MKL_INT *m,
    const MKL_INT *n, const MKL_INT *k, const double *alpha, const double *a,
    const MKL_INT *lda, const double *b, const MKL_INT *ldb, const double *beta,
    double *c, const MKL_INT *ldc);
void fpk_blas_avx_dgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_avx_dgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, const double *x,
    const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void fpk_blas_avx_dsymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
    const double *b, const MKL_INT *ldb, const double *beta, double *c,
    const MKL_INT *ldc);
void fpk_blas_avx_dsyr(const char *uplo, const MKL_INT *n, const double *alpha,
    const double *x, const MKL_INT *incx, double *a, const MKL_INT *lda);
void fpk_blas_avx_dsyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *lda,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_avx_dtrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const double *alpha,
    const double *a, const MKL_INT *lda, double *b, const MKL_INT *ldb);
void fpk_blas_avx_saxpy(const MKL_INT *n, const float *alpha, const float *x,
    const MKL_INT *incx, float *y, const MKL_INT *incy);
void fpk_blas_avx_scopy(const MKL_INT *n, const float *x, const MKL_INT *incx,
    float *y, const MKL_INT *incy);
float fpk_blas_avx_sdot(const MKL_INT *n, const float *x, const MKL_INT *incx,
    const float *y, const MKL_INT *incy);
void fpk_blas_avx_sgemm(const char *transa, const char *transb, const MKL_INT *m,
    const MKL_INT *n, const MKL_INT *k, const float *alpha, const float *a,
    const MKL_INT *lda, const float *b, const MKL_INT *ldb, const float *beta,
    float *c, const MKL_INT *ldc);
void fpk_blas_avx_sgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_avx_sgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, const float *x,
    const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void fpk_blas_avx_ssymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
    const float *b, const MKL_INT *ldb, const float *beta, float *c,
    const MKL_INT *ldc);
void fpk_blas_avx_ssyr(const char *uplo, const MKL_INT *n, const float *alpha,
    const float *x, const MKL_INT *incx, float *a, const MKL_INT *lda);
void fpk_blas_avx_ssyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *lda,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_avx_strmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const float *alpha,
    const float *a, const MKL_INT *lda, float *b, const MKL_INT *ldb);
void fpk_blas_avx_xdaxpy(const MKL_INT *n, const double *alpha, const double *x,
    const MKL_INT *incx, double *y, const MKL_INT *incy);
void fpk_blas_avx_xdcopy(const MKL_INT *n, const double *x, const MKL_INT *incx,
    double *y, const MKL_INT *incy);
double fpk_blas_avx_xddot(const MKL_INT *n, const double *x, const MKL_INT *incx,
    const double *y, const MKL_INT *incy);
void fpk_blas_avx_xdgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_avx_xdgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_avx_xdgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, const double *x,
    const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void fpk_blas_avx_xdsymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
    const double *b, const MKL_INT *ldb, const double *beta, double *c,
    const MKL_INT *ldc);
void fpk_blas_avx_xdsyr(const char *uplo, const MKL_INT *n, const double *alpha,
    const double *x, const MKL_INT *incx, double *a, const MKL_INT *lda);
void fpk_blas_avx_xdsyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *lda,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_avx_xdtrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const double *alpha,
    const double *a, const MKL_INT *lda, double *b, const MKL_INT *ldb);
void fpk_blas_avx_xsaxpy(const MKL_INT *n, const float *alpha, const float *x,
    const MKL_INT *incx, float *y, const MKL_INT *incy);
void fpk_blas_avx_xscopy(const MKL_INT *n, const float *x, const MKL_INT *incx,
    float *y, const MKL_INT *incy);
float fpk_blas_avx_xsdot(const MKL_INT *n, const float *x, const MKL_INT *incx,
    const float *y, const MKL_INT *incy);
void fpk_blas_avx_xsgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_avx_xsgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_avx_xsgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, const float *x,
    const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void fpk_blas_avx_xssymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
    const float *b, const MKL_INT *ldb, const float *beta, float *c,
    const MKL_INT *ldc);
void fpk_blas_avx_xssyr(const char *uplo, const MKL_INT *n, const float *alpha,
    const float *x, const MKL_INT *incx, float *a, const MKL_INT *lda);
void fpk_blas_avx_xssyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *lda,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_avx_xstrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const float *alpha,
    const float *a, const MKL_INT *lda, float *b, const MKL_INT *ldb);

void fpk_blas_avx2_daxpy(const MKL_INT *n, const double *alpha, const double *x,
    const MKL_INT *incx, double *y, const MKL_INT *incy);
void fpk_blas_avx2_dcopy(const MKL_INT *n, const double *x, const MKL_INT *incx,
    double *y, const MKL_INT *incy);
double fpk_blas_avx2_ddot(const MKL_INT *n, const double *x, const MKL_INT *incx,
    const double *y, const MKL_INT *incy);
void fpk_blas_avx2_dgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_avx2_dgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_avx2_dgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, const double *x,
    const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void fpk_blas_avx2_dsymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
    const double *b, const MKL_INT *ldb, const double *beta, double *c,
    const MKL_INT *ldc);
void fpk_blas_avx2_dsyr(const char *uplo, const MKL_INT *n, const double *alpha,
    const double *x, const MKL_INT *incx, double *a, const MKL_INT *lda);
void fpk_blas_avx2_dsyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *lda,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_avx2_dtrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const double *alpha,
    const double *a, const MKL_INT *lda, double *b, const MKL_INT *ldb);
void fpk_blas_avx2_saxpy(const MKL_INT *n, const float *alpha, const float *x,
    const MKL_INT *incx, float *y, const MKL_INT *incy);
void fpk_blas_avx2_scopy(const MKL_INT *n, const float *x, const MKL_INT *incx,
    float *y, const MKL_INT *incy);
float fpk_blas_avx2_sdot(const MKL_INT *n, const float *x, const MKL_INT *incx,
    const float *y, const MKL_INT *incy);
void fpk_blas_avx2_sgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_avx2_sgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_avx2_sgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, const float *x,
    const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void fpk_blas_avx2_ssymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
    const float *b, const MKL_INT *ldb, const float *beta, float *c,
    const MKL_INT *ldc);
void fpk_blas_avx2_ssyr(const char *uplo, const MKL_INT *n, const float *alpha,
    const float *x, const MKL_INT *incx, float *a, const MKL_INT *lda);
void fpk_blas_avx2_ssyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *lda,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_avx2_strmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const float *alpha,
    const float *a, const MKL_INT *lda, float *b, const MKL_INT *ldb);
void fpk_blas_avx2_xdaxpy(const MKL_INT *n, const double *alpha, const double *x,
    const MKL_INT *incx, double *y, const MKL_INT *incy);
void fpk_blas_avx2_xdcopy(const MKL_INT *n, const double *x, const MKL_INT *incx,
    double *y, const MKL_INT *incy);
double fpk_blas_avx2_xddot(const MKL_INT *n, const double *x,
    const MKL_INT *incx, const double *y, const MKL_INT *incy);
void fpk_blas_avx2_xdgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_avx2_xdgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_avx2_xdgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, const double *x,
    const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void fpk_blas_avx2_xdsymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
    const double *b, const MKL_INT *ldb, const double *beta, double *c,
    const MKL_INT *ldc);
void fpk_blas_avx2_xdsyr(const char *uplo, const MKL_INT *n, const double *alpha,
    const double *x, const MKL_INT *incx, double *a, const MKL_INT *lda);
void fpk_blas_avx2_xdsyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *lda,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_avx2_xdtrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const double *alpha,
    const double *a, const MKL_INT *lda, double *b, const MKL_INT *ldb);
void fpk_blas_avx2_xsaxpy(const MKL_INT *n, const float *alpha, const float *x,
    const MKL_INT *incx, float *y, const MKL_INT *incy);
void fpk_blas_avx2_xscopy(const MKL_INT *n, const float *x, const MKL_INT *incx,
    float *y, const MKL_INT *incy);
float fpk_blas_avx2_xsdot(const MKL_INT *n, const float *x, const MKL_INT *incx,
    const float *y, const MKL_INT *incy);
void fpk_blas_avx2_xsgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_avx2_xsgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_avx2_xsgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, const float *x,
    const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void fpk_blas_avx2_xssymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
    const float *b, const MKL_INT *ldb, const float *beta, float *c,
    const MKL_INT *ldc);
void fpk_blas_avx2_xssyr(const char *uplo, const MKL_INT *n, const float *alpha,
    const float *x, const MKL_INT *incx, float *a, const MKL_INT *lda);
void fpk_blas_avx2_xssyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *lda,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_avx2_xstrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const MKL_INT *m, const MKL_INT *n, const float *alpha,
    const float *a, const MKL_INT *lda, float *b, const MKL_INT *ldb);

void fpk_blas_avx512_daxpy(const MKL_INT *n, const double *alpha,
    const double *x, const MKL_INT *incx, double *y, const MKL_INT *incy);
void fpk_blas_avx512_dcopy(const MKL_INT *n, const double *x,
    const MKL_INT *incx, double *y, const MKL_INT *incy);
double fpk_blas_avx512_ddot(const MKL_INT *n, const double *x,
    const MKL_INT *incx, const double *y, const MKL_INT *incy);
void fpk_blas_avx512_dgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_avx512_dgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_avx512_dgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, const double *x,
    const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void fpk_blas_avx512_dsymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
    const double *b, const MKL_INT *ldb, const double *beta, double *c,
    const MKL_INT *ldc);
void fpk_blas_avx512_dsyr(const char *uplo, const MKL_INT *n,
    const double *alpha, const double *x, const MKL_INT *incx, double *a,
    const MKL_INT *lda);
void fpk_blas_avx512_dsyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *lda,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_avx512_dtrmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, double *b,
    const MKL_INT *ldb);
void fpk_blas_avx512_saxpy(const MKL_INT *n, const float *alpha, const float *x,
    const MKL_INT *incx, float *y, const MKL_INT *incy);
void fpk_blas_avx512_scopy(const MKL_INT *n, const float *x, const MKL_INT *incx,
    float *y, const MKL_INT *incy);
float fpk_blas_avx512_sdot(const MKL_INT *n, const float *x, const MKL_INT *incx,
    const float *y, const MKL_INT *incy);
void fpk_blas_avx512_sgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_avx512_sgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_avx512_sgemv(const char *trans, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, const float *x,
    const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void fpk_blas_avx512_ssymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
    const float *b, const MKL_INT *ldb, const float *beta, float *c,
    const MKL_INT *ldc);
void fpk_blas_avx512_ssyr(const char *uplo, const MKL_INT *n, const float *alpha,
    const float *x, const MKL_INT *incx, float *a, const MKL_INT *lda);
void fpk_blas_avx512_ssyrk(const char *uplo, const char *trans, const MKL_INT *n,
    const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *lda,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_avx512_strmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, float *b,
    const MKL_INT *ldb);
void fpk_blas_avx512_xdaxpy(const MKL_INT *n, const double *alpha,
    const double *x, const MKL_INT *incx, double *y, const MKL_INT *incy);
void fpk_blas_avx512_xdcopy(const MKL_INT *n, const double *x,
    const MKL_INT *incx, double *y, const MKL_INT *incy);
double fpk_blas_avx512_xddot(const MKL_INT *n, const double *x,
    const MKL_INT *incx, const double *y, const MKL_INT *incy);
void fpk_blas_avx512_xdgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_avx512_xdgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
    const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_avx512_xdgemv(const char *trans, const MKL_INT *m,
    const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
    const double *x, const MKL_INT *incx, const double *beta, double *y,
    const MKL_INT *incy);
void fpk_blas_avx512_xdsymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
    const double *b, const MKL_INT *ldb, const double *beta, double *c,
    const MKL_INT *ldc);
void fpk_blas_avx512_xdsyr(const char *uplo, const MKL_INT *n,
    const double *alpha, const double *x, const MKL_INT *incx, double *a,
    const MKL_INT *lda);
void fpk_blas_avx512_xdsyrk(const char *uplo, const char *trans,
    const MKL_INT *n, const MKL_INT *k, const double *alpha, const double *a,
    const MKL_INT *lda, const double *beta, double *c, const MKL_INT *ldc);
void fpk_blas_avx512_xdtrmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, double *b,
    const MKL_INT *ldb);
void fpk_blas_avx512_xsaxpy(const MKL_INT *n, const float *alpha, const float *x,
    const MKL_INT *incx, float *y, const MKL_INT *incy);
void fpk_blas_avx512_xscopy(const MKL_INT *n, const float *x,
    const MKL_INT *incx, float *y, const MKL_INT *incy);
float fpk_blas_avx512_xsdot(const MKL_INT *n, const float *x,
    const MKL_INT *incx, const float *y, const MKL_INT *incy);
void fpk_blas_avx512_xsgemm(const char *transa, const char *transb,
    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_avx512_xsgemmt(const char *uplo, const char *transa,
    const char *transb, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
    const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_avx512_xsgemv(const char *trans, const MKL_INT *m,
    const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
    const float *x, const MKL_INT *incx, const float *beta, float *y,
    const MKL_INT *incy);
void fpk_blas_avx512_xssymm(const char *side, const char *uplo, const MKL_INT *m,
    const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
    const float *b, const MKL_INT *ldb, const float *beta, float *c,
    const MKL_INT *ldc);
void fpk_blas_avx512_xssyr(const char *uplo, const MKL_INT *n,
    const float *alpha, const float *x, const MKL_INT *incx, float *a,
    const MKL_INT *lda);
void fpk_blas_avx512_xssyrk(const char *uplo, const char *trans,
    const MKL_INT *n, const MKL_INT *k, const float *alpha, const float *a,
    const MKL_INT *lda, const float *beta, float *c, const MKL_INT *ldc);
void fpk_blas_avx512_xstrmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, float *b,
    const MKL_INT *ldb);


IppStatus fpk_dft_ssse3_ippsSortRadixAscend_16s_I(Ipp16s *pSrcDst, Ipp16s *pTmp,
    Ipp32s len);
IppStatus fpk_dft_ssse3_ippsSortRadixAscend_16u_I(Ipp16u *pSrcDst, Ipp16u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_ssse3_ippsSortRadixAscend_32f_I(Ipp32f *pSrcDst, Ipp32f *pTmp,
    Ipp32s len);
IppStatus fpk_dft_ssse3_ippsSortRadixAscend_32s_I(Ipp32s *pSrcDst, Ipp32s *pTmp,
    Ipp32s len);
IppStatus fpk_dft_ssse3_ippsSortRadixAscend_32u_I(Ipp32u *pSrcDst, Ipp32u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_ssse3_ippsSortRadixAscend_64f_I(Ipp64f *pSrcDst, Ipp64f *pTmp,
    Ipp32s len);
IppStatus fpk_dft_ssse3_ippsSortRadixAscend_8u_I(Ipp8u *pSrcDst, Ipp8u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_ssse3_ippsSortRadixDescend_16s_I(Ipp16s *pSrcDst, Ipp16s *pTmp,
    Ipp32s len);
IppStatus fpk_dft_ssse3_ippsSortRadixDescend_16u_I(Ipp16u *pSrcDst, Ipp16u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_ssse3_ippsSortRadixDescend_32f_I(Ipp32f *pSrcDst, Ipp32f *pTmp,
    Ipp32s len);
IppStatus fpk_dft_ssse3_ippsSortRadixDescend_32s_I(Ipp32s *pSrcDst, Ipp32s *pTmp,
    Ipp32s len);
IppStatus fpk_dft_ssse3_ippsSortRadixDescend_32u_I(Ipp32u *pSrcDst, Ipp32u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_ssse3_ippsSortRadixDescend_64f_I(Ipp64f *pSrcDst, Ipp64f *pTmp,
    Ipp32s len);
IppStatus fpk_dft_ssse3_ippsSortRadixDescend_8u_I(Ipp8u *pSrcDst, Ipp8u *pTmp,
    Ipp32s len);

IppStatus fpk_dft_sse42_ippsSortRadixAscend_16s_I(Ipp16s *pSrcDst, Ipp16s *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse42_ippsSortRadixAscend_16u_I(Ipp16u *pSrcDst, Ipp16u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse42_ippsSortRadixAscend_32f_I(Ipp32f *pSrcDst, Ipp32f *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse42_ippsSortRadixAscend_32s_I(Ipp32s *pSrcDst, Ipp32s *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse42_ippsSortRadixAscend_32u_I(Ipp32u *pSrcDst, Ipp32u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse42_ippsSortRadixAscend_64f_I(Ipp64f *pSrcDst, Ipp64f *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse42_ippsSortRadixAscend_8u_I(Ipp8u *pSrcDst, Ipp8u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse42_ippsSortRadixDescend_16s_I(Ipp16s *pSrcDst, Ipp16s *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse42_ippsSortRadixDescend_16u_I(Ipp16u *pSrcDst, Ipp16u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse42_ippsSortRadixDescend_32f_I(Ipp32f *pSrcDst, Ipp32f *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse42_ippsSortRadixDescend_32s_I(Ipp32s *pSrcDst, Ipp32s *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse42_ippsSortRadixDescend_32u_I(Ipp32u *pSrcDst, Ipp32u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse42_ippsSortRadixDescend_64f_I(Ipp64f *pSrcDst, Ipp64f *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse42_ippsSortRadixDescend_8u_I(Ipp8u *pSrcDst, Ipp8u *pTmp,
    Ipp32s len);

IppStatus fpk_dft_avx_ippsSortRadixAscend_16s_I(Ipp16s *pSrcDst, Ipp16s *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx_ippsSortRadixAscend_16u_I(Ipp16u *pSrcDst, Ipp16u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx_ippsSortRadixAscend_32f_I(Ipp32f *pSrcDst, Ipp32f *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx_ippsSortRadixAscend_32s_I(Ipp32s *pSrcDst, Ipp32s *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx_ippsSortRadixAscend_32u_I(Ipp32u *pSrcDst, Ipp32u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx_ippsSortRadixAscend_64f_I(Ipp64f *pSrcDst, Ipp64f *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx_ippsSortRadixAscend_8u_I(Ipp8u *pSrcDst, Ipp8u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx_ippsSortRadixDescend_16s_I(Ipp16s *pSrcDst, Ipp16s *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx_ippsSortRadixDescend_16u_I(Ipp16u *pSrcDst, Ipp16u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx_ippsSortRadixDescend_32f_I(Ipp32f *pSrcDst, Ipp32f *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx_ippsSortRadixDescend_32s_I(Ipp32s *pSrcDst, Ipp32s *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx_ippsSortRadixDescend_32u_I(Ipp32u *pSrcDst, Ipp32u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx_ippsSortRadixDescend_64f_I(Ipp64f *pSrcDst, Ipp64f *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx_ippsSortRadixDescend_8u_I(Ipp8u *pSrcDst, Ipp8u *pTmp,
    Ipp32s len);

IppStatus fpk_dft_avx2_ippsSortRadixAscend_16s_I(Ipp16s *pSrcDst, Ipp16s *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx2_ippsSortRadixAscend_16u_I(Ipp16u *pSrcDst, Ipp16u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx2_ippsSortRadixAscend_32f_I(Ipp32f *pSrcDst, Ipp32f *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx2_ippsSortRadixAscend_32s_I(Ipp32s *pSrcDst, Ipp32s *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx2_ippsSortRadixAscend_32u_I(Ipp32u *pSrcDst, Ipp32u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx2_ippsSortRadixAscend_64f_I(Ipp64f *pSrcDst, Ipp64f *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx2_ippsSortRadixAscend_8u_I(Ipp8u *pSrcDst, Ipp8u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx2_ippsSortRadixDescend_16s_I(Ipp16s *pSrcDst, Ipp16s *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx2_ippsSortRadixDescend_16u_I(Ipp16u *pSrcDst, Ipp16u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx2_ippsSortRadixDescend_32f_I(Ipp32f *pSrcDst, Ipp32f *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx2_ippsSortRadixDescend_32s_I(Ipp32s *pSrcDst, Ipp32s *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx2_ippsSortRadixDescend_32u_I(Ipp32u *pSrcDst, Ipp32u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx2_ippsSortRadixDescend_64f_I(Ipp64f *pSrcDst, Ipp64f *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx2_ippsSortRadixDescend_8u_I(Ipp8u *pSrcDst, Ipp8u *pTmp,
    Ipp32s len);

IppStatus fpk_dft_avx512_ippsSortRadixAscend_16s_I(Ipp16s *pSrcDst, Ipp16s *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx512_ippsSortRadixAscend_16u_I(Ipp16u *pSrcDst, Ipp16u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx512_ippsSortRadixAscend_32f_I(Ipp32f *pSrcDst, Ipp32f *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx512_ippsSortRadixAscend_32s_I(Ipp32s *pSrcDst, Ipp32s *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx512_ippsSortRadixAscend_32u_I(Ipp32u *pSrcDst, Ipp32u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx512_ippsSortRadixAscend_64f_I(Ipp64f *pSrcDst, Ipp64f *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx512_ippsSortRadixAscend_8u_I(Ipp8u *pSrcDst, Ipp8u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_avx512_ippsSortRadixDescend_16s_I(Ipp16s *pSrcDst,
    Ipp16s *pTmp, Ipp32s len);
IppStatus fpk_dft_avx512_ippsSortRadixDescend_16u_I(Ipp16u *pSrcDst,
    Ipp16u *pTmp, Ipp32s len);
IppStatus fpk_dft_avx512_ippsSortRadixDescend_32f_I(Ipp32f *pSrcDst,
    Ipp32f *pTmp, Ipp32s len);
IppStatus fpk_dft_avx512_ippsSortRadixDescend_32s_I(Ipp32s *pSrcDst,
    Ipp32s *pTmp, Ipp32s len);
IppStatus fpk_dft_avx512_ippsSortRadixDescend_32u_I(Ipp32u *pSrcDst,
    Ipp32u *pTmp, Ipp32s len);
IppStatus fpk_dft_avx512_ippsSortRadixDescend_64f_I(Ipp64f *pSrcDst,
    Ipp64f *pTmp, Ipp32s len);
IppStatus fpk_dft_avx512_ippsSortRadixDescend_8u_I(Ipp8u *pSrcDst, Ipp8u *pTmp,
    Ipp32s len);


dnnError_t fpk_dnn_ssse3_AllocateBuffer_F32(void **pPtr, dnnLayout_t layout);
dnnError_t fpk_dnn_ssse3_AllocateBuffer_F64(void **pPtr, dnnLayout_t layout);
dnnError_t fpk_dnn_ssse3_ConversionCreate_F32(dnnPrimitive_t* pConversion,
    const dnnLayout_t from, const dnnLayout_t to);
dnnError_t fpk_dnn_ssse3_ConversionCreate_F64(dnnPrimitive_t* pConversion,
    const dnnLayout_t from, const dnnLayout_t to);
dnnError_t fpk_dnn_ssse3_ConversionExecute_F32(dnnPrimitive_t conversion,
    void *from, void *to);
dnnError_t fpk_dnn_ssse3_ConversionExecute_F64(dnnPrimitive_t conversion,
    void *from, void *to);
dnnError_t fpk_dnn_ssse3_ConvolutionCreateBackwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t dstSize[]);
dnnError_t fpk_dnn_ssse3_ConvolutionCreateBackwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t dstSize[]);
dnnError_t fpk_dnn_ssse3_ConvolutionCreateBackwardData_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_ssse3_ConvolutionCreateBackwardData_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_ssse3_ConvolutionCreateBackwardFilter_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_ssse3_ConvolutionCreateBackwardFilter_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_ssse3_ConvolutionCreateForwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_ssse3_ConvolutionCreateForwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_ssse3_ConvolutionCreateForward_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_ssse3_ConvolutionCreateForward_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_ssse3_Delete_F32(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_ssse3_Delete_F64(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_ssse3_ExecuteAsync_F32(dnnPrimitive_t primitive,
    void *resources[]);
dnnError_t fpk_dnn_ssse3_ExecuteAsync_F64(dnnPrimitive_t primitive,
    void *resources[]);
dnnError_t fpk_dnn_ssse3_Execute_F32(dnnPrimitive_t primitive,
    void *resources[]);
dnnError_t fpk_dnn_ssse3_Execute_F64(dnnPrimitive_t primitive,
    void *resources[]);
dnnError_t fpk_dnn_ssse3_GroupsConvolutionCreateBackwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t dstSize[]);
dnnError_t fpk_dnn_ssse3_GroupsConvolutionCreateBackwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t dstSize[]);
dnnError_t fpk_dnn_ssse3_GroupsConvolutionCreateBackwardData_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_ssse3_GroupsConvolutionCreateBackwardData_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_ssse3_GroupsConvolutionCreateBackwardFilter_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_ssse3_GroupsConvolutionCreateBackwardFilter_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_ssse3_GroupsConvolutionCreateForwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_ssse3_GroupsConvolutionCreateForwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_ssse3_GroupsConvolutionCreateForward_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_ssse3_GroupsConvolutionCreateForward_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_ssse3_LRNCreateBackward_F32(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, size_t kernel_size, float alpha, float beta,
    float k);
dnnError_t fpk_dnn_ssse3_LRNCreateBackward_F64(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, size_t kernel_size, double alpha, double beta,
    double k);
dnnError_t fpk_dnn_ssse3_LRNCreateForward_F32(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    size_t kernel_size, float alpha, float beta, float k);
dnnError_t fpk_dnn_ssse3_LRNCreateForward_F64(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    size_t kernel_size, double alpha, double beta, double k);
int fpk_dnn_ssse3_LayoutCompare_F32(const dnnLayout_t l1, const dnnLayout_t l2);
int fpk_dnn_ssse3_LayoutCompare_F64(const dnnLayout_t l1, const dnnLayout_t l2);
dnnError_t fpk_dnn_ssse3_LayoutCreateFromPrimitive_F32(dnnLayout_t *pLayout,
    const dnnPrimitive_t primitive, dnnResourceType_t type);
dnnError_t fpk_dnn_ssse3_LayoutCreateFromPrimitive_F64(dnnLayout_t *pLayout,
    const dnnPrimitive_t primitive, dnnResourceType_t type);
dnnError_t fpk_dnn_ssse3_LayoutCreate_F32(dnnLayout_t *pLayout, size_t dimension,
    const size_t size[], const size_t strides[]);
dnnError_t fpk_dnn_ssse3_LayoutCreate_F64(dnnLayout_t *pLayout, size_t dimension,
    const size_t size[], const size_t strides[]);
dnnError_t fpk_dnn_ssse3_LayoutDelete_F32(dnnLayout_t layout);
dnnError_t fpk_dnn_ssse3_LayoutDelete_F64(dnnLayout_t layout);
size_t fpk_dnn_ssse3_LayoutGetMemorySize_F32(const dnnLayout_t layout);
size_t fpk_dnn_ssse3_LayoutGetMemorySize_F64(const dnnLayout_t layout);
dnnError_t fpk_dnn_ssse3_PoolingCreateBackward_F32(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_ssse3_PoolingCreateBackward_F64(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_ssse3_PoolingCreateForward_F32(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_ssse3_PoolingCreateForward_F64(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_ssse3_ReLUCreateBackward_F32(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, float negativeSlope);
dnnError_t fpk_dnn_ssse3_ReLUCreateBackward_F64(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, double negativeSlope);
dnnError_t fpk_dnn_ssse3_ReLUCreateForward_F32(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    float negativeSlope);
dnnError_t fpk_dnn_ssse3_ReLUCreateForward_F64(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    double negativeSlope);
dnnError_t fpk_dnn_ssse3_ReleaseBuffer_F32(void *ptr);
dnnError_t fpk_dnn_ssse3_ReleaseBuffer_F64(void *ptr);
dnnError_t fpk_dnn_ssse3_WaitFor_F32(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_ssse3_WaitFor_F64(dnnPrimitive_t primitive);

dnnError_t fpk_dnn_sse42_AllocateBuffer_F32(void **pPtr, dnnLayout_t layout);
dnnError_t fpk_dnn_sse42_AllocateBuffer_F64(void **pPtr, dnnLayout_t layout);
dnnError_t fpk_dnn_sse42_ConversionCreate_F32(dnnPrimitive_t* pConversion,
    const dnnLayout_t from, const dnnLayout_t to);
dnnError_t fpk_dnn_sse42_ConversionCreate_F64(dnnPrimitive_t* pConversion,
    const dnnLayout_t from, const dnnLayout_t to);
dnnError_t fpk_dnn_sse42_ConversionExecute_F32(dnnPrimitive_t conversion,
    void *from, void *to);
dnnError_t fpk_dnn_sse42_ConversionExecute_F64(dnnPrimitive_t conversion,
    void *from, void *to);
dnnError_t fpk_dnn_sse42_ConvolutionCreateBackwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t dstSize[]);
dnnError_t fpk_dnn_sse42_ConvolutionCreateBackwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t dstSize[]);
dnnError_t fpk_dnn_sse42_ConvolutionCreateBackwardData_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse42_ConvolutionCreateBackwardData_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse42_ConvolutionCreateBackwardFilter_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse42_ConvolutionCreateBackwardFilter_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse42_ConvolutionCreateForwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse42_ConvolutionCreateForwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse42_ConvolutionCreateForward_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse42_ConvolutionCreateForward_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse42_Delete_F32(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_sse42_Delete_F64(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_sse42_ExecuteAsync_F32(dnnPrimitive_t primitive,
    void *resources[]);
dnnError_t fpk_dnn_sse42_ExecuteAsync_F64(dnnPrimitive_t primitive,
    void *resources[]);
dnnError_t fpk_dnn_sse42_Execute_F32(dnnPrimitive_t primitive,
    void *resources[]);
dnnError_t fpk_dnn_sse42_Execute_F64(dnnPrimitive_t primitive,
    void *resources[]);
dnnError_t fpk_dnn_sse42_GroupsConvolutionCreateBackwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t dstSize[]);
dnnError_t fpk_dnn_sse42_GroupsConvolutionCreateBackwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t dstSize[]);
dnnError_t fpk_dnn_sse42_GroupsConvolutionCreateBackwardData_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse42_GroupsConvolutionCreateBackwardData_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse42_GroupsConvolutionCreateBackwardFilter_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse42_GroupsConvolutionCreateBackwardFilter_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse42_GroupsConvolutionCreateForwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse42_GroupsConvolutionCreateForwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse42_GroupsConvolutionCreateForward_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse42_GroupsConvolutionCreateForward_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse42_LRNCreateBackward_F32(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, size_t kernel_size, float alpha, float beta,
    float k);
dnnError_t fpk_dnn_sse42_LRNCreateBackward_F64(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, size_t kernel_size, double alpha, double beta,
    double k);
dnnError_t fpk_dnn_sse42_LRNCreateForward_F32(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    size_t kernel_size, float alpha, float beta, float k);
dnnError_t fpk_dnn_sse42_LRNCreateForward_F64(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    size_t kernel_size, double alpha, double beta, double k);
int fpk_dnn_sse42_LayoutCompare_F32(const dnnLayout_t l1, const dnnLayout_t l2);
int fpk_dnn_sse42_LayoutCompare_F64(const dnnLayout_t l1, const dnnLayout_t l2);
dnnError_t fpk_dnn_sse42_LayoutCreateFromPrimitive_F32(dnnLayout_t *pLayout,
    const dnnPrimitive_t primitive, dnnResourceType_t type);
dnnError_t fpk_dnn_sse42_LayoutCreateFromPrimitive_F64(dnnLayout_t *pLayout,
    const dnnPrimitive_t primitive, dnnResourceType_t type);
dnnError_t fpk_dnn_sse42_LayoutCreate_F32(dnnLayout_t *pLayout, size_t dimension,
    const size_t size[], const size_t strides[]);
dnnError_t fpk_dnn_sse42_LayoutCreate_F64(dnnLayout_t *pLayout, size_t dimension,
    const size_t size[], const size_t strides[]);
dnnError_t fpk_dnn_sse42_LayoutDelete_F32(dnnLayout_t layout);
dnnError_t fpk_dnn_sse42_LayoutDelete_F64(dnnLayout_t layout);
size_t fpk_dnn_sse42_LayoutGetMemorySize_F32(const dnnLayout_t layout);
size_t fpk_dnn_sse42_LayoutGetMemorySize_F64(const dnnLayout_t layout);
dnnError_t fpk_dnn_sse42_PoolingCreateBackward_F32(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse42_PoolingCreateBackward_F64(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse42_PoolingCreateForward_F32(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse42_PoolingCreateForward_F64(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse42_ReLUCreateBackward_F32(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, float negativeSlope);
dnnError_t fpk_dnn_sse42_ReLUCreateBackward_F64(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, double negativeSlope);
dnnError_t fpk_dnn_sse42_ReLUCreateForward_F32(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    float negativeSlope);
dnnError_t fpk_dnn_sse42_ReLUCreateForward_F64(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    double negativeSlope);
dnnError_t fpk_dnn_sse42_ReleaseBuffer_F32(void *ptr);
dnnError_t fpk_dnn_sse42_ReleaseBuffer_F64(void *ptr);
dnnError_t fpk_dnn_sse42_WaitFor_F32(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_sse42_WaitFor_F64(dnnPrimitive_t primitive);

dnnError_t fpk_dnn_avx_AllocateBuffer_F32(void **pPtr, dnnLayout_t layout);
dnnError_t fpk_dnn_avx_AllocateBuffer_F64(void **pPtr, dnnLayout_t layout);
dnnError_t fpk_dnn_avx_ConversionCreate_F32(dnnPrimitive_t* pConversion,
    const dnnLayout_t from, const dnnLayout_t to);
dnnError_t fpk_dnn_avx_ConversionCreate_F64(dnnPrimitive_t* pConversion,
    const dnnLayout_t from, const dnnLayout_t to);
dnnError_t fpk_dnn_avx_ConversionExecute_F32(dnnPrimitive_t conversion,
    void *from, void *to);
dnnError_t fpk_dnn_avx_ConversionExecute_F64(dnnPrimitive_t conversion,
    void *from, void *to);
dnnError_t fpk_dnn_avx_ConvolutionCreateBackwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t dstSize[]);
dnnError_t fpk_dnn_avx_ConvolutionCreateBackwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t dstSize[]);
dnnError_t fpk_dnn_avx_ConvolutionCreateBackwardData_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx_ConvolutionCreateBackwardData_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx_ConvolutionCreateBackwardFilter_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx_ConvolutionCreateBackwardFilter_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx_ConvolutionCreateForwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx_ConvolutionCreateForwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx_ConvolutionCreateForward_F32(dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t algorithm,
    size_t dimension, const size_t srcSize[], const size_t dstSize[],
    const size_t filterSize[], const size_t convolutionStrides[],
    const int inputOffset[], const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx_ConvolutionCreateForward_F64(dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t algorithm,
    size_t dimension, const size_t srcSize[], const size_t dstSize[],
    const size_t filterSize[], const size_t convolutionStrides[],
    const int inputOffset[], const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx_Delete_F32(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_avx_Delete_F64(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_avx_ExecuteAsync_F32(dnnPrimitive_t primitive,
    void *resources[]);
dnnError_t fpk_dnn_avx_ExecuteAsync_F64(dnnPrimitive_t primitive,
    void *resources[]);
dnnError_t fpk_dnn_avx_Execute_F32(dnnPrimitive_t primitive, void *resources[]);
dnnError_t fpk_dnn_avx_Execute_F64(dnnPrimitive_t primitive, void *resources[]);
dnnError_t fpk_dnn_avx_GroupsConvolutionCreateBackwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t dstSize[]);
dnnError_t fpk_dnn_avx_GroupsConvolutionCreateBackwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t dstSize[]);
dnnError_t fpk_dnn_avx_GroupsConvolutionCreateBackwardData_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx_GroupsConvolutionCreateBackwardData_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx_GroupsConvolutionCreateBackwardFilter_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx_GroupsConvolutionCreateBackwardFilter_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx_GroupsConvolutionCreateForwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx_GroupsConvolutionCreateForwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx_GroupsConvolutionCreateForward_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx_GroupsConvolutionCreateForward_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx_LRNCreateBackward_F32(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, size_t kernel_size, float alpha, float beta,
    float k);
dnnError_t fpk_dnn_avx_LRNCreateBackward_F64(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, size_t kernel_size, double alpha, double beta,
    double k);
dnnError_t fpk_dnn_avx_LRNCreateForward_F32(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    size_t kernel_size, float alpha, float beta, float k);
dnnError_t fpk_dnn_avx_LRNCreateForward_F64(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    size_t kernel_size, double alpha, double beta, double k);
int fpk_dnn_avx_LayoutCompare_F32(const dnnLayout_t l1, const dnnLayout_t l2);
int fpk_dnn_avx_LayoutCompare_F64(const dnnLayout_t l1, const dnnLayout_t l2);
dnnError_t fpk_dnn_avx_LayoutCreateFromPrimitive_F32(dnnLayout_t *pLayout,
    const dnnPrimitive_t primitive, dnnResourceType_t type);
dnnError_t fpk_dnn_avx_LayoutCreateFromPrimitive_F64(dnnLayout_t *pLayout,
    const dnnPrimitive_t primitive, dnnResourceType_t type);
dnnError_t fpk_dnn_avx_LayoutCreate_F32(dnnLayout_t *pLayout, size_t dimension,
    const size_t size[], const size_t strides[]);
dnnError_t fpk_dnn_avx_LayoutCreate_F64(dnnLayout_t *pLayout, size_t dimension,
    const size_t size[], const size_t strides[]);
dnnError_t fpk_dnn_avx_LayoutDelete_F32(dnnLayout_t layout);
dnnError_t fpk_dnn_avx_LayoutDelete_F64(dnnLayout_t layout);
size_t fpk_dnn_avx_LayoutGetMemorySize_F32(const dnnLayout_t layout);
size_t fpk_dnn_avx_LayoutGetMemorySize_F64(const dnnLayout_t layout);
dnnError_t fpk_dnn_avx_PoolingCreateBackward_F32(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx_PoolingCreateBackward_F64(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx_PoolingCreateForward_F32(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx_PoolingCreateForward_F64(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx_ReLUCreateBackward_F32(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, float negativeSlope);
dnnError_t fpk_dnn_avx_ReLUCreateBackward_F64(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, double negativeSlope);
dnnError_t fpk_dnn_avx_ReLUCreateForward_F32(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    float negativeSlope);
dnnError_t fpk_dnn_avx_ReLUCreateForward_F64(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    double negativeSlope);
dnnError_t fpk_dnn_avx_ReleaseBuffer_F32(void *ptr);
dnnError_t fpk_dnn_avx_ReleaseBuffer_F64(void *ptr);
dnnError_t fpk_dnn_avx_WaitFor_F32(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_avx_WaitFor_F64(dnnPrimitive_t primitive);

dnnError_t fpk_dnn_avx2_AllocateBuffer_F32(void **pPtr, dnnLayout_t layout);
dnnError_t fpk_dnn_avx2_AllocateBuffer_F64(void **pPtr, dnnLayout_t layout);
dnnError_t fpk_dnn_avx2_ConversionCreate_F32(dnnPrimitive_t* pConversion,
    const dnnLayout_t from, const dnnLayout_t to);
dnnError_t fpk_dnn_avx2_ConversionCreate_F64(dnnPrimitive_t* pConversion,
    const dnnLayout_t from, const dnnLayout_t to);
dnnError_t fpk_dnn_avx2_ConversionExecute_F32(dnnPrimitive_t conversion,
    void *from, void *to);
dnnError_t fpk_dnn_avx2_ConversionExecute_F64(dnnPrimitive_t conversion,
    void *from, void *to);
dnnError_t fpk_dnn_avx2_ConvolutionCreateBackwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t dstSize[]);
dnnError_t fpk_dnn_avx2_ConvolutionCreateBackwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t dstSize[]);
dnnError_t fpk_dnn_avx2_ConvolutionCreateBackwardData_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx2_ConvolutionCreateBackwardData_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx2_ConvolutionCreateBackwardFilter_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx2_ConvolutionCreateBackwardFilter_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx2_ConvolutionCreateForwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx2_ConvolutionCreateForwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx2_ConvolutionCreateForward_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx2_ConvolutionCreateForward_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx2_Delete_F32(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_avx2_Delete_F64(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_avx2_ExecuteAsync_F32(dnnPrimitive_t primitive,
    void *resources[]);
dnnError_t fpk_dnn_avx2_ExecuteAsync_F64(dnnPrimitive_t primitive,
    void *resources[]);
dnnError_t fpk_dnn_avx2_Execute_F32(dnnPrimitive_t primitive, void *resources[]);
dnnError_t fpk_dnn_avx2_Execute_F64(dnnPrimitive_t primitive, void *resources[]);
dnnError_t fpk_dnn_avx2_GroupsConvolutionCreateBackwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t dstSize[]);
dnnError_t fpk_dnn_avx2_GroupsConvolutionCreateBackwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t dstSize[]);
dnnError_t fpk_dnn_avx2_GroupsConvolutionCreateBackwardData_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx2_GroupsConvolutionCreateBackwardData_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx2_GroupsConvolutionCreateBackwardFilter_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx2_GroupsConvolutionCreateBackwardFilter_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx2_GroupsConvolutionCreateForwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx2_GroupsConvolutionCreateForwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx2_GroupsConvolutionCreateForward_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx2_GroupsConvolutionCreateForward_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx2_LRNCreateBackward_F32(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, size_t kernel_size, float alpha, float beta,
    float k);
dnnError_t fpk_dnn_avx2_LRNCreateBackward_F64(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, size_t kernel_size, double alpha, double beta,
    double k);
dnnError_t fpk_dnn_avx2_LRNCreateForward_F32(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    size_t kernel_size, float alpha, float beta, float k);
dnnError_t fpk_dnn_avx2_LRNCreateForward_F64(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    size_t kernel_size, double alpha, double beta, double k);
int fpk_dnn_avx2_LayoutCompare_F32(const dnnLayout_t l1, const dnnLayout_t l2);
int fpk_dnn_avx2_LayoutCompare_F64(const dnnLayout_t l1, const dnnLayout_t l2);
dnnError_t fpk_dnn_avx2_LayoutCreateFromPrimitive_F32(dnnLayout_t *pLayout,
    const dnnPrimitive_t primitive, dnnResourceType_t type);
dnnError_t fpk_dnn_avx2_LayoutCreateFromPrimitive_F64(dnnLayout_t *pLayout,
    const dnnPrimitive_t primitive, dnnResourceType_t type);
dnnError_t fpk_dnn_avx2_LayoutCreate_F32(dnnLayout_t *pLayout, size_t dimension,
    const size_t size[], const size_t strides[]);
dnnError_t fpk_dnn_avx2_LayoutCreate_F64(dnnLayout_t *pLayout, size_t dimension,
    const size_t size[], const size_t strides[]);
dnnError_t fpk_dnn_avx2_LayoutDelete_F32(dnnLayout_t layout);
dnnError_t fpk_dnn_avx2_LayoutDelete_F64(dnnLayout_t layout);
size_t fpk_dnn_avx2_LayoutGetMemorySize_F32(const dnnLayout_t layout);
size_t fpk_dnn_avx2_LayoutGetMemorySize_F64(const dnnLayout_t layout);
dnnError_t fpk_dnn_avx2_PoolingCreateBackward_F32(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx2_PoolingCreateBackward_F64(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx2_PoolingCreateForward_F32(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx2_PoolingCreateForward_F64(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx2_ReLUCreateBackward_F32(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, float negativeSlope);
dnnError_t fpk_dnn_avx2_ReLUCreateBackward_F64(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, double negativeSlope);
dnnError_t fpk_dnn_avx2_ReLUCreateForward_F32(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    float negativeSlope);
dnnError_t fpk_dnn_avx2_ReLUCreateForward_F64(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    double negativeSlope);
dnnError_t fpk_dnn_avx2_ReleaseBuffer_F32(void *ptr);
dnnError_t fpk_dnn_avx2_ReleaseBuffer_F64(void *ptr);
dnnError_t fpk_dnn_avx2_WaitFor_F32(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_avx2_WaitFor_F64(dnnPrimitive_t primitive);

dnnError_t fpk_dnn_avx512_AllocateBuffer_F32(void **pPtr, dnnLayout_t layout);
dnnError_t fpk_dnn_avx512_AllocateBuffer_F64(void **pPtr, dnnLayout_t layout);
dnnError_t fpk_dnn_avx512_ConversionCreate_F32(dnnPrimitive_t* pConversion,
    const dnnLayout_t from, const dnnLayout_t to);
dnnError_t fpk_dnn_avx512_ConversionCreate_F64(dnnPrimitive_t* pConversion,
    const dnnLayout_t from, const dnnLayout_t to);
dnnError_t fpk_dnn_avx512_ConversionExecute_F32(dnnPrimitive_t conversion,
    void *from, void *to);
dnnError_t fpk_dnn_avx512_ConversionExecute_F64(dnnPrimitive_t conversion,
    void *from, void *to);
dnnError_t fpk_dnn_avx512_ConvolutionCreateBackwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t dstSize[]);
dnnError_t fpk_dnn_avx512_ConvolutionCreateBackwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t dstSize[]);
dnnError_t fpk_dnn_avx512_ConvolutionCreateBackwardData_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_ConvolutionCreateBackwardData_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_ConvolutionCreateBackwardFilter_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_ConvolutionCreateBackwardFilter_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_ConvolutionCreateForwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_ConvolutionCreateForwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_ConvolutionCreateForward_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_ConvolutionCreateForward_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_Delete_F32(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_avx512_Delete_F64(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_avx512_ExecuteAsync_F32(dnnPrimitive_t primitive,
    void *resources[]);
dnnError_t fpk_dnn_avx512_ExecuteAsync_F64(dnnPrimitive_t primitive,
    void *resources[]);
dnnError_t fpk_dnn_avx512_Execute_F32(dnnPrimitive_t primitive,
    void *resources[]);
dnnError_t fpk_dnn_avx512_Execute_F64(dnnPrimitive_t primitive,
    void *resources[]);
dnnError_t fpk_dnn_avx512_GroupsConvolutionCreateBackwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t dstSize[]);
dnnError_t fpk_dnn_avx512_GroupsConvolutionCreateBackwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t dstSize[]);
dnnError_t fpk_dnn_avx512_GroupsConvolutionCreateBackwardData_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_GroupsConvolutionCreateBackwardData_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_GroupsConvolutionCreateBackwardFilter_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_GroupsConvolutionCreateBackwardFilter_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_GroupsConvolutionCreateForwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_GroupsConvolutionCreateForwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_GroupsConvolutionCreateForward_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_GroupsConvolutionCreateForward_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_LRNCreateBackward_F32(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, size_t kernel_size, float alpha, float beta,
    float k);
dnnError_t fpk_dnn_avx512_LRNCreateBackward_F64(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, size_t kernel_size, double alpha, double beta,
    double k);
dnnError_t fpk_dnn_avx512_LRNCreateForward_F32(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    size_t kernel_size, float alpha, float beta, float k);
dnnError_t fpk_dnn_avx512_LRNCreateForward_F64(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    size_t kernel_size, double alpha, double beta, double k);
int fpk_dnn_avx512_LayoutCompare_F32(const dnnLayout_t l1, const dnnLayout_t l2);
int fpk_dnn_avx512_LayoutCompare_F64(const dnnLayout_t l1, const dnnLayout_t l2);
dnnError_t fpk_dnn_avx512_LayoutCreateFromPrimitive_F32(dnnLayout_t *pLayout,
    const dnnPrimitive_t primitive, dnnResourceType_t type);
dnnError_t fpk_dnn_avx512_LayoutCreateFromPrimitive_F64(dnnLayout_t *pLayout,
    const dnnPrimitive_t primitive, dnnResourceType_t type);
dnnError_t fpk_dnn_avx512_LayoutCreate_F32(dnnLayout_t *pLayout,
    size_t dimension, const size_t size[], const size_t strides[]);
dnnError_t fpk_dnn_avx512_LayoutCreate_F64(dnnLayout_t *pLayout,
    size_t dimension, const size_t size[], const size_t strides[]);
dnnError_t fpk_dnn_avx512_LayoutDelete_F32(dnnLayout_t layout);
dnnError_t fpk_dnn_avx512_LayoutDelete_F64(dnnLayout_t layout);
size_t fpk_dnn_avx512_LayoutGetMemorySize_F32(const dnnLayout_t layout);
size_t fpk_dnn_avx512_LayoutGetMemorySize_F64(const dnnLayout_t layout);
dnnError_t fpk_dnn_avx512_PoolingCreateBackward_F32(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_PoolingCreateBackward_F64(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_PoolingCreateForward_F32(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_PoolingCreateForward_F64(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_ReLUCreateBackward_F32(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, float negativeSlope);
dnnError_t fpk_dnn_avx512_ReLUCreateBackward_F64(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, double negativeSlope);
dnnError_t fpk_dnn_avx512_ReLUCreateForward_F32(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    float negativeSlope);
dnnError_t fpk_dnn_avx512_ReLUCreateForward_F64(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    double negativeSlope);
dnnError_t fpk_dnn_avx512_ReleaseBuffer_F32(void *ptr);
dnnError_t fpk_dnn_avx512_ReleaseBuffer_F64(void *ptr);
dnnError_t fpk_dnn_avx512_WaitFor_F32(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_avx512_WaitFor_F64(dnnPrimitive_t primitive);


void fpk_feast_ssse3_cfeast_hbev(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const MKL_Complex8* a, const MKL_INT* lda, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, MKL_Complex8* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_ssse3_cfeast_hbgv(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const MKL_Complex8* a, const MKL_INT* lda,
    const MKL_INT* klb, const MKL_Complex8* b, const MKL_INT* ldb, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, MKL_Complex8* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_ssse3_cfeast_hcsrev(const char* uplo, const MKL_INT* n,
    const MKL_Complex8* sa, const MKL_INT* isa, const MKL_INT* jsa, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, MKL_Complex8* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_ssse3_cfeast_hcsrgv(const char* uplo, const MKL_INT* n,
    const MKL_Complex8* sa, const MKL_INT* isa, const MKL_INT* jsa,
    const MKL_Complex8* sb, const MKL_INT* isb, const MKL_INT* jsb, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, MKL_Complex8* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_ssse3_cfeast_heev(const char* uplo, const MKL_INT* n,
    const MKL_Complex8* a, const MKL_INT* lda, MKL_INT* fpm, float* epsout,
    MKL_INT* loop, const float* emin, const float* emax, MKL_INT* m0, float* e,
    MKL_Complex8* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_ssse3_cfeast_hegv(const char* uplo, const MKL_INT* n,
    const MKL_Complex8* a, const MKL_INT* lda, const MKL_Complex8* b,
    const MKL_INT* ldb, MKL_INT* fpm, float* epsout, MKL_INT* loop,
    const float* emin, const float* emax, MKL_INT* m0, float* e, MKL_Complex8* x,
    MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_ssse3_cfeast_hrci(MKL_INT* ijob, const MKL_INT* n,
    MKL_Complex8* ze, MKL_Complex8* work, MKL_Complex8* workc, MKL_Complex8* zaq,
    MKL_Complex8* zsq, MKL_INT* fpm, float* epsout, MKL_INT* loop,
    const float* emin, const float* emax, MKL_INT* m0, float* lambda,
    MKL_Complex8* q, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_ssse3_dfeast_sbev(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const double* a, const MKL_INT* lda, MKL_INT* fpm,
    double* epsout, MKL_INT* loop, const double* emin, const double* emax,
    MKL_INT* m0, double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_ssse3_dfeast_sbgv(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const double* a, const MKL_INT* lda, const MKL_INT* klb,
    const double* b, const MKL_INT* ldb, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_ssse3_dfeast_scsrev(const char* uplo, const MKL_INT* n,
    const double* sa, const MKL_INT* isa, const MKL_INT* jsa, MKL_INT* fpm,
    double* epsout, MKL_INT* loop, const double* emin, const double* emax,
    MKL_INT* m0, double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_ssse3_dfeast_scsrgv(const char* uplo, const MKL_INT* n,
    const double* sa, const MKL_INT* isa, const MKL_INT* jsa, const double* sb,
    const MKL_INT* isb, const MKL_INT* jsb, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_ssse3_dfeast_srci(MKL_INT* ijob, const MKL_INT* n,
    MKL_Complex16* ze, double* work, MKL_Complex16* workc, double* aq,
    double* sq, MKL_INT* fpm, double* epsout, MKL_INT* loop, const double* emin,
    const double* emax, MKL_INT* m0, double* lambda, double* q, MKL_INT* m,
    double* res, MKL_INT* info);
void fpk_feast_ssse3_dfeast_syev(const char* uplo, const MKL_INT* n,
    const double* a, const MKL_INT* lda, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_ssse3_dfeast_sygv(const char* uplo, const MKL_INT* n,
    const double* a, const MKL_INT* lda, const double* b, const MKL_INT* ldb,
    MKL_INT* fpm, double* epsout, MKL_INT* loop, const double* emin,
    const double* emax, MKL_INT* m0, double* e, double* x, MKL_INT* m,
    double* res, MKL_INT* info);
void fpk_feast_ssse3_feastinit(MKL_INT* fpm);
void fpk_feast_ssse3_sfeast_sbev(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const float* a, const MKL_INT* lda, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_ssse3_sfeast_sbgv(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const float* a, const MKL_INT* lda, const MKL_INT* klb,
    const float* b, const MKL_INT* ldb, MKL_INT* fpm, float* epsout,
    MKL_INT* loop, const float* emin, const float* emax, MKL_INT* m0, float* e,
    float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_ssse3_sfeast_scsrev(const char* uplo, const MKL_INT* n,
    const float* sa, const MKL_INT* isa, const MKL_INT* jsa, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_ssse3_sfeast_scsrgv(const char* uplo, const MKL_INT* n,
    const float* sa, const MKL_INT* isa, const MKL_INT* jsa, const float* sb,
    const MKL_INT* isb, const MKL_INT* jsb, MKL_INT* fpm, float* epsout,
    MKL_INT* loop, const float* emin, const float* emax, MKL_INT* m0, float* e,
    float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_ssse3_sfeast_srci(MKL_INT* ijob, const MKL_INT* n,
    MKL_Complex8* ze, float* work, MKL_Complex8* workc, float* aq, float* sq,
    MKL_INT* fpm, float* epsout, MKL_INT* loop, const float* emin,
    const float* emax, MKL_INT* m0, float* lambda, float* q, MKL_INT* m,
    float* res, MKL_INT* info);
void fpk_feast_ssse3_sfeast_syev(const char* uplo, const MKL_INT* n,
    const float* a, const MKL_INT* lda, MKL_INT* fpm, float* epsout,
    MKL_INT* loop, const float* emin, const float* emax, MKL_INT* m0, float* e,
    float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_ssse3_sfeast_sygv(const char* uplo, const MKL_INT* n,
    const float* a, const MKL_INT* lda, const float* b, const MKL_INT* ldb,
    MKL_INT* fpm, float* epsout, MKL_INT* loop, const float* emin,
    const float* emax, MKL_INT* m0, float* e, float* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_ssse3_zfeast_hbev(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const MKL_Complex16* a, const MKL_INT* lda, MKL_INT* fpm,
    double* epsout, MKL_INT* loop, const double* emin, const double* emax,
    MKL_INT* m0, double* e, MKL_Complex16* x, MKL_INT* m, double* res,
    MKL_INT* info);
void fpk_feast_ssse3_zfeast_hbgv(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const MKL_Complex16* a, const MKL_INT* lda,
    const MKL_INT* klb, const MKL_Complex16* b, const MKL_INT* ldb, MKL_INT* fpm,
    double* epsout, MKL_INT* loop, const double* emin, const double* emax,
    MKL_INT* m0, double* e, MKL_Complex16* x, MKL_INT* m, double* res,
    MKL_INT* info);
void fpk_feast_ssse3_zfeast_hcsrev(const char* uplo, const MKL_INT* n,
    const MKL_Complex16* sa, const MKL_INT* isa, const MKL_INT* jsa,
    MKL_INT* fpm, double* epsout, MKL_INT* loop, const double* emin,
    const double* emax, MKL_INT* m0, double* e, MKL_Complex16* x, MKL_INT* m,
    double* res, MKL_INT* info);
void fpk_feast_ssse3_zfeast_hcsrgv(const char* uplo, const MKL_INT* n,
    const MKL_Complex16* sa, const MKL_INT* isa, const MKL_INT* jsa,
    const MKL_Complex16* sb, const MKL_INT* isb, const MKL_INT* jsb,
    MKL_INT* fpm, double* epsout, MKL_INT* loop, const double* emin,
    const double* emax, MKL_INT* m0, double* e, MKL_Complex16* x, MKL_INT* m,
    double* res, MKL_INT* info);
void fpk_feast_ssse3_zfeast_heev(const char* uplo, const MKL_INT* n,
    const MKL_Complex16* a, const MKL_INT* lda, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* e, MKL_Complex16* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_ssse3_zfeast_hegv(const char* uplo, const MKL_INT* n,
    const MKL_Complex16* a, const MKL_INT* lda, const MKL_Complex16* b,
    const MKL_INT* ldb, MKL_INT* fpm, double* epsout, MKL_INT* loop,
    const double* emin, const double* emax, MKL_INT* m0, double* e,
    MKL_Complex16* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_ssse3_zfeast_hrci(MKL_INT* ijob, const MKL_INT* n,
    MKL_Complex16* ze, MKL_Complex16* work, MKL_Complex16* workc,
    MKL_Complex16* zaq, MKL_Complex16* zsq, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* lambda, MKL_Complex16* q, MKL_INT* m, double* res, MKL_INT* info);

void fpk_feast_sse42_cfeast_hbev(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const MKL_Complex8* a, const MKL_INT* lda, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, MKL_Complex8* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_sse42_cfeast_hbgv(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const MKL_Complex8* a, const MKL_INT* lda,
    const MKL_INT* klb, const MKL_Complex8* b, const MKL_INT* ldb, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, MKL_Complex8* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_sse42_cfeast_hcsrev(const char* uplo, const MKL_INT* n,
    const MKL_Complex8* sa, const MKL_INT* isa, const MKL_INT* jsa, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, MKL_Complex8* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_sse42_cfeast_hcsrgv(const char* uplo, const MKL_INT* n,
    const MKL_Complex8* sa, const MKL_INT* isa, const MKL_INT* jsa,
    const MKL_Complex8* sb, const MKL_INT* isb, const MKL_INT* jsb, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, MKL_Complex8* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_sse42_cfeast_heev(const char* uplo, const MKL_INT* n,
    const MKL_Complex8* a, const MKL_INT* lda, MKL_INT* fpm, float* epsout,
    MKL_INT* loop, const float* emin, const float* emax, MKL_INT* m0, float* e,
    MKL_Complex8* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_sse42_cfeast_hegv(const char* uplo, const MKL_INT* n,
    const MKL_Complex8* a, const MKL_INT* lda, const MKL_Complex8* b,
    const MKL_INT* ldb, MKL_INT* fpm, float* epsout, MKL_INT* loop,
    const float* emin, const float* emax, MKL_INT* m0, float* e, MKL_Complex8* x,
    MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_sse42_cfeast_hrci(MKL_INT* ijob, const MKL_INT* n,
    MKL_Complex8* ze, MKL_Complex8* work, MKL_Complex8* workc, MKL_Complex8* zaq,
    MKL_Complex8* zsq, MKL_INT* fpm, float* epsout, MKL_INT* loop,
    const float* emin, const float* emax, MKL_INT* m0, float* lambda,
    MKL_Complex8* q, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_sse42_dfeast_sbev(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const double* a, const MKL_INT* lda, MKL_INT* fpm,
    double* epsout, MKL_INT* loop, const double* emin, const double* emax,
    MKL_INT* m0, double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_sse42_dfeast_sbgv(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const double* a, const MKL_INT* lda, const MKL_INT* klb,
    const double* b, const MKL_INT* ldb, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_sse42_dfeast_scsrev(const char* uplo, const MKL_INT* n,
    const double* sa, const MKL_INT* isa, const MKL_INT* jsa, MKL_INT* fpm,
    double* epsout, MKL_INT* loop, const double* emin, const double* emax,
    MKL_INT* m0, double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_sse42_dfeast_scsrgv(const char* uplo, const MKL_INT* n,
    const double* sa, const MKL_INT* isa, const MKL_INT* jsa, const double* sb,
    const MKL_INT* isb, const MKL_INT* jsb, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_sse42_dfeast_srci(MKL_INT* ijob, const MKL_INT* n,
    MKL_Complex16* ze, double* work, MKL_Complex16* workc, double* aq,
    double* sq, MKL_INT* fpm, double* epsout, MKL_INT* loop, const double* emin,
    const double* emax, MKL_INT* m0, double* lambda, double* q, MKL_INT* m,
    double* res, MKL_INT* info);
void fpk_feast_sse42_dfeast_syev(const char* uplo, const MKL_INT* n,
    const double* a, const MKL_INT* lda, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_sse42_dfeast_sygv(const char* uplo, const MKL_INT* n,
    const double* a, const MKL_INT* lda, const double* b, const MKL_INT* ldb,
    MKL_INT* fpm, double* epsout, MKL_INT* loop, const double* emin,
    const double* emax, MKL_INT* m0, double* e, double* x, MKL_INT* m,
    double* res, MKL_INT* info);
void fpk_feast_sse42_feastinit(MKL_INT* fpm);
void fpk_feast_sse42_sfeast_sbev(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const float* a, const MKL_INT* lda, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_sse42_sfeast_sbgv(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const float* a, const MKL_INT* lda, const MKL_INT* klb,
    const float* b, const MKL_INT* ldb, MKL_INT* fpm, float* epsout,
    MKL_INT* loop, const float* emin, const float* emax, MKL_INT* m0, float* e,
    float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_sse42_sfeast_scsrev(const char* uplo, const MKL_INT* n,
    const float* sa, const MKL_INT* isa, const MKL_INT* jsa, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_sse42_sfeast_scsrgv(const char* uplo, const MKL_INT* n,
    const float* sa, const MKL_INT* isa, const MKL_INT* jsa, const float* sb,
    const MKL_INT* isb, const MKL_INT* jsb, MKL_INT* fpm, float* epsout,
    MKL_INT* loop, const float* emin, const float* emax, MKL_INT* m0, float* e,
    float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_sse42_sfeast_srci(MKL_INT* ijob, const MKL_INT* n,
    MKL_Complex8* ze, float* work, MKL_Complex8* workc, float* aq, float* sq,
    MKL_INT* fpm, float* epsout, MKL_INT* loop, const float* emin,
    const float* emax, MKL_INT* m0, float* lambda, float* q, MKL_INT* m,
    float* res, MKL_INT* info);
void fpk_feast_sse42_sfeast_syev(const char* uplo, const MKL_INT* n,
    const float* a, const MKL_INT* lda, MKL_INT* fpm, float* epsout,
    MKL_INT* loop, const float* emin, const float* emax, MKL_INT* m0, float* e,
    float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_sse42_sfeast_sygv(const char* uplo, const MKL_INT* n,
    const float* a, const MKL_INT* lda, const float* b, const MKL_INT* ldb,
    MKL_INT* fpm, float* epsout, MKL_INT* loop, const float* emin,
    const float* emax, MKL_INT* m0, float* e, float* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_sse42_zfeast_hbev(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const MKL_Complex16* a, const MKL_INT* lda, MKL_INT* fpm,
    double* epsout, MKL_INT* loop, const double* emin, const double* emax,
    MKL_INT* m0, double* e, MKL_Complex16* x, MKL_INT* m, double* res,
    MKL_INT* info);
void fpk_feast_sse42_zfeast_hbgv(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const MKL_Complex16* a, const MKL_INT* lda,
    const MKL_INT* klb, const MKL_Complex16* b, const MKL_INT* ldb, MKL_INT* fpm,
    double* epsout, MKL_INT* loop, const double* emin, const double* emax,
    MKL_INT* m0, double* e, MKL_Complex16* x, MKL_INT* m, double* res,
    MKL_INT* info);
void fpk_feast_sse42_zfeast_hcsrev(const char* uplo, const MKL_INT* n,
    const MKL_Complex16* sa, const MKL_INT* isa, const MKL_INT* jsa,
    MKL_INT* fpm, double* epsout, MKL_INT* loop, const double* emin,
    const double* emax, MKL_INT* m0, double* e, MKL_Complex16* x, MKL_INT* m,
    double* res, MKL_INT* info);
void fpk_feast_sse42_zfeast_hcsrgv(const char* uplo, const MKL_INT* n,
    const MKL_Complex16* sa, const MKL_INT* isa, const MKL_INT* jsa,
    const MKL_Complex16* sb, const MKL_INT* isb, const MKL_INT* jsb,
    MKL_INT* fpm, double* epsout, MKL_INT* loop, const double* emin,
    const double* emax, MKL_INT* m0, double* e, MKL_Complex16* x, MKL_INT* m,
    double* res, MKL_INT* info);
void fpk_feast_sse42_zfeast_heev(const char* uplo, const MKL_INT* n,
    const MKL_Complex16* a, const MKL_INT* lda, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* e, MKL_Complex16* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_sse42_zfeast_hegv(const char* uplo, const MKL_INT* n,
    const MKL_Complex16* a, const MKL_INT* lda, const MKL_Complex16* b,
    const MKL_INT* ldb, MKL_INT* fpm, double* epsout, MKL_INT* loop,
    const double* emin, const double* emax, MKL_INT* m0, double* e,
    MKL_Complex16* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_sse42_zfeast_hrci(MKL_INT* ijob, const MKL_INT* n,
    MKL_Complex16* ze, MKL_Complex16* work, MKL_Complex16* workc,
    MKL_Complex16* zaq, MKL_Complex16* zsq, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* lambda, MKL_Complex16* q, MKL_INT* m, double* res, MKL_INT* info);

void fpk_feast_avx_cfeast_hbev(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const MKL_Complex8* a, const MKL_INT* lda, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, MKL_Complex8* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_avx_cfeast_hbgv(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const MKL_Complex8* a, const MKL_INT* lda,
    const MKL_INT* klb, const MKL_Complex8* b, const MKL_INT* ldb, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, MKL_Complex8* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_avx_cfeast_hcsrev(const char* uplo, const MKL_INT* n,
    const MKL_Complex8* sa, const MKL_INT* isa, const MKL_INT* jsa, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, MKL_Complex8* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_avx_cfeast_hcsrgv(const char* uplo, const MKL_INT* n,
    const MKL_Complex8* sa, const MKL_INT* isa, const MKL_INT* jsa,
    const MKL_Complex8* sb, const MKL_INT* isb, const MKL_INT* jsb, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, MKL_Complex8* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_avx_cfeast_heev(const char* uplo, const MKL_INT* n,
    const MKL_Complex8* a, const MKL_INT* lda, MKL_INT* fpm, float* epsout,
    MKL_INT* loop, const float* emin, const float* emax, MKL_INT* m0, float* e,
    MKL_Complex8* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx_cfeast_hegv(const char* uplo, const MKL_INT* n,
    const MKL_Complex8* a, const MKL_INT* lda, const MKL_Complex8* b,
    const MKL_INT* ldb, MKL_INT* fpm, float* epsout, MKL_INT* loop,
    const float* emin, const float* emax, MKL_INT* m0, float* e, MKL_Complex8* x,
    MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx_cfeast_hrci(MKL_INT* ijob, const MKL_INT* n, MKL_Complex8* ze,
    MKL_Complex8* work, MKL_Complex8* workc, MKL_Complex8* zaq,
    MKL_Complex8* zsq, MKL_INT* fpm, float* epsout, MKL_INT* loop,
    const float* emin, const float* emax, MKL_INT* m0, float* lambda,
    MKL_Complex8* q, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx_dfeast_sbev(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const double* a, const MKL_INT* lda, MKL_INT* fpm,
    double* epsout, MKL_INT* loop, const double* emin, const double* emax,
    MKL_INT* m0, double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_avx_dfeast_sbgv(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const double* a, const MKL_INT* lda, const MKL_INT* klb,
    const double* b, const MKL_INT* ldb, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_avx_dfeast_scsrev(const char* uplo, const MKL_INT* n,
    const double* sa, const MKL_INT* isa, const MKL_INT* jsa, MKL_INT* fpm,
    double* epsout, MKL_INT* loop, const double* emin, const double* emax,
    MKL_INT* m0, double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_avx_dfeast_scsrgv(const char* uplo, const MKL_INT* n,
    const double* sa, const MKL_INT* isa, const MKL_INT* jsa, const double* sb,
    const MKL_INT* isb, const MKL_INT* jsb, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_avx_dfeast_srci(MKL_INT* ijob, const MKL_INT* n,
    MKL_Complex16* ze, double* work, MKL_Complex16* workc, double* aq,
    double* sq, MKL_INT* fpm, double* epsout, MKL_INT* loop, const double* emin,
    const double* emax, MKL_INT* m0, double* lambda, double* q, MKL_INT* m,
    double* res, MKL_INT* info);
void fpk_feast_avx_dfeast_syev(const char* uplo, const MKL_INT* n,
    const double* a, const MKL_INT* lda, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_avx_dfeast_sygv(const char* uplo, const MKL_INT* n,
    const double* a, const MKL_INT* lda, const double* b, const MKL_INT* ldb,
    MKL_INT* fpm, double* epsout, MKL_INT* loop, const double* emin,
    const double* emax, MKL_INT* m0, double* e, double* x, MKL_INT* m,
    double* res, MKL_INT* info);
void fpk_feast_avx_feastinit(MKL_INT* fpm);
void fpk_feast_avx_sfeast_sbev(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const float* a, const MKL_INT* lda, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx_sfeast_sbgv(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const float* a, const MKL_INT* lda, const MKL_INT* klb,
    const float* b, const MKL_INT* ldb, MKL_INT* fpm, float* epsout,
    MKL_INT* loop, const float* emin, const float* emax, MKL_INT* m0, float* e,
    float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx_sfeast_scsrev(const char* uplo, const MKL_INT* n,
    const float* sa, const MKL_INT* isa, const MKL_INT* jsa, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx_sfeast_scsrgv(const char* uplo, const MKL_INT* n,
    const float* sa, const MKL_INT* isa, const MKL_INT* jsa, const float* sb,
    const MKL_INT* isb, const MKL_INT* jsb, MKL_INT* fpm, float* epsout,
    MKL_INT* loop, const float* emin, const float* emax, MKL_INT* m0, float* e,
    float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx_sfeast_srci(MKL_INT* ijob, const MKL_INT* n, MKL_Complex8* ze,
    float* work, MKL_Complex8* workc, float* aq, float* sq, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* lambda, float* q, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx_sfeast_syev(const char* uplo, const MKL_INT* n,
    const float* a, const MKL_INT* lda, MKL_INT* fpm, float* epsout,
    MKL_INT* loop, const float* emin, const float* emax, MKL_INT* m0, float* e,
    float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx_sfeast_sygv(const char* uplo, const MKL_INT* n,
    const float* a, const MKL_INT* lda, const float* b, const MKL_INT* ldb,
    MKL_INT* fpm, float* epsout, MKL_INT* loop, const float* emin,
    const float* emax, MKL_INT* m0, float* e, float* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_avx_zfeast_hbev(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const MKL_Complex16* a, const MKL_INT* lda, MKL_INT* fpm,
    double* epsout, MKL_INT* loop, const double* emin, const double* emax,
    MKL_INT* m0, double* e, MKL_Complex16* x, MKL_INT* m, double* res,
    MKL_INT* info);
void fpk_feast_avx_zfeast_hbgv(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const MKL_Complex16* a, const MKL_INT* lda,
    const MKL_INT* klb, const MKL_Complex16* b, const MKL_INT* ldb, MKL_INT* fpm,
    double* epsout, MKL_INT* loop, const double* emin, const double* emax,
    MKL_INT* m0, double* e, MKL_Complex16* x, MKL_INT* m, double* res,
    MKL_INT* info);
void fpk_feast_avx_zfeast_hcsrev(const char* uplo, const MKL_INT* n,
    const MKL_Complex16* sa, const MKL_INT* isa, const MKL_INT* jsa,
    MKL_INT* fpm, double* epsout, MKL_INT* loop, const double* emin,
    const double* emax, MKL_INT* m0, double* e, MKL_Complex16* x, MKL_INT* m,
    double* res, MKL_INT* info);
void fpk_feast_avx_zfeast_hcsrgv(const char* uplo, const MKL_INT* n,
    const MKL_Complex16* sa, const MKL_INT* isa, const MKL_INT* jsa,
    const MKL_Complex16* sb, const MKL_INT* isb, const MKL_INT* jsb,
    MKL_INT* fpm, double* epsout, MKL_INT* loop, const double* emin,
    const double* emax, MKL_INT* m0, double* e, MKL_Complex16* x, MKL_INT* m,
    double* res, MKL_INT* info);
void fpk_feast_avx_zfeast_heev(const char* uplo, const MKL_INT* n,
    const MKL_Complex16* a, const MKL_INT* lda, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* e, MKL_Complex16* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_avx_zfeast_hegv(const char* uplo, const MKL_INT* n,
    const MKL_Complex16* a, const MKL_INT* lda, const MKL_Complex16* b,
    const MKL_INT* ldb, MKL_INT* fpm, double* epsout, MKL_INT* loop,
    const double* emin, const double* emax, MKL_INT* m0, double* e,
    MKL_Complex16* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_avx_zfeast_hrci(MKL_INT* ijob, const MKL_INT* n,
    MKL_Complex16* ze, MKL_Complex16* work, MKL_Complex16* workc,
    MKL_Complex16* zaq, MKL_Complex16* zsq, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* lambda, MKL_Complex16* q, MKL_INT* m, double* res, MKL_INT* info);

void fpk_feast_avx2_cfeast_hbev(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const MKL_Complex8* a, const MKL_INT* lda, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, MKL_Complex8* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_avx2_cfeast_hbgv(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const MKL_Complex8* a, const MKL_INT* lda,
    const MKL_INT* klb, const MKL_Complex8* b, const MKL_INT* ldb, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, MKL_Complex8* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_avx2_cfeast_hcsrev(const char* uplo, const MKL_INT* n,
    const MKL_Complex8* sa, const MKL_INT* isa, const MKL_INT* jsa, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, MKL_Complex8* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_avx2_cfeast_hcsrgv(const char* uplo, const MKL_INT* n,
    const MKL_Complex8* sa, const MKL_INT* isa, const MKL_INT* jsa,
    const MKL_Complex8* sb, const MKL_INT* isb, const MKL_INT* jsb, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, MKL_Complex8* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_avx2_cfeast_heev(const char* uplo, const MKL_INT* n,
    const MKL_Complex8* a, const MKL_INT* lda, MKL_INT* fpm, float* epsout,
    MKL_INT* loop, const float* emin, const float* emax, MKL_INT* m0, float* e,
    MKL_Complex8* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx2_cfeast_hegv(const char* uplo, const MKL_INT* n,
    const MKL_Complex8* a, const MKL_INT* lda, const MKL_Complex8* b,
    const MKL_INT* ldb, MKL_INT* fpm, float* epsout, MKL_INT* loop,
    const float* emin, const float* emax, MKL_INT* m0, float* e, MKL_Complex8* x,
    MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx2_cfeast_hrci(MKL_INT* ijob, const MKL_INT* n,
    MKL_Complex8* ze, MKL_Complex8* work, MKL_Complex8* workc, MKL_Complex8* zaq,
    MKL_Complex8* zsq, MKL_INT* fpm, float* epsout, MKL_INT* loop,
    const float* emin, const float* emax, MKL_INT* m0, float* lambda,
    MKL_Complex8* q, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx2_dfeast_sbev(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const double* a, const MKL_INT* lda, MKL_INT* fpm,
    double* epsout, MKL_INT* loop, const double* emin, const double* emax,
    MKL_INT* m0, double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_avx2_dfeast_sbgv(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const double* a, const MKL_INT* lda, const MKL_INT* klb,
    const double* b, const MKL_INT* ldb, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_avx2_dfeast_scsrev(const char* uplo, const MKL_INT* n,
    const double* sa, const MKL_INT* isa, const MKL_INT* jsa, MKL_INT* fpm,
    double* epsout, MKL_INT* loop, const double* emin, const double* emax,
    MKL_INT* m0, double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_avx2_dfeast_scsrgv(const char* uplo, const MKL_INT* n,
    const double* sa, const MKL_INT* isa, const MKL_INT* jsa, const double* sb,
    const MKL_INT* isb, const MKL_INT* jsb, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_avx2_dfeast_srci(MKL_INT* ijob, const MKL_INT* n,
    MKL_Complex16* ze, double* work, MKL_Complex16* workc, double* aq,
    double* sq, MKL_INT* fpm, double* epsout, MKL_INT* loop, const double* emin,
    const double* emax, MKL_INT* m0, double* lambda, double* q, MKL_INT* m,
    double* res, MKL_INT* info);
void fpk_feast_avx2_dfeast_syev(const char* uplo, const MKL_INT* n,
    const double* a, const MKL_INT* lda, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_avx2_dfeast_sygv(const char* uplo, const MKL_INT* n,
    const double* a, const MKL_INT* lda, const double* b, const MKL_INT* ldb,
    MKL_INT* fpm, double* epsout, MKL_INT* loop, const double* emin,
    const double* emax, MKL_INT* m0, double* e, double* x, MKL_INT* m,
    double* res, MKL_INT* info);
void fpk_feast_avx2_feastinit(MKL_INT* fpm);
void fpk_feast_avx2_sfeast_sbev(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const float* a, const MKL_INT* lda, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx2_sfeast_sbgv(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const float* a, const MKL_INT* lda, const MKL_INT* klb,
    const float* b, const MKL_INT* ldb, MKL_INT* fpm, float* epsout,
    MKL_INT* loop, const float* emin, const float* emax, MKL_INT* m0, float* e,
    float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx2_sfeast_scsrev(const char* uplo, const MKL_INT* n,
    const float* sa, const MKL_INT* isa, const MKL_INT* jsa, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx2_sfeast_scsrgv(const char* uplo, const MKL_INT* n,
    const float* sa, const MKL_INT* isa, const MKL_INT* jsa, const float* sb,
    const MKL_INT* isb, const MKL_INT* jsb, MKL_INT* fpm, float* epsout,
    MKL_INT* loop, const float* emin, const float* emax, MKL_INT* m0, float* e,
    float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx2_sfeast_srci(MKL_INT* ijob, const MKL_INT* n,
    MKL_Complex8* ze, float* work, MKL_Complex8* workc, float* aq, float* sq,
    MKL_INT* fpm, float* epsout, MKL_INT* loop, const float* emin,
    const float* emax, MKL_INT* m0, float* lambda, float* q, MKL_INT* m,
    float* res, MKL_INT* info);
void fpk_feast_avx2_sfeast_syev(const char* uplo, const MKL_INT* n,
    const float* a, const MKL_INT* lda, MKL_INT* fpm, float* epsout,
    MKL_INT* loop, const float* emin, const float* emax, MKL_INT* m0, float* e,
    float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx2_sfeast_sygv(const char* uplo, const MKL_INT* n,
    const float* a, const MKL_INT* lda, const float* b, const MKL_INT* ldb,
    MKL_INT* fpm, float* epsout, MKL_INT* loop, const float* emin,
    const float* emax, MKL_INT* m0, float* e, float* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_avx2_zfeast_hbev(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const MKL_Complex16* a, const MKL_INT* lda, MKL_INT* fpm,
    double* epsout, MKL_INT* loop, const double* emin, const double* emax,
    MKL_INT* m0, double* e, MKL_Complex16* x, MKL_INT* m, double* res,
    MKL_INT* info);
void fpk_feast_avx2_zfeast_hbgv(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const MKL_Complex16* a, const MKL_INT* lda,
    const MKL_INT* klb, const MKL_Complex16* b, const MKL_INT* ldb, MKL_INT* fpm,
    double* epsout, MKL_INT* loop, const double* emin, const double* emax,
    MKL_INT* m0, double* e, MKL_Complex16* x, MKL_INT* m, double* res,
    MKL_INT* info);
void fpk_feast_avx2_zfeast_hcsrev(const char* uplo, const MKL_INT* n,
    const MKL_Complex16* sa, const MKL_INT* isa, const MKL_INT* jsa,
    MKL_INT* fpm, double* epsout, MKL_INT* loop, const double* emin,
    const double* emax, MKL_INT* m0, double* e, MKL_Complex16* x, MKL_INT* m,
    double* res, MKL_INT* info);
void fpk_feast_avx2_zfeast_hcsrgv(const char* uplo, const MKL_INT* n,
    const MKL_Complex16* sa, const MKL_INT* isa, const MKL_INT* jsa,
    const MKL_Complex16* sb, const MKL_INT* isb, const MKL_INT* jsb,
    MKL_INT* fpm, double* epsout, MKL_INT* loop, const double* emin,
    const double* emax, MKL_INT* m0, double* e, MKL_Complex16* x, MKL_INT* m,
    double* res, MKL_INT* info);
void fpk_feast_avx2_zfeast_heev(const char* uplo, const MKL_INT* n,
    const MKL_Complex16* a, const MKL_INT* lda, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* e, MKL_Complex16* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_avx2_zfeast_hegv(const char* uplo, const MKL_INT* n,
    const MKL_Complex16* a, const MKL_INT* lda, const MKL_Complex16* b,
    const MKL_INT* ldb, MKL_INT* fpm, double* epsout, MKL_INT* loop,
    const double* emin, const double* emax, MKL_INT* m0, double* e,
    MKL_Complex16* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_avx2_zfeast_hrci(MKL_INT* ijob, const MKL_INT* n,
    MKL_Complex16* ze, MKL_Complex16* work, MKL_Complex16* workc,
    MKL_Complex16* zaq, MKL_Complex16* zsq, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* lambda, MKL_Complex16* q, MKL_INT* m, double* res, MKL_INT* info);

void fpk_feast_avx512_cfeast_hbev(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const MKL_Complex8* a, const MKL_INT* lda, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, MKL_Complex8* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_avx512_cfeast_hbgv(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const MKL_Complex8* a, const MKL_INT* lda,
    const MKL_INT* klb, const MKL_Complex8* b, const MKL_INT* ldb, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, MKL_Complex8* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_avx512_cfeast_hcsrev(const char* uplo, const MKL_INT* n,
    const MKL_Complex8* sa, const MKL_INT* isa, const MKL_INT* jsa, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, MKL_Complex8* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_avx512_cfeast_hcsrgv(const char* uplo, const MKL_INT* n,
    const MKL_Complex8* sa, const MKL_INT* isa, const MKL_INT* jsa,
    const MKL_Complex8* sb, const MKL_INT* isb, const MKL_INT* jsb, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, MKL_Complex8* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_avx512_cfeast_heev(const char* uplo, const MKL_INT* n,
    const MKL_Complex8* a, const MKL_INT* lda, MKL_INT* fpm, float* epsout,
    MKL_INT* loop, const float* emin, const float* emax, MKL_INT* m0, float* e,
    MKL_Complex8* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx512_cfeast_hegv(const char* uplo, const MKL_INT* n,
    const MKL_Complex8* a, const MKL_INT* lda, const MKL_Complex8* b,
    const MKL_INT* ldb, MKL_INT* fpm, float* epsout, MKL_INT* loop,
    const float* emin, const float* emax, MKL_INT* m0, float* e, MKL_Complex8* x,
    MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx512_cfeast_hrci(MKL_INT* ijob, const MKL_INT* n,
    MKL_Complex8* ze, MKL_Complex8* work, MKL_Complex8* workc, MKL_Complex8* zaq,
    MKL_Complex8* zsq, MKL_INT* fpm, float* epsout, MKL_INT* loop,
    const float* emin, const float* emax, MKL_INT* m0, float* lambda,
    MKL_Complex8* q, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx512_dfeast_sbev(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const double* a, const MKL_INT* lda, MKL_INT* fpm,
    double* epsout, MKL_INT* loop, const double* emin, const double* emax,
    MKL_INT* m0, double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_avx512_dfeast_sbgv(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const double* a, const MKL_INT* lda, const MKL_INT* klb,
    const double* b, const MKL_INT* ldb, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_avx512_dfeast_scsrev(const char* uplo, const MKL_INT* n,
    const double* sa, const MKL_INT* isa, const MKL_INT* jsa, MKL_INT* fpm,
    double* epsout, MKL_INT* loop, const double* emin, const double* emax,
    MKL_INT* m0, double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_avx512_dfeast_scsrgv(const char* uplo, const MKL_INT* n,
    const double* sa, const MKL_INT* isa, const MKL_INT* jsa, const double* sb,
    const MKL_INT* isb, const MKL_INT* jsb, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_avx512_dfeast_srci(MKL_INT* ijob, const MKL_INT* n,
    MKL_Complex16* ze, double* work, MKL_Complex16* workc, double* aq,
    double* sq, MKL_INT* fpm, double* epsout, MKL_INT* loop, const double* emin,
    const double* emax, MKL_INT* m0, double* lambda, double* q, MKL_INT* m,
    double* res, MKL_INT* info);
void fpk_feast_avx512_dfeast_syev(const char* uplo, const MKL_INT* n,
    const double* a, const MKL_INT* lda, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* e, double* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_avx512_dfeast_sygv(const char* uplo, const MKL_INT* n,
    const double* a, const MKL_INT* lda, const double* b, const MKL_INT* ldb,
    MKL_INT* fpm, double* epsout, MKL_INT* loop, const double* emin,
    const double* emax, MKL_INT* m0, double* e, double* x, MKL_INT* m,
    double* res, MKL_INT* info);
void fpk_feast_avx512_feastinit(MKL_INT* fpm);
void fpk_feast_avx512_sfeast_sbev(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const float* a, const MKL_INT* lda, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx512_sfeast_sbgv(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const float* a, const MKL_INT* lda, const MKL_INT* klb,
    const float* b, const MKL_INT* ldb, MKL_INT* fpm, float* epsout,
    MKL_INT* loop, const float* emin, const float* emax, MKL_INT* m0, float* e,
    float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx512_sfeast_scsrev(const char* uplo, const MKL_INT* n,
    const float* sa, const MKL_INT* isa, const MKL_INT* jsa, MKL_INT* fpm,
    float* epsout, MKL_INT* loop, const float* emin, const float* emax,
    MKL_INT* m0, float* e, float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx512_sfeast_scsrgv(const char* uplo, const MKL_INT* n,
    const float* sa, const MKL_INT* isa, const MKL_INT* jsa, const float* sb,
    const MKL_INT* isb, const MKL_INT* jsb, MKL_INT* fpm, float* epsout,
    MKL_INT* loop, const float* emin, const float* emax, MKL_INT* m0, float* e,
    float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx512_sfeast_srci(MKL_INT* ijob, const MKL_INT* n,
    MKL_Complex8* ze, float* work, MKL_Complex8* workc, float* aq, float* sq,
    MKL_INT* fpm, float* epsout, MKL_INT* loop, const float* emin,
    const float* emax, MKL_INT* m0, float* lambda, float* q, MKL_INT* m,
    float* res, MKL_INT* info);
void fpk_feast_avx512_sfeast_syev(const char* uplo, const MKL_INT* n,
    const float* a, const MKL_INT* lda, MKL_INT* fpm, float* epsout,
    MKL_INT* loop, const float* emin, const float* emax, MKL_INT* m0, float* e,
    float* x, MKL_INT* m, float* res, MKL_INT* info);
void fpk_feast_avx512_sfeast_sygv(const char* uplo, const MKL_INT* n,
    const float* a, const MKL_INT* lda, const float* b, const MKL_INT* ldb,
    MKL_INT* fpm, float* epsout, MKL_INT* loop, const float* emin,
    const float* emax, MKL_INT* m0, float* e, float* x, MKL_INT* m, float* res,
    MKL_INT* info);
void fpk_feast_avx512_zfeast_hbev(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const MKL_Complex16* a, const MKL_INT* lda, MKL_INT* fpm,
    double* epsout, MKL_INT* loop, const double* emin, const double* emax,
    MKL_INT* m0, double* e, MKL_Complex16* x, MKL_INT* m, double* res,
    MKL_INT* info);
void fpk_feast_avx512_zfeast_hbgv(const char* uplo, const MKL_INT* n,
    const MKL_INT* kla, const MKL_Complex16* a, const MKL_INT* lda,
    const MKL_INT* klb, const MKL_Complex16* b, const MKL_INT* ldb, MKL_INT* fpm,
    double* epsout, MKL_INT* loop, const double* emin, const double* emax,
    MKL_INT* m0, double* e, MKL_Complex16* x, MKL_INT* m, double* res,
    MKL_INT* info);
void fpk_feast_avx512_zfeast_hcsrev(const char* uplo, const MKL_INT* n,
    const MKL_Complex16* sa, const MKL_INT* isa, const MKL_INT* jsa,
    MKL_INT* fpm, double* epsout, MKL_INT* loop, const double* emin,
    const double* emax, MKL_INT* m0, double* e, MKL_Complex16* x, MKL_INT* m,
    double* res, MKL_INT* info);
void fpk_feast_avx512_zfeast_hcsrgv(const char* uplo, const MKL_INT* n,
    const MKL_Complex16* sa, const MKL_INT* isa, const MKL_INT* jsa,
    const MKL_Complex16* sb, const MKL_INT* isb, const MKL_INT* jsb,
    MKL_INT* fpm, double* epsout, MKL_INT* loop, const double* emin,
    const double* emax, MKL_INT* m0, double* e, MKL_Complex16* x, MKL_INT* m,
    double* res, MKL_INT* info);
void fpk_feast_avx512_zfeast_heev(const char* uplo, const MKL_INT* n,
    const MKL_Complex16* a, const MKL_INT* lda, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* e, MKL_Complex16* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_avx512_zfeast_hegv(const char* uplo, const MKL_INT* n,
    const MKL_Complex16* a, const MKL_INT* lda, const MKL_Complex16* b,
    const MKL_INT* ldb, MKL_INT* fpm, double* epsout, MKL_INT* loop,
    const double* emin, const double* emax, MKL_INT* m0, double* e,
    MKL_Complex16* x, MKL_INT* m, double* res, MKL_INT* info);
void fpk_feast_avx512_zfeast_hrci(MKL_INT* ijob, const MKL_INT* n,
    MKL_Complex16* ze, MKL_Complex16* work, MKL_Complex16* workc,
    MKL_Complex16* zaq, MKL_Complex16* zsq, MKL_INT* fpm, double* epsout,
    MKL_INT* loop, const double* emin, const double* emax, MKL_INT* m0,
    double* lambda, MKL_Complex16* q, MKL_INT* m, double* res, MKL_INT* info);


void fpk_lapack_ssse3_dgels(const char* trans, const MKL_INT* m,
    const MKL_INT* n, const MKL_INT* nrhs, double* a, const MKL_INT* lda,
    double* b, const MKL_INT* ldb, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_ssse3_dgeqp3(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* jpvt, double* tau, double* work,
    const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_ssse3_dgeqrf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, double* tau, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_ssse3_dgerqf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, double* tau, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_ssse3_dgesvd(const char* jobu, const char* jobvt,
    const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, double* s,
    double* u, const MKL_INT* ldu, double* vt, const MKL_INT* ldvt, double* work,
    const MKL_INT* lwork, MKL_INT* info );
double fpk_lapack_ssse3_dlange(const char* norm, const MKL_INT* m,
    const MKL_INT* n, const double* a, const MKL_INT* lda, double* work );
void fpk_lapack_ssse3_dlarnv(const MKL_INT* idist, MKL_INT* iseed,
    const MKL_INT* n, double* x );
void fpk_lapack_ssse3_dorgqr(const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* k, double* a, const MKL_INT* lda, const double* tau,
    double* work, const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_ssse3_dorgrq(const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* k, double* a, const MKL_INT* lda, const double* tau,
    double* work, const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_ssse3_dormrq(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const double* a,
    const MKL_INT* lda, const double* tau, double* c, const MKL_INT* ldc,
    double* work, const MKL_INT* lwork, MKL_INT* info, int, int );
void fpk_lapack_ssse3_dpftrf(const char* transr, const char* uplo,
    const MKL_INT* n, double* a, MKL_INT* info );
void fpk_lapack_ssse3_dpotrf(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* info );
void fpk_lapack_ssse3_dpotri(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* info );
void fpk_lapack_ssse3_dpotrs(const char* uplo, const MKL_INT* n,
    const MKL_INT* nrhs, const double* a, const MKL_INT* lda, double* b,
    const MKL_INT* ldb, MKL_INT* info );
void fpk_lapack_ssse3_dpptrf(const char* uplo, const MKL_INT* n, double* ap,
    MKL_INT* info );
void fpk_lapack_ssse3_dpstrf(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* piv, MKL_INT* rank, const double* tol,
    double* work, MKL_INT* info );
void fpk_lapack_ssse3_dspevd(const char* jobz, const char* uplo,
    const MKL_INT* n, double* ap, double* w, double* z, const MKL_INT* ldz,
    double* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork,
    MKL_INT* info );
void fpk_lapack_ssse3_dsyev(const char* jobz, const char* uplo, const MKL_INT* n,
    double* a, const MKL_INT* lda, double* w, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_ssse3_dsyevd(const char* jobz, const char* uplo,
    const MKL_INT* n, double* a, const MKL_INT* lda, double* w, double* work,
    const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info );
void fpk_lapack_ssse3_dsyevr(const char* jobz, const char* range,
    const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda,
    const double* vl, const double* vu, const MKL_INT* il, const MKL_INT* iu,
    const double* abstol, MKL_INT* m, double* w, double* z, const MKL_INT* ldz,
    MKL_INT* isuppz, double* work, const MKL_INT* lwork, MKL_INT* iwork,
    const MKL_INT* liwork, MKL_INT* info );
void fpk_lapack_ssse3_dtrtrs(const char* uplo, const char* trans,
    const char* diag, const MKL_INT* n, const MKL_INT* nrhs, const double* a,
    const MKL_INT* lda, double* b, const MKL_INT* ldb, MKL_INT* info );
void fpk_lapack_ssse3_sgels(const char* trans, const MKL_INT* m,
    const MKL_INT* n, const MKL_INT* nrhs, float* a, const MKL_INT* lda,
    float* b, const MKL_INT* ldb, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_ssse3_sgeqp3(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* jpvt, float* tau, float* work,
    const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_ssse3_sgeqrf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, float* tau, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_ssse3_sgerqf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, float* tau, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_ssse3_sgesvd(const char* jobu, const char* jobvt,
    const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda, float* s,
    float* u, const MKL_INT* ldu, float* vt, const MKL_INT* ldvt, float* work,
    const MKL_INT* lwork, MKL_INT* info );
float fpk_lapack_ssse3_slange(const char* norm, const MKL_INT* m,
    const MKL_INT* n, const float* a, const MKL_INT* lda, float* work );
void fpk_lapack_ssse3_slarnv(const MKL_INT* idist, MKL_INT* iseed,
    const MKL_INT* n, float* x );
void fpk_lapack_ssse3_sorgqr(const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* k, float* a, const MKL_INT* lda, const float* tau,
    float* work, const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_ssse3_sorgrq(const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* k, float* a, const MKL_INT* lda, const float* tau,
    float* work, const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_ssse3_sormrq(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const float* a,
    const MKL_INT* lda, const float* tau, float* c, const MKL_INT* ldc,
    float* work, const MKL_INT* lwork, MKL_INT* info, int, int );
void fpk_lapack_ssse3_spftrf(const char* transr, const char* uplo,
    const MKL_INT* n, float* a, MKL_INT* info );
void fpk_lapack_ssse3_spotrf(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* info );
void fpk_lapack_ssse3_spotri(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* info );
void fpk_lapack_ssse3_spotrs(const char* uplo, const MKL_INT* n,
    const MKL_INT* nrhs, const float* a, const MKL_INT* lda, float* b,
    const MKL_INT* ldb, MKL_INT* info );
void fpk_lapack_ssse3_spptrf(const char* uplo, const MKL_INT* n, float* ap,
    MKL_INT* info );
void fpk_lapack_ssse3_spstrf(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* piv, MKL_INT* rank, const float* tol,
    float* work, MKL_INT* info );
void fpk_lapack_ssse3_sspevd(const char* jobz, const char* uplo,
    const MKL_INT* n, float* ap, float* w, float* z, const MKL_INT* ldz,
    float* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork,
    MKL_INT* info );
void fpk_lapack_ssse3_ssyev(const char* jobz, const char* uplo, const MKL_INT* n,
    float* a, const MKL_INT* lda, float* w, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_ssse3_ssyevd(const char* jobz, const char* uplo,
    const MKL_INT* n, float* a, const MKL_INT* lda, float* w, float* work,
    const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info );
void fpk_lapack_ssse3_ssyevr(const char* jobz, const char* range,
    const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda,
    const float* vl, const float* vu, const MKL_INT* il, const MKL_INT* iu,
    const float* abstol, MKL_INT* m, float* w, float* z, const MKL_INT* ldz,
    MKL_INT* isuppz, float* work, const MKL_INT* lwork, MKL_INT* iwork,
    const MKL_INT* liwork, MKL_INT* info );
void fpk_lapack_ssse3_strtrs(const char* uplo, const char* trans,
    const char* diag, const MKL_INT* n, const MKL_INT* nrhs, const float* a,
    const MKL_INT* lda, float* b, const MKL_INT* ldb, MKL_INT* info );

void fpk_lapack_sse42_dgels(const char* trans, const MKL_INT* m,
    const MKL_INT* n, const MKL_INT* nrhs, double* a, const MKL_INT* lda,
    double* b, const MKL_INT* ldb, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_sse42_dgeqp3(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* jpvt, double* tau, double* work,
    const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_sse42_dgeqrf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, double* tau, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_sse42_dgerqf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, double* tau, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_sse42_dgesvd(const char* jobu, const char* jobvt,
    const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, double* s,
    double* u, const MKL_INT* ldu, double* vt, const MKL_INT* ldvt, double* work,
    const MKL_INT* lwork, MKL_INT* info );
double fpk_lapack_sse42_dlange(const char* norm, const MKL_INT* m,
    const MKL_INT* n, const double* a, const MKL_INT* lda, double* work );
void fpk_lapack_sse42_dlarnv(const MKL_INT* idist, MKL_INT* iseed,
    const MKL_INT* n, double* x );
void fpk_lapack_sse42_dorgqr(const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* k, double* a, const MKL_INT* lda, const double* tau,
    double* work, const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_sse42_dorgrq(const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* k, double* a, const MKL_INT* lda, const double* tau,
    double* work, const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_sse42_dormrq(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const double* a,
    const MKL_INT* lda, const double* tau, double* c, const MKL_INT* ldc,
    double* work, const MKL_INT* lwork, MKL_INT* info, int, int );
void fpk_lapack_sse42_dpftrf(const char* transr, const char* uplo,
    const MKL_INT* n, double* a, MKL_INT* info );
void fpk_lapack_sse42_dpotrf(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* info );
void fpk_lapack_sse42_dpotri(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* info );
void fpk_lapack_sse42_dpotrs(const char* uplo, const MKL_INT* n,
    const MKL_INT* nrhs, const double* a, const MKL_INT* lda, double* b,
    const MKL_INT* ldb, MKL_INT* info );
void fpk_lapack_sse42_dpptrf(const char* uplo, const MKL_INT* n, double* ap,
    MKL_INT* info );
void fpk_lapack_sse42_dpstrf(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* piv, MKL_INT* rank, const double* tol,
    double* work, MKL_INT* info );
void fpk_lapack_sse42_dspevd(const char* jobz, const char* uplo,
    const MKL_INT* n, double* ap, double* w, double* z, const MKL_INT* ldz,
    double* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork,
    MKL_INT* info );
void fpk_lapack_sse42_dsyev(const char* jobz, const char* uplo, const MKL_INT* n,
    double* a, const MKL_INT* lda, double* w, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_sse42_dsyevd(const char* jobz, const char* uplo,
    const MKL_INT* n, double* a, const MKL_INT* lda, double* w, double* work,
    const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info );
void fpk_lapack_sse42_dsyevr(const char* jobz, const char* range,
    const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda,
    const double* vl, const double* vu, const MKL_INT* il, const MKL_INT* iu,
    const double* abstol, MKL_INT* m, double* w, double* z, const MKL_INT* ldz,
    MKL_INT* isuppz, double* work, const MKL_INT* lwork, MKL_INT* iwork,
    const MKL_INT* liwork, MKL_INT* info );
void fpk_lapack_sse42_dtrtrs(const char* uplo, const char* trans,
    const char* diag, const MKL_INT* n, const MKL_INT* nrhs, const double* a,
    const MKL_INT* lda, double* b, const MKL_INT* ldb, MKL_INT* info );
void fpk_lapack_sse42_sgels(const char* trans, const MKL_INT* m,
    const MKL_INT* n, const MKL_INT* nrhs, float* a, const MKL_INT* lda,
    float* b, const MKL_INT* ldb, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_sse42_sgeqp3(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* jpvt, float* tau, float* work,
    const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_sse42_sgeqrf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, float* tau, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_sse42_sgerqf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, float* tau, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_sse42_sgesvd(const char* jobu, const char* jobvt,
    const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda, float* s,
    float* u, const MKL_INT* ldu, float* vt, const MKL_INT* ldvt, float* work,
    const MKL_INT* lwork, MKL_INT* info );
float fpk_lapack_sse42_slange(const char* norm, const MKL_INT* m,
    const MKL_INT* n, const float* a, const MKL_INT* lda, float* work );
void fpk_lapack_sse42_slarnv(const MKL_INT* idist, MKL_INT* iseed,
    const MKL_INT* n, float* x );
void fpk_lapack_sse42_sorgqr(const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* k, float* a, const MKL_INT* lda, const float* tau,
    float* work, const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_sse42_sorgrq(const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* k, float* a, const MKL_INT* lda, const float* tau,
    float* work, const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_sse42_sormrq(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const float* a,
    const MKL_INT* lda, const float* tau, float* c, const MKL_INT* ldc,
    float* work, const MKL_INT* lwork, MKL_INT* info, int, int );
void fpk_lapack_sse42_spftrf(const char* transr, const char* uplo,
    const MKL_INT* n, float* a, MKL_INT* info );
void fpk_lapack_sse42_spotrf(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* info );
void fpk_lapack_sse42_spotri(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* info );
void fpk_lapack_sse42_spotrs(const char* uplo, const MKL_INT* n,
    const MKL_INT* nrhs, const float* a, const MKL_INT* lda, float* b,
    const MKL_INT* ldb, MKL_INT* info );
void fpk_lapack_sse42_spptrf(const char* uplo, const MKL_INT* n, float* ap,
    MKL_INT* info );
void fpk_lapack_sse42_spstrf(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* piv, MKL_INT* rank, const float* tol,
    float* work, MKL_INT* info );
void fpk_lapack_sse42_sspevd(const char* jobz, const char* uplo,
    const MKL_INT* n, float* ap, float* w, float* z, const MKL_INT* ldz,
    float* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork,
    MKL_INT* info );
void fpk_lapack_sse42_ssyev(const char* jobz, const char* uplo, const MKL_INT* n,
    float* a, const MKL_INT* lda, float* w, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_sse42_ssyevd(const char* jobz, const char* uplo,
    const MKL_INT* n, float* a, const MKL_INT* lda, float* w, float* work,
    const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info );
void fpk_lapack_sse42_ssyevr(const char* jobz, const char* range,
    const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda,
    const float* vl, const float* vu, const MKL_INT* il, const MKL_INT* iu,
    const float* abstol, MKL_INT* m, float* w, float* z, const MKL_INT* ldz,
    MKL_INT* isuppz, float* work, const MKL_INT* lwork, MKL_INT* iwork,
    const MKL_INT* liwork, MKL_INT* info );
void fpk_lapack_sse42_strtrs(const char* uplo, const char* trans,
    const char* diag, const MKL_INT* n, const MKL_INT* nrhs, const float* a,
    const MKL_INT* lda, float* b, const MKL_INT* ldb, MKL_INT* info );

void fpk_lapack_avx_dgels(const char* trans, const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* nrhs, double* a, const MKL_INT* lda, double* b,
    const MKL_INT* ldb, double* work, const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_avx_dgeqp3(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* jpvt, double* tau, double* work,
    const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_avx_dgeqrf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, double* tau, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_avx_dgerqf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, double* tau, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_avx_dgesvd(const char* jobu, const char* jobvt, const MKL_INT* m,
    const MKL_INT* n, double* a, const MKL_INT* lda, double* s, double* u,
    const MKL_INT* ldu, double* vt, const MKL_INT* ldvt, double* work,
    const MKL_INT* lwork, MKL_INT* info );
double fpk_lapack_avx_dlange(const char* norm, const MKL_INT* m,
    const MKL_INT* n, const double* a, const MKL_INT* lda, double* work );
void fpk_lapack_avx_dlarnv(const MKL_INT* idist, MKL_INT* iseed,
    const MKL_INT* n, double* x );
void fpk_lapack_avx_dorgqr(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
    double* a, const MKL_INT* lda, const double* tau, double* work,
    const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_avx_dorgrq(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
    double* a, const MKL_INT* lda, const double* tau, double* work,
    const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_avx_dormrq(const char* side, const char* trans, const MKL_INT* m,
    const MKL_INT* n, const MKL_INT* k, const double* a, const MKL_INT* lda,
    const double* tau, double* c, const MKL_INT* ldc, double* work,
    const MKL_INT* lwork, MKL_INT* info, int, int );
void fpk_lapack_avx_dpftrf(const char* transr, const char* uplo,
    const MKL_INT* n, double* a, MKL_INT* info );
void fpk_lapack_avx_dpotrf(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* info );
void fpk_lapack_avx_dpotri(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* info );
void fpk_lapack_avx_dpotrs(const char* uplo, const MKL_INT* n,
    const MKL_INT* nrhs, const double* a, const MKL_INT* lda, double* b,
    const MKL_INT* ldb, MKL_INT* info );
void fpk_lapack_avx_dpptrf(const char* uplo, const MKL_INT* n, double* ap,
    MKL_INT* info );
void fpk_lapack_avx_dpstrf(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* piv, MKL_INT* rank, const double* tol,
    double* work, MKL_INT* info );
void fpk_lapack_avx_dspevd(const char* jobz, const char* uplo, const MKL_INT* n,
    double* ap, double* w, double* z, const MKL_INT* ldz, double* work,
    const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info );
void fpk_lapack_avx_dsyev(const char* jobz, const char* uplo, const MKL_INT* n,
    double* a, const MKL_INT* lda, double* w, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_avx_dsyevd(const char* jobz, const char* uplo, const MKL_INT* n,
    double* a, const MKL_INT* lda, double* w, double* work, const MKL_INT* lwork,
    MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info );
void fpk_lapack_avx_dsyevr(const char* jobz, const char* range, const char* uplo,
    const MKL_INT* n, double* a, const MKL_INT* lda, const double* vl,
    const double* vu, const MKL_INT* il, const MKL_INT* iu, const double* abstol,
    MKL_INT* m, double* w, double* z, const MKL_INT* ldz, MKL_INT* isuppz,
    double* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork,
    MKL_INT* info );
void fpk_lapack_avx_dtrtrs(const char* uplo, const char* trans, const char* diag,
    const MKL_INT* n, const MKL_INT* nrhs, const double* a, const MKL_INT* lda,
    double* b, const MKL_INT* ldb, MKL_INT* info );
void fpk_lapack_avx_sgels(const char* trans, const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* nrhs, float* a, const MKL_INT* lda, float* b,
    const MKL_INT* ldb, float* work, const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_avx_sgeqp3(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* jpvt, float* tau, float* work,
    const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_avx_sgeqrf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, float* tau, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_avx_sgerqf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, float* tau, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_avx_sgesvd(const char* jobu, const char* jobvt, const MKL_INT* m,
    const MKL_INT* n, float* a, const MKL_INT* lda, float* s, float* u,
    const MKL_INT* ldu, float* vt, const MKL_INT* ldvt, float* work,
    const MKL_INT* lwork, MKL_INT* info );
float fpk_lapack_avx_slange(const char* norm, const MKL_INT* m, const MKL_INT* n,
    const float* a, const MKL_INT* lda, float* work );
void fpk_lapack_avx_slarnv(const MKL_INT* idist, MKL_INT* iseed,
    const MKL_INT* n, float* x );
void fpk_lapack_avx_sorgqr(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
    float* a, const MKL_INT* lda, const float* tau, float* work,
    const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_avx_sorgrq(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
    float* a, const MKL_INT* lda, const float* tau, float* work,
    const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_avx_sormrq(const char* side, const char* trans, const MKL_INT* m,
    const MKL_INT* n, const MKL_INT* k, const float* a, const MKL_INT* lda,
    const float* tau, float* c, const MKL_INT* ldc, float* work,
    const MKL_INT* lwork, MKL_INT* info, int, int );
void fpk_lapack_avx_spftrf(const char* transr, const char* uplo,
    const MKL_INT* n, float* a, MKL_INT* info );
void fpk_lapack_avx_spotrf(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* info );
void fpk_lapack_avx_spotri(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* info );
void fpk_lapack_avx_spotrs(const char* uplo, const MKL_INT* n,
    const MKL_INT* nrhs, const float* a, const MKL_INT* lda, float* b,
    const MKL_INT* ldb, MKL_INT* info );
void fpk_lapack_avx_spptrf(const char* uplo, const MKL_INT* n, float* ap,
    MKL_INT* info );
void fpk_lapack_avx_spstrf(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* piv, MKL_INT* rank, const float* tol,
    float* work, MKL_INT* info );
void fpk_lapack_avx_sspevd(const char* jobz, const char* uplo, const MKL_INT* n,
    float* ap, float* w, float* z, const MKL_INT* ldz, float* work,
    const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info );
void fpk_lapack_avx_ssyev(const char* jobz, const char* uplo, const MKL_INT* n,
    float* a, const MKL_INT* lda, float* w, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_avx_ssyevd(const char* jobz, const char* uplo, const MKL_INT* n,
    float* a, const MKL_INT* lda, float* w, float* work, const MKL_INT* lwork,
    MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info );
void fpk_lapack_avx_ssyevr(const char* jobz, const char* range, const char* uplo,
    const MKL_INT* n, float* a, const MKL_INT* lda, const float* vl,
    const float* vu, const MKL_INT* il, const MKL_INT* iu, const float* abstol,
    MKL_INT* m, float* w, float* z, const MKL_INT* ldz, MKL_INT* isuppz,
    float* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork,
    MKL_INT* info );
void fpk_lapack_avx_strtrs(const char* uplo, const char* trans, const char* diag,
    const MKL_INT* n, const MKL_INT* nrhs, const float* a, const MKL_INT* lda,
    float* b, const MKL_INT* ldb, MKL_INT* info );

void fpk_lapack_avx2_dgels(const char* trans, const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* nrhs, double* a, const MKL_INT* lda, double* b,
    const MKL_INT* ldb, double* work, const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_avx2_dgeqp3(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* jpvt, double* tau, double* work,
    const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_avx2_dgeqrf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, double* tau, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_avx2_dgerqf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, double* tau, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_avx2_dgesvd(const char* jobu, const char* jobvt,
    const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, double* s,
    double* u, const MKL_INT* ldu, double* vt, const MKL_INT* ldvt, double* work,
    const MKL_INT* lwork, MKL_INT* info );
double fpk_lapack_avx2_dlange(const char* norm, const MKL_INT* m,
    const MKL_INT* n, const double* a, const MKL_INT* lda, double* work );
void fpk_lapack_avx2_dlarnv(const MKL_INT* idist, MKL_INT* iseed,
    const MKL_INT* n, double* x );
void fpk_lapack_avx2_dorgqr(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
    double* a, const MKL_INT* lda, const double* tau, double* work,
    const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_avx2_dorgrq(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
    double* a, const MKL_INT* lda, const double* tau, double* work,
    const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_avx2_dormrq(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const double* a,
    const MKL_INT* lda, const double* tau, double* c, const MKL_INT* ldc,
    double* work, const MKL_INT* lwork, MKL_INT* info, int, int );
void fpk_lapack_avx2_dpftrf(const char* transr, const char* uplo,
    const MKL_INT* n, double* a, MKL_INT* info );
void fpk_lapack_avx2_dpotrf(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* info );
void fpk_lapack_avx2_dpotri(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* info );
void fpk_lapack_avx2_dpotrs(const char* uplo, const MKL_INT* n,
    const MKL_INT* nrhs, const double* a, const MKL_INT* lda, double* b,
    const MKL_INT* ldb, MKL_INT* info );
void fpk_lapack_avx2_dpptrf(const char* uplo, const MKL_INT* n, double* ap,
    MKL_INT* info );
void fpk_lapack_avx2_dpstrf(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* piv, MKL_INT* rank, const double* tol,
    double* work, MKL_INT* info );
void fpk_lapack_avx2_dspevd(const char* jobz, const char* uplo, const MKL_INT* n,
    double* ap, double* w, double* z, const MKL_INT* ldz, double* work,
    const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info );
void fpk_lapack_avx2_dsyev(const char* jobz, const char* uplo, const MKL_INT* n,
    double* a, const MKL_INT* lda, double* w, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_avx2_dsyevd(const char* jobz, const char* uplo, const MKL_INT* n,
    double* a, const MKL_INT* lda, double* w, double* work, const MKL_INT* lwork,
    MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info );
void fpk_lapack_avx2_dsyevr(const char* jobz, const char* range,
    const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda,
    const double* vl, const double* vu, const MKL_INT* il, const MKL_INT* iu,
    const double* abstol, MKL_INT* m, double* w, double* z, const MKL_INT* ldz,
    MKL_INT* isuppz, double* work, const MKL_INT* lwork, MKL_INT* iwork,
    const MKL_INT* liwork, MKL_INT* info );
void fpk_lapack_avx2_dtrtrs(const char* uplo, const char* trans,
    const char* diag, const MKL_INT* n, const MKL_INT* nrhs, const double* a,
    const MKL_INT* lda, double* b, const MKL_INT* ldb, MKL_INT* info );
void fpk_lapack_avx2_sgels(const char* trans, const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* nrhs, float* a, const MKL_INT* lda, float* b,
    const MKL_INT* ldb, float* work, const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_avx2_sgeqp3(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* jpvt, float* tau, float* work,
    const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_avx2_sgeqrf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, float* tau, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_avx2_sgerqf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, float* tau, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_avx2_sgesvd(const char* jobu, const char* jobvt,
    const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda, float* s,
    float* u, const MKL_INT* ldu, float* vt, const MKL_INT* ldvt, float* work,
    const MKL_INT* lwork, MKL_INT* info );
float fpk_lapack_avx2_slange(const char* norm, const MKL_INT* m,
    const MKL_INT* n, const float* a, const MKL_INT* lda, float* work );
void fpk_lapack_avx2_slarnv(const MKL_INT* idist, MKL_INT* iseed,
    const MKL_INT* n, float* x );
void fpk_lapack_avx2_sorgqr(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
    float* a, const MKL_INT* lda, const float* tau, float* work,
    const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_avx2_sorgrq(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
    float* a, const MKL_INT* lda, const float* tau, float* work,
    const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_avx2_sormrq(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const float* a,
    const MKL_INT* lda, const float* tau, float* c, const MKL_INT* ldc,
    float* work, const MKL_INT* lwork, MKL_INT* info, int, int );
void fpk_lapack_avx2_spftrf(const char* transr, const char* uplo,
    const MKL_INT* n, float* a, MKL_INT* info );
void fpk_lapack_avx2_spotrf(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* info );
void fpk_lapack_avx2_spotri(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* info );
void fpk_lapack_avx2_spotrs(const char* uplo, const MKL_INT* n,
    const MKL_INT* nrhs, const float* a, const MKL_INT* lda, float* b,
    const MKL_INT* ldb, MKL_INT* info );
void fpk_lapack_avx2_spptrf(const char* uplo, const MKL_INT* n, float* ap,
    MKL_INT* info );
void fpk_lapack_avx2_spstrf(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* piv, MKL_INT* rank, const float* tol,
    float* work, MKL_INT* info );
void fpk_lapack_avx2_sspevd(const char* jobz, const char* uplo, const MKL_INT* n,
    float* ap, float* w, float* z, const MKL_INT* ldz, float* work,
    const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info );
void fpk_lapack_avx2_ssyev(const char* jobz, const char* uplo, const MKL_INT* n,
    float* a, const MKL_INT* lda, float* w, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_avx2_ssyevd(const char* jobz, const char* uplo, const MKL_INT* n,
    float* a, const MKL_INT* lda, float* w, float* work, const MKL_INT* lwork,
    MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info );
void fpk_lapack_avx2_ssyevr(const char* jobz, const char* range,
    const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda,
    const float* vl, const float* vu, const MKL_INT* il, const MKL_INT* iu,
    const float* abstol, MKL_INT* m, float* w, float* z, const MKL_INT* ldz,
    MKL_INT* isuppz, float* work, const MKL_INT* lwork, MKL_INT* iwork,
    const MKL_INT* liwork, MKL_INT* info );
void fpk_lapack_avx2_strtrs(const char* uplo, const char* trans,
    const char* diag, const MKL_INT* n, const MKL_INT* nrhs, const float* a,
    const MKL_INT* lda, float* b, const MKL_INT* ldb, MKL_INT* info );

void fpk_lapack_avx512_dgels(const char* trans, const MKL_INT* m,
    const MKL_INT* n, const MKL_INT* nrhs, double* a, const MKL_INT* lda,
    double* b, const MKL_INT* ldb, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_avx512_dgeqp3(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* jpvt, double* tau, double* work,
    const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_avx512_dgeqrf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, double* tau, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_avx512_dgerqf(const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, double* tau, double* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_avx512_dgesvd(const char* jobu, const char* jobvt,
    const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, double* s,
    double* u, const MKL_INT* ldu, double* vt, const MKL_INT* ldvt, double* work,
    const MKL_INT* lwork, MKL_INT* info );
double fpk_lapack_avx512_dlange(const char* norm, const MKL_INT* m,
    const MKL_INT* n, const double* a, const MKL_INT* lda, double* work );
void fpk_lapack_avx512_dlarnv(const MKL_INT* idist, MKL_INT* iseed,
    const MKL_INT* n, double* x );
void fpk_lapack_avx512_dorgqr(const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* k, double* a, const MKL_INT* lda, const double* tau,
    double* work, const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_avx512_dorgrq(const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* k, double* a, const MKL_INT* lda, const double* tau,
    double* work, const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_avx512_dormrq(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const double* a,
    const MKL_INT* lda, const double* tau, double* c, const MKL_INT* ldc,
    double* work, const MKL_INT* lwork, MKL_INT* info, int, int );
void fpk_lapack_avx512_dpftrf(const char* transr, const char* uplo,
    const MKL_INT* n, double* a, MKL_INT* info );
void fpk_lapack_avx512_dpotrf(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* info );
void fpk_lapack_avx512_dpotri(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* info );
void fpk_lapack_avx512_dpotrs(const char* uplo, const MKL_INT* n,
    const MKL_INT* nrhs, const double* a, const MKL_INT* lda, double* b,
    const MKL_INT* ldb, MKL_INT* info );
void fpk_lapack_avx512_dpptrf(const char* uplo, const MKL_INT* n, double* ap,
    MKL_INT* info );
void fpk_lapack_avx512_dpstrf(const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, MKL_INT* piv, MKL_INT* rank, const double* tol,
    double* work, MKL_INT* info );
void fpk_lapack_avx512_dspevd(const char* jobz, const char* uplo,
    const MKL_INT* n, double* ap, double* w, double* z, const MKL_INT* ldz,
    double* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork,
    MKL_INT* info );
void fpk_lapack_avx512_dsyev(const char* jobz, const char* uplo,
    const MKL_INT* n, double* a, const MKL_INT* lda, double* w, double* work,
    const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_avx512_dsyevd(const char* jobz, const char* uplo,
    const MKL_INT* n, double* a, const MKL_INT* lda, double* w, double* work,
    const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info );
void fpk_lapack_avx512_dsyevr(const char* jobz, const char* range,
    const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda,
    const double* vl, const double* vu, const MKL_INT* il, const MKL_INT* iu,
    const double* abstol, MKL_INT* m, double* w, double* z, const MKL_INT* ldz,
    MKL_INT* isuppz, double* work, const MKL_INT* lwork, MKL_INT* iwork,
    const MKL_INT* liwork, MKL_INT* info );
void fpk_lapack_avx512_dtrtrs(const char* uplo, const char* trans,
    const char* diag, const MKL_INT* n, const MKL_INT* nrhs, const double* a,
    const MKL_INT* lda, double* b, const MKL_INT* ldb, MKL_INT* info );
void fpk_lapack_avx512_sgels(const char* trans, const MKL_INT* m,
    const MKL_INT* n, const MKL_INT* nrhs, float* a, const MKL_INT* lda,
    float* b, const MKL_INT* ldb, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_avx512_sgeqp3(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* jpvt, float* tau, float* work,
    const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_avx512_sgeqrf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, float* tau, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_avx512_sgerqf(const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, float* tau, float* work, const MKL_INT* lwork,
    MKL_INT* info );
void fpk_lapack_avx512_sgesvd(const char* jobu, const char* jobvt,
    const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda, float* s,
    float* u, const MKL_INT* ldu, float* vt, const MKL_INT* ldvt, float* work,
    const MKL_INT* lwork, MKL_INT* info );
float fpk_lapack_avx512_slange(const char* norm, const MKL_INT* m,
    const MKL_INT* n, const float* a, const MKL_INT* lda, float* work );
void fpk_lapack_avx512_slarnv(const MKL_INT* idist, MKL_INT* iseed,
    const MKL_INT* n, float* x );
void fpk_lapack_avx512_sorgqr(const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* k, float* a, const MKL_INT* lda, const float* tau,
    float* work, const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_avx512_sorgrq(const MKL_INT* m, const MKL_INT* n,
    const MKL_INT* k, float* a, const MKL_INT* lda, const float* tau,
    float* work, const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_avx512_sormrq(const char* side, const char* trans,
    const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const float* a,
    const MKL_INT* lda, const float* tau, float* c, const MKL_INT* ldc,
    float* work, const MKL_INT* lwork, MKL_INT* info, int, int );
void fpk_lapack_avx512_spftrf(const char* transr, const char* uplo,
    const MKL_INT* n, float* a, MKL_INT* info );
void fpk_lapack_avx512_spotrf(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* info );
void fpk_lapack_avx512_spotri(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* info );
void fpk_lapack_avx512_spotrs(const char* uplo, const MKL_INT* n,
    const MKL_INT* nrhs, const float* a, const MKL_INT* lda, float* b,
    const MKL_INT* ldb, MKL_INT* info );
void fpk_lapack_avx512_spptrf(const char* uplo, const MKL_INT* n, float* ap,
    MKL_INT* info );
void fpk_lapack_avx512_spstrf(const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, MKL_INT* piv, MKL_INT* rank, const float* tol,
    float* work, MKL_INT* info );
void fpk_lapack_avx512_sspevd(const char* jobz, const char* uplo,
    const MKL_INT* n, float* ap, float* w, float* z, const MKL_INT* ldz,
    float* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork,
    MKL_INT* info );
void fpk_lapack_avx512_ssyev(const char* jobz, const char* uplo,
    const MKL_INT* n, float* a, const MKL_INT* lda, float* w, float* work,
    const MKL_INT* lwork, MKL_INT* info );
void fpk_lapack_avx512_ssyevd(const char* jobz, const char* uplo,
    const MKL_INT* n, float* a, const MKL_INT* lda, float* w, float* work,
    const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info );
void fpk_lapack_avx512_ssyevr(const char* jobz, const char* range,
    const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda,
    const float* vl, const float* vu, const MKL_INT* il, const MKL_INT* iu,
    const float* abstol, MKL_INT* m, float* w, float* z, const MKL_INT* ldz,
    MKL_INT* isuppz, float* work, const MKL_INT* lwork, MKL_INT* iwork,
    const MKL_INT* liwork, MKL_INT* info );
void fpk_lapack_avx512_strtrs(const char* uplo, const char* trans,
    const char* diag, const MKL_INT* n, const MKL_INT* nrhs, const float* a,
    const MKL_INT* lda, float* b, const MKL_INT* ldb, MKL_INT* info );


void fpk_pds_ssse3_pardiso(_MKL_DSS_HANDLE_t, const MKL_INT *, const MKL_INT *,
    const MKL_INT *, const MKL_INT *, const MKL_INT *, const void *,
    const MKL_INT *, const MKL_INT *, MKL_INT *, const MKL_INT *, MKL_INT *,
    const MKL_INT *, void *, void *, MKL_INT * );
MKL_INT fpk_pds_ssse3_pardiso_getenv(const _MKL_DSS_HANDLE_t,
    const enum PARDISO_ENV_PARAM*, char* );
MKL_INT fpk_pds_ssse3_pardiso_setenv(_MKL_DSS_HANDLE_t,
    const enum PARDISO_ENV_PARAM*, const char* );

void fpk_pds_sse42_pardiso(_MKL_DSS_HANDLE_t, const MKL_INT *, const MKL_INT *,
    const MKL_INT *, const MKL_INT *, const MKL_INT *, const void *,
    const MKL_INT *, const MKL_INT *, MKL_INT *, const MKL_INT *, MKL_INT *,
    const MKL_INT *, void *, void *, MKL_INT * );
MKL_INT fpk_pds_sse42_pardiso_getenv(const _MKL_DSS_HANDLE_t,
    const enum PARDISO_ENV_PARAM*, char* );
MKL_INT fpk_pds_sse42_pardiso_setenv(_MKL_DSS_HANDLE_t,
    const enum PARDISO_ENV_PARAM*, const char* );

void fpk_pds_avx_pardiso(_MKL_DSS_HANDLE_t, const MKL_INT *, const MKL_INT *,
    const MKL_INT *, const MKL_INT *, const MKL_INT *, const void *,
    const MKL_INT *, const MKL_INT *, MKL_INT *, const MKL_INT *, MKL_INT *,
    const MKL_INT *, void *, void *, MKL_INT * );
MKL_INT fpk_pds_avx_pardiso_getenv(const _MKL_DSS_HANDLE_t,
    const enum PARDISO_ENV_PARAM*, char* );
MKL_INT fpk_pds_avx_pardiso_setenv(_MKL_DSS_HANDLE_t,
    const enum PARDISO_ENV_PARAM*, const char* );

void fpk_pds_avx2_pardiso(_MKL_DSS_HANDLE_t, const MKL_INT *, const MKL_INT *,
    const MKL_INT *, const MKL_INT *, const MKL_INT *, const void *,
    const MKL_INT *, const MKL_INT *, MKL_INT *, const MKL_INT *, MKL_INT *,
    const MKL_INT *, void *, void *, MKL_INT * );
MKL_INT fpk_pds_avx2_pardiso_getenv(const _MKL_DSS_HANDLE_t,
    const enum PARDISO_ENV_PARAM*, char* );
MKL_INT fpk_pds_avx2_pardiso_setenv(_MKL_DSS_HANDLE_t,
    const enum PARDISO_ENV_PARAM*, const char* );

void fpk_pds_avx512_pardiso(_MKL_DSS_HANDLE_t, const MKL_INT *, const MKL_INT *,
    const MKL_INT *, const MKL_INT *, const MKL_INT *, const void *,
    const MKL_INT *, const MKL_INT *, MKL_INT *, const MKL_INT *, MKL_INT *,
    const MKL_INT *, void *, void *, MKL_INT * );
MKL_INT fpk_pds_avx512_pardiso_getenv(const _MKL_DSS_HANDLE_t,
    const enum PARDISO_ENV_PARAM*, char* );
MKL_INT fpk_pds_avx512_pardiso_setenv(_MKL_DSS_HANDLE_t,
    const enum PARDISO_ENV_PARAM*, const char* );


void* fpk_serv_allocate(size_t size, int alignment);
void fpk_serv_deallocate(void *ptr);

void fpk_serv_free(void *ptr);

int fpk_serv_get_ht(void);

int fpk_serv_get_max_threads(void);

int fpk_serv_get_ncorespercpu(void);
int fpk_serv_get_ncpus(void);
int fpk_serv_get_nlogicalcores(void);
int fpk_serv_set_memory_limit(int mem_type, size_t limit);

void* fpk_serv_malloc(size_t size, int align);

int fpk_serv_memcpy_s(void *dest, size_t dmax, const void *src, size_t slen);

void fpk_serv_set_num_threads(int nth);
int fpk_serv_set_num_threads_local(int nth);

int fpk_serv_strncat_s(char *dest, size_t dmax, const char *src, size_t slen);
int fpk_serv_strncpy_s(char *dest, size_t dmax, const char *src, size_t slen);
size_t fpk_serv_strnlen_s(const char *s, size_t smax);


void fpk_spblas_ssse3_mkl_dcsrmm(const char *transa, const MKL_INT *m,
    const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const char *matdescra, const double *val, const MKL_INT *indx,
    const MKL_INT *pntrb, const MKL_INT *pntre, const double *b,
    const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
void fpk_spblas_ssse3_mkl_dcsrmultd(const char *transa, const MKL_INT *m,
    const MKL_INT *n, const MKL_INT *k, double *a, MKL_INT *ja, MKL_INT *ia,
    double *b, MKL_INT *jb, MKL_INT *ib, double *c, MKL_INT *ldc);
void fpk_spblas_ssse3_mkl_dcsrmv(const char *transa, const MKL_INT *m,
    const MKL_INT *k, const double *alpha, const char *matdescra,
    const double *val, const MKL_INT *indx, const MKL_INT *pntrb,
    const MKL_INT *pntre, const double *x, const double *beta, double *y);
void fpk_spblas_ssse3_mkl_ddnscsr(const MKL_INT *job, const MKL_INT *m,
    const MKL_INT *n, double *Adns, const MKL_INT *lda, double *Acsr,
    MKL_INT *AJ, MKL_INT *AI, MKL_INT *info);
void fpk_spblas_ssse3_mkl_scsrmm(const char *transa, const MKL_INT *m,
    const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const char *matdescra, const float *val, const MKL_INT *indx,
    const MKL_INT *pntrb, const MKL_INT *pntre, const float *b,
    const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
void fpk_spblas_ssse3_mkl_scsrmultd(const char *transa, const MKL_INT *m,
    const MKL_INT *n, const MKL_INT *k, float *a, MKL_INT *ja, MKL_INT *ia,
    float *b, MKL_INT *jb, MKL_INT *ib, float *c, MKL_INT *ldc);
void fpk_spblas_ssse3_mkl_scsrmv(const char *transa, const MKL_INT *m,
    const MKL_INT *k, const float *alpha, const char *matdescra,
    const float *val, const MKL_INT *indx, const MKL_INT *pntrb,
    const MKL_INT *pntre, const float *x, const float *beta, float *y);
void fpk_spblas_ssse3_mkl_sdnscsr(const MKL_INT *job, const MKL_INT *m,
    const MKL_INT *n, float *Adns, const MKL_INT *lda, float *Acsr, MKL_INT *AJ,
    MKL_INT *AI, MKL_INT *info);

void fpk_spblas_sse42_mkl_dcsrmm(const char *transa, const MKL_INT *m,
    const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const char *matdescra, const double *val, const MKL_INT *indx,
    const MKL_INT *pntrb, const MKL_INT *pntre, const double *b,
    const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
void fpk_spblas_sse42_mkl_dcsrmultd(const char *transa, const MKL_INT *m,
    const MKL_INT *n, const MKL_INT *k, double *a, MKL_INT *ja, MKL_INT *ia,
    double *b, MKL_INT *jb, MKL_INT *ib, double *c, MKL_INT *ldc);
void fpk_spblas_sse42_mkl_dcsrmv(const char *transa, const MKL_INT *m,
    const MKL_INT *k, const double *alpha, const char *matdescra,
    const double *val, const MKL_INT *indx, const MKL_INT *pntrb,
    const MKL_INT *pntre, const double *x, const double *beta, double *y);
void fpk_spblas_sse42_mkl_ddnscsr(const MKL_INT *job, const MKL_INT *m,
    const MKL_INT *n, double *Adns, const MKL_INT *lda, double *Acsr,
    MKL_INT *AJ, MKL_INT *AI, MKL_INT *info);
void fpk_spblas_sse42_mkl_scsrmm(const char *transa, const MKL_INT *m,
    const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const char *matdescra, const float *val, const MKL_INT *indx,
    const MKL_INT *pntrb, const MKL_INT *pntre, const float *b,
    const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
void fpk_spblas_sse42_mkl_scsrmultd(const char *transa, const MKL_INT *m,
    const MKL_INT *n, const MKL_INT *k, float *a, MKL_INT *ja, MKL_INT *ia,
    float *b, MKL_INT *jb, MKL_INT *ib, float *c, MKL_INT *ldc);
void fpk_spblas_sse42_mkl_scsrmv(const char *transa, const MKL_INT *m,
    const MKL_INT *k, const float *alpha, const char *matdescra,
    const float *val, const MKL_INT *indx, const MKL_INT *pntrb,
    const MKL_INT *pntre, const float *x, const float *beta, float *y);
void fpk_spblas_sse42_mkl_sdnscsr(const MKL_INT *job, const MKL_INT *m,
    const MKL_INT *n, float *Adns, const MKL_INT *lda, float *Acsr, MKL_INT *AJ,
    MKL_INT *AI, MKL_INT *info);

void fpk_spblas_avx_mkl_dcsrmm(const char *transa, const MKL_INT *m,
    const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const char *matdescra, const double *val, const MKL_INT *indx,
    const MKL_INT *pntrb, const MKL_INT *pntre, const double *b,
    const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
void fpk_spblas_avx_mkl_dcsrmultd(const char *transa, const MKL_INT *m,
    const MKL_INT *n, const MKL_INT *k, double *a, MKL_INT *ja, MKL_INT *ia,
    double *b, MKL_INT *jb, MKL_INT *ib, double *c, MKL_INT *ldc);
void fpk_spblas_avx_mkl_dcsrmv(const char *transa, const MKL_INT *m,
    const MKL_INT *k, const double *alpha, const char *matdescra,
    const double *val, const MKL_INT *indx, const MKL_INT *pntrb,
    const MKL_INT *pntre, const double *x, const double *beta, double *y);
void fpk_spblas_avx_mkl_ddnscsr(const MKL_INT *job, const MKL_INT *m,
    const MKL_INT *n, double *Adns, const MKL_INT *lda, double *Acsr,
    MKL_INT *AJ, MKL_INT *AI, MKL_INT *info);
void fpk_spblas_avx_mkl_scsrmm(const char *transa, const MKL_INT *m,
    const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const char *matdescra, const float *val, const MKL_INT *indx,
    const MKL_INT *pntrb, const MKL_INT *pntre, const float *b,
    const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
void fpk_spblas_avx_mkl_scsrmultd(const char *transa, const MKL_INT *m,
    const MKL_INT *n, const MKL_INT *k, float *a, MKL_INT *ja, MKL_INT *ia,
    float *b, MKL_INT *jb, MKL_INT *ib, float *c, MKL_INT *ldc);
void fpk_spblas_avx_mkl_scsrmv(const char *transa, const MKL_INT *m,
    const MKL_INT *k, const float *alpha, const char *matdescra,
    const float *val, const MKL_INT *indx, const MKL_INT *pntrb,
    const MKL_INT *pntre, const float *x, const float *beta, float *y);
void fpk_spblas_avx_mkl_sdnscsr(const MKL_INT *job, const MKL_INT *m,
    const MKL_INT *n, float *Adns, const MKL_INT *lda, float *Acsr, MKL_INT *AJ,
    MKL_INT *AI, MKL_INT *info);

void fpk_spblas_avx2_mkl_dcsrmm(const char *transa, const MKL_INT *m,
    const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const char *matdescra, const double *val, const MKL_INT *indx,
    const MKL_INT *pntrb, const MKL_INT *pntre, const double *b,
    const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
void fpk_spblas_avx2_mkl_dcsrmultd(const char *transa, const MKL_INT *m,
    const MKL_INT *n, const MKL_INT *k, double *a, MKL_INT *ja, MKL_INT *ia,
    double *b, MKL_INT *jb, MKL_INT *ib, double *c, MKL_INT *ldc);
void fpk_spblas_avx2_mkl_dcsrmv(const char *transa, const MKL_INT *m,
    const MKL_INT *k, const double *alpha, const char *matdescra,
    const double *val, const MKL_INT *indx, const MKL_INT *pntrb,
    const MKL_INT *pntre, const double *x, const double *beta, double *y);
void fpk_spblas_avx2_mkl_ddnscsr(const MKL_INT *job, const MKL_INT *m,
    const MKL_INT *n, double *Adns, const MKL_INT *lda, double *Acsr,
    MKL_INT *AJ, MKL_INT *AI, MKL_INT *info);
void fpk_spblas_avx2_mkl_scsrmm(const char *transa, const MKL_INT *m,
    const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const char *matdescra, const float *val, const MKL_INT *indx,
    const MKL_INT *pntrb, const MKL_INT *pntre, const float *b,
    const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
void fpk_spblas_avx2_mkl_scsrmultd(const char *transa, const MKL_INT *m,
    const MKL_INT *n, const MKL_INT *k, float *a, MKL_INT *ja, MKL_INT *ia,
    float *b, MKL_INT *jb, MKL_INT *ib, float *c, MKL_INT *ldc);
void fpk_spblas_avx2_mkl_scsrmv(const char *transa, const MKL_INT *m,
    const MKL_INT *k, const float *alpha, const char *matdescra,
    const float *val, const MKL_INT *indx, const MKL_INT *pntrb,
    const MKL_INT *pntre, const float *x, const float *beta, float *y);
void fpk_spblas_avx2_mkl_sdnscsr(const MKL_INT *job, const MKL_INT *m,
    const MKL_INT *n, float *Adns, const MKL_INT *lda, float *Acsr, MKL_INT *AJ,
    MKL_INT *AI, MKL_INT *info);

void fpk_spblas_avx512_mkl_dcsrmm(const char *transa, const MKL_INT *m,
    const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const char *matdescra, const double *val, const MKL_INT *indx,
    const MKL_INT *pntrb, const MKL_INT *pntre, const double *b,
    const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
void fpk_spblas_avx512_mkl_dcsrmultd(const char *transa, const MKL_INT *m,
    const MKL_INT *n, const MKL_INT *k, double *a, MKL_INT *ja, MKL_INT *ia,
    double *b, MKL_INT *jb, MKL_INT *ib, double *c, MKL_INT *ldc);
void fpk_spblas_avx512_mkl_dcsrmv(const char *transa, const MKL_INT *m,
    const MKL_INT *k, const double *alpha, const char *matdescra,
    const double *val, const MKL_INT *indx, const MKL_INT *pntrb,
    const MKL_INT *pntre, const double *x, const double *beta, double *y);
void fpk_spblas_avx512_mkl_ddnscsr(const MKL_INT *job, const MKL_INT *m,
    const MKL_INT *n, double *Adns, const MKL_INT *lda, double *Acsr,
    MKL_INT *AJ, MKL_INT *AI, MKL_INT *info);
void fpk_spblas_avx512_mkl_scsrmm(const char *transa, const MKL_INT *m,
    const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const char *matdescra, const float *val, const MKL_INT *indx,
    const MKL_INT *pntrb, const MKL_INT *pntre, const float *b,
    const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
void fpk_spblas_avx512_mkl_scsrmultd(const char *transa, const MKL_INT *m,
    const MKL_INT *n, const MKL_INT *k, float *a, MKL_INT *ja, MKL_INT *ia,
    float *b, MKL_INT *jb, MKL_INT *ib, float *c, MKL_INT *ldc);
void fpk_spblas_avx512_mkl_scsrmv(const char *transa, const MKL_INT *m,
    const MKL_INT *k, const float *alpha, const char *matdescra,
    const float *val, const MKL_INT *indx, const MKL_INT *pntrb,
    const MKL_INT *pntre, const float *x, const float *beta, float *y);
void fpk_spblas_avx512_mkl_sdnscsr(const MKL_INT *job, const MKL_INT *m,
    const MKL_INT *n, float *Adns, const MKL_INT *lda, float *Acsr, MKL_INT *AJ,
    MKL_INT *AI, MKL_INT *info);


void fpk_trans_ssse3_mkl_domatcopy(char ordering, char trans, size_t rows,
    size_t cols, const double alpha, const double * A, size_t lda, double * B,
    size_t ldb);
void fpk_trans_ssse3_mkl_somatcopy(char ordering, char trans, size_t rows,
    size_t cols, const float alpha, const float * A, size_t lda, float * B,
    size_t ldb);

void fpk_trans_sse42_mkl_domatcopy(char ordering, char trans, size_t rows,
    size_t cols, const double alpha, const double * A, size_t lda, double * B,
    size_t ldb);
void fpk_trans_sse42_mkl_somatcopy(char ordering, char trans, size_t rows,
    size_t cols, const float alpha, const float * A, size_t lda, float * B,
    size_t ldb);

void fpk_trans_avx_mkl_domatcopy(char ordering, char trans, size_t rows,
    size_t cols, const double alpha, const double * A, size_t lda, double * B,
    size_t ldb);
void fpk_trans_avx_mkl_somatcopy(char ordering, char trans, size_t rows,
    size_t cols, const float alpha, const float * A, size_t lda, float * B,
    size_t ldb);

void fpk_trans_avx2_mkl_domatcopy(char ordering, char trans, size_t rows,
    size_t cols, const double alpha, const double * A, size_t lda, double * B,
    size_t ldb);
void fpk_trans_avx2_mkl_somatcopy(char ordering, char trans, size_t rows,
    size_t cols, const float alpha, const float * A, size_t lda, float * B,
    size_t ldb);

void fpk_trans_avx512_mkl_domatcopy(char ordering, char trans, size_t rows,
    size_t cols, const double alpha, const double * A, size_t lda, double * B,
    size_t ldb);
void fpk_trans_avx512_mkl_somatcopy(char ordering, char trans, size_t rows,
    size_t cols, const float alpha, const float * A, size_t lda, float * B,
    size_t ldb);


#if defined(__cplusplus)
}
#endif

#endif /*MKL_DAL_H*/
