/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
#include "services/daal_defines.h"
#include "mkl_dnn_types.h"

#if defined(__cplusplus)
extern "C" {
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

#define MKL_MEM_MCDRAM 1

#define MKL_ENABLE_AVX512_MIC_E1 5


typedef int             IppStatus;
typedef unsigned char   Ipp8u;
typedef unsigned short  Ipp16u;
typedef unsigned int    Ipp32u;
typedef signed short    Ipp16s;
typedef signed int      Ipp32s;
typedef float           Ipp32f;
typedef double          Ipp64f;

void fpk_blas_sse2_daxpy(const DAAL_INT *n, const double *alpha, const double *x,
    const DAAL_INT *incx, double *y, const DAAL_INT *incy);
void fpk_blas_sse2_dcopy(const DAAL_INT *n, const double *x, const DAAL_INT *incx,
    double *y, const DAAL_INT *incy);
double fpk_blas_sse2_ddot(const DAAL_INT *n, const double *x, const DAAL_INT *incx,
    const double *y, const DAAL_INT *incy);
void fpk_blas_sse2_dgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_sse2_dgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_sse2_dgemv(const char *trans, const DAAL_INT *m, const DAAL_INT *n,
    const double *alpha, const double *a, const DAAL_INT *lda, const double *x,
    const DAAL_INT *incx, const double *beta, double *y, const DAAL_INT *incy);
void fpk_blas_sse2_dsymm(const char *side, const char *uplo, const DAAL_INT *m,
    const DAAL_INT *n, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *b, const DAAL_INT *ldb, const double *beta, double *c,
    const DAAL_INT *ldc);
void fpk_blas_sse2_dsyr(const char *uplo, const DAAL_INT *n, const double *alpha,
    const double *x, const DAAL_INT *incx, double *a, const DAAL_INT *lda);
void fpk_blas_sse2_dsyrk(const char *uplo, const char *trans, const DAAL_INT *n,
    const DAAL_INT *k, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_sse2_dtrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const DAAL_INT *m, const DAAL_INT *n, const double *alpha,
    const double *a, const DAAL_INT *lda, double *b, const DAAL_INT *ldb);
void fpk_blas_sse2_saxpy(const DAAL_INT *n, const float *alpha, const float *x,
    const DAAL_INT *incx, float *y, const DAAL_INT *incy);
void fpk_blas_sse2_scopy(const DAAL_INT *n, const float *x, const DAAL_INT *incx,
    float *y, const DAAL_INT *incy);
float fpk_blas_sse2_sdot(const DAAL_INT *n, const float *x, const DAAL_INT *incx,
    const float *y, const DAAL_INT *incy);
void fpk_blas_sse2_sgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_sse2_sgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_sse2_sgemv(const char *trans, const DAAL_INT *m, const DAAL_INT *n,
    const float *alpha, const float *a, const DAAL_INT *lda, const float *x,
    const DAAL_INT *incx, const float *beta, float *y, const DAAL_INT *incy);
void fpk_blas_sse2_ssymm(const char *side, const char *uplo, const DAAL_INT *m,
    const DAAL_INT *n, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *b, const DAAL_INT *ldb, const float *beta, float *c,
    const DAAL_INT *ldc);
void fpk_blas_sse2_ssyr(const char *uplo, const DAAL_INT *n, const float *alpha,
    const float *x, const DAAL_INT *incx, float *a, const DAAL_INT *lda);
void fpk_blas_sse2_ssyrk(const char *uplo, const char *trans, const DAAL_INT *n,
    const DAAL_INT *k, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_sse2_strmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const DAAL_INT *m, const DAAL_INT *n, const float *alpha,
    const float *a, const DAAL_INT *lda, float *b, const DAAL_INT *ldb);
void fpk_blas_sse2_xdaxpy(const DAAL_INT *n, const double *alpha, const double *x,
    const DAAL_INT *incx, double *y, const DAAL_INT *incy);
void fpk_blas_sse2_xdcopy(const DAAL_INT *n, const double *x, const DAAL_INT *incx,
    double *y, const DAAL_INT *incy);
double fpk_blas_sse2_xddot(const DAAL_INT *n, const double *x,
    const DAAL_INT *incx, const double *y, const DAAL_INT *incy);
void fpk_blas_sse2_xdgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_sse2_xdgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_sse2_xdgemv(const char *trans, const DAAL_INT *m, const DAAL_INT *n,
    const double *alpha, const double *a, const DAAL_INT *lda, const double *x,
    const DAAL_INT *incx, const double *beta, double *y, const DAAL_INT *incy);
void fpk_blas_sse2_xdsymm(const char *side, const char *uplo, const DAAL_INT *m,
    const DAAL_INT *n, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *b, const DAAL_INT *ldb, const double *beta, double *c,
    const DAAL_INT *ldc);
void fpk_blas_sse2_xdsyr(const char *uplo, const DAAL_INT *n, const double *alpha,
    const double *x, const DAAL_INT *incx, double *a, const DAAL_INT *lda);
void fpk_blas_sse2_xdsyrk(const char *uplo, const char *trans, const DAAL_INT *n,
    const DAAL_INT *k, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_sse2_xdtrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const DAAL_INT *m, const DAAL_INT *n, const double *alpha,
    const double *a, const DAAL_INT *lda, double *b, const DAAL_INT *ldb);
void fpk_blas_sse2_xsaxpy(const DAAL_INT *n, const float *alpha, const float *x,
    const DAAL_INT *incx, float *y, const DAAL_INT *incy);
void fpk_blas_sse2_xscopy(const DAAL_INT *n, const float *x, const DAAL_INT *incx,
    float *y, const DAAL_INT *incy);
float fpk_blas_sse2_xsdot(const DAAL_INT *n, const float *x, const DAAL_INT *incx,
    const float *y, const DAAL_INT *incy);
void fpk_blas_sse2_xsgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_sse2_xsgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_sse2_xsgemv(const char *trans, const DAAL_INT *m, const DAAL_INT *n,
    const float *alpha, const float *a, const DAAL_INT *lda, const float *x,
    const DAAL_INT *incx, const float *beta, float *y, const DAAL_INT *incy);
void fpk_blas_sse2_xssymm(const char *side, const char *uplo, const DAAL_INT *m,
    const DAAL_INT *n, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *b, const DAAL_INT *ldb, const float *beta, float *c,
    const DAAL_INT *ldc);
void fpk_blas_sse2_xssyr(const char *uplo, const DAAL_INT *n, const float *alpha,
    const float *x, const DAAL_INT *incx, float *a, const DAAL_INT *lda);
void fpk_blas_sse2_xssyrk(const char *uplo, const char *trans, const DAAL_INT *n,
    const DAAL_INT *k, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_sse2_xstrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const DAAL_INT *m, const DAAL_INT *n, const float *alpha,
    const float *a, const DAAL_INT *lda, float *b, const DAAL_INT *ldb);

void fpk_blas_ssse3_daxpy(const DAAL_INT *n, const double *alpha, const double *x,
    const DAAL_INT *incx, double *y, const DAAL_INT *incy);
void fpk_blas_ssse3_dcopy(const DAAL_INT *n, const double *x, const DAAL_INT *incx,
    double *y, const DAAL_INT *incy);
double fpk_blas_ssse3_ddot(const DAAL_INT *n, const double *x,
    const DAAL_INT *incx, const double *y, const DAAL_INT *incy);
void fpk_blas_ssse3_dgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_ssse3_dgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_ssse3_dgemv(const char *trans, const DAAL_INT *m, const DAAL_INT *n,
    const double *alpha, const double *a, const DAAL_INT *lda, const double *x,
    const DAAL_INT *incx, const double *beta, double *y, const DAAL_INT *incy);
void fpk_blas_ssse3_dsymm(const char *side, const char *uplo, const DAAL_INT *m,
    const DAAL_INT *n, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *b, const DAAL_INT *ldb, const double *beta, double *c,
    const DAAL_INT *ldc);
void fpk_blas_ssse3_dsyr(const char *uplo, const DAAL_INT *n, const double *alpha,
    const double *x, const DAAL_INT *incx, double *a, const DAAL_INT *lda);
void fpk_blas_ssse3_dsyrk(const char *uplo, const char *trans, const DAAL_INT *n,
    const DAAL_INT *k, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_ssse3_dtrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const DAAL_INT *m, const DAAL_INT *n, const double *alpha,
    const double *a, const DAAL_INT *lda, double *b, const DAAL_INT *ldb);
void fpk_blas_ssse3_saxpy(const DAAL_INT *n, const float *alpha, const float *x,
    const DAAL_INT *incx, float *y, const DAAL_INT *incy);
void fpk_blas_ssse3_scopy(const DAAL_INT *n, const float *x, const DAAL_INT *incx,
    float *y, const DAAL_INT *incy);
float fpk_blas_ssse3_sdot(const DAAL_INT *n, const float *x, const DAAL_INT *incx,
    const float *y, const DAAL_INT *incy);
void fpk_blas_ssse3_sgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_ssse3_sgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_ssse3_sgemv(const char *trans, const DAAL_INT *m, const DAAL_INT *n,
    const float *alpha, const float *a, const DAAL_INT *lda, const float *x,
    const DAAL_INT *incx, const float *beta, float *y, const DAAL_INT *incy);
void fpk_blas_ssse3_ssymm(const char *side, const char *uplo, const DAAL_INT *m,
    const DAAL_INT *n, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *b, const DAAL_INT *ldb, const float *beta, float *c,
    const DAAL_INT *ldc);
void fpk_blas_ssse3_ssyr(const char *uplo, const DAAL_INT *n, const float *alpha,
    const float *x, const DAAL_INT *incx, float *a, const DAAL_INT *lda);
void fpk_blas_ssse3_ssyrk(const char *uplo, const char *trans, const DAAL_INT *n,
    const DAAL_INT *k, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_ssse3_strmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const DAAL_INT *m, const DAAL_INT *n, const float *alpha,
    const float *a, const DAAL_INT *lda, float *b, const DAAL_INT *ldb);
void fpk_blas_ssse3_xdaxpy(const DAAL_INT *n, const double *alpha,
    const double *x, const DAAL_INT *incx, double *y, const DAAL_INT *incy);
void fpk_blas_ssse3_xdcopy(const DAAL_INT *n, const double *x,
    const DAAL_INT *incx, double *y, const DAAL_INT *incy);
double fpk_blas_ssse3_xddot(const DAAL_INT *n, const double *x,
    const DAAL_INT *incx, const double *y, const DAAL_INT *incy);
void fpk_blas_ssse3_xdgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_ssse3_xdgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_ssse3_xdgemv(const char *trans, const DAAL_INT *m, const DAAL_INT *n,
    const double *alpha, const double *a, const DAAL_INT *lda, const double *x,
    const DAAL_INT *incx, const double *beta, double *y, const DAAL_INT *incy);
void fpk_blas_ssse3_xdsymm(const char *side, const char *uplo, const DAAL_INT *m,
    const DAAL_INT *n, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *b, const DAAL_INT *ldb, const double *beta, double *c,
    const DAAL_INT *ldc);
void fpk_blas_ssse3_xdsyr(const char *uplo, const DAAL_INT *n,
    const double *alpha, const double *x, const DAAL_INT *incx, double *a,
    const DAAL_INT *lda);
void fpk_blas_ssse3_xdsyrk(const char *uplo, const char *trans, const DAAL_INT *n,
    const DAAL_INT *k, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_ssse3_xdtrmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const DAAL_INT *m, const DAAL_INT *n,
    const double *alpha, const double *a, const DAAL_INT *lda, double *b,
    const DAAL_INT *ldb);
void fpk_blas_ssse3_xsaxpy(const DAAL_INT *n, const float *alpha, const float *x,
    const DAAL_INT *incx, float *y, const DAAL_INT *incy);
void fpk_blas_ssse3_xscopy(const DAAL_INT *n, const float *x, const DAAL_INT *incx,
    float *y, const DAAL_INT *incy);
float fpk_blas_ssse3_xsdot(const DAAL_INT *n, const float *x, const DAAL_INT *incx,
    const float *y, const DAAL_INT *incy);
void fpk_blas_ssse3_xsgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_ssse3_xsgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_ssse3_xsgemv(const char *trans, const DAAL_INT *m, const DAAL_INT *n,
    const float *alpha, const float *a, const DAAL_INT *lda, const float *x,
    const DAAL_INT *incx, const float *beta, float *y, const DAAL_INT *incy);
void fpk_blas_ssse3_xssymm(const char *side, const char *uplo, const DAAL_INT *m,
    const DAAL_INT *n, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *b, const DAAL_INT *ldb, const float *beta, float *c,
    const DAAL_INT *ldc);
void fpk_blas_ssse3_xssyr(const char *uplo, const DAAL_INT *n, const float *alpha,
    const float *x, const DAAL_INT *incx, float *a, const DAAL_INT *lda);
void fpk_blas_ssse3_xssyrk(const char *uplo, const char *trans, const DAAL_INT *n,
    const DAAL_INT *k, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_ssse3_xstrmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const DAAL_INT *m, const DAAL_INT *n,
    const float *alpha, const float *a, const DAAL_INT *lda, float *b,
    const DAAL_INT *ldb);

void fpk_blas_sse42_daxpy(const DAAL_INT *n, const double *alpha, const double *x,
    const DAAL_INT *incx, double *y, const DAAL_INT *incy);
void fpk_blas_sse42_dcopy(const DAAL_INT *n, const double *x, const DAAL_INT *incx,
    double *y, const DAAL_INT *incy);
double fpk_blas_sse42_ddot(const DAAL_INT *n, const double *x,
    const DAAL_INT *incx, const double *y, const DAAL_INT *incy);
void fpk_blas_sse42_dgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_sse42_dgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_sse42_dgemv(const char *trans, const DAAL_INT *m, const DAAL_INT *n,
    const double *alpha, const double *a, const DAAL_INT *lda, const double *x,
    const DAAL_INT *incx, const double *beta, double *y, const DAAL_INT *incy);
void fpk_blas_sse42_dsymm(const char *side, const char *uplo, const DAAL_INT *m,
    const DAAL_INT *n, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *b, const DAAL_INT *ldb, const double *beta, double *c,
    const DAAL_INT *ldc);
void fpk_blas_sse42_dsyr(const char *uplo, const DAAL_INT *n, const double *alpha,
    const double *x, const DAAL_INT *incx, double *a, const DAAL_INT *lda);
void fpk_blas_sse42_dsyrk(const char *uplo, const char *trans, const DAAL_INT *n,
    const DAAL_INT *k, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_sse42_dtrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const DAAL_INT *m, const DAAL_INT *n, const double *alpha,
    const double *a, const DAAL_INT *lda, double *b, const DAAL_INT *ldb);
void fpk_blas_sse42_saxpy(const DAAL_INT *n, const float *alpha, const float *x,
    const DAAL_INT *incx, float *y, const DAAL_INT *incy);
void fpk_blas_sse42_scopy(const DAAL_INT *n, const float *x, const DAAL_INT *incx,
    float *y, const DAAL_INT *incy);
float fpk_blas_sse42_sdot(const DAAL_INT *n, const float *x, const DAAL_INT *incx,
    const float *y, const DAAL_INT *incy);
void fpk_blas_sse42_sgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_sse42_sgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_sse42_sgemv(const char *trans, const DAAL_INT *m, const DAAL_INT *n,
    const float *alpha, const float *a, const DAAL_INT *lda, const float *x,
    const DAAL_INT *incx, const float *beta, float *y, const DAAL_INT *incy);
void fpk_blas_sse42_ssymm(const char *side, const char *uplo, const DAAL_INT *m,
    const DAAL_INT *n, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *b, const DAAL_INT *ldb, const float *beta, float *c,
    const DAAL_INT *ldc);
void fpk_blas_sse42_ssyr(const char *uplo, const DAAL_INT *n, const float *alpha,
    const float *x, const DAAL_INT *incx, float *a, const DAAL_INT *lda);
void fpk_blas_sse42_ssyrk(const char *uplo, const char *trans, const DAAL_INT *n,
    const DAAL_INT *k, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_sse42_strmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const DAAL_INT *m, const DAAL_INT *n, const float *alpha,
    const float *a, const DAAL_INT *lda, float *b, const DAAL_INT *ldb);
void fpk_blas_sse42_xdaxpy(const DAAL_INT *n, const double *alpha,
    const double *x, const DAAL_INT *incx, double *y, const DAAL_INT *incy);
void fpk_blas_sse42_xdcopy(const DAAL_INT *n, const double *x,
    const DAAL_INT *incx, double *y, const DAAL_INT *incy);
double fpk_blas_sse42_xddot(const DAAL_INT *n, const double *x,
    const DAAL_INT *incx, const double *y, const DAAL_INT *incy);
void fpk_blas_sse42_xdgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_sse42_xdgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_sse42_xdgemv(const char *trans, const DAAL_INT *m, const DAAL_INT *n,
    const double *alpha, const double *a, const DAAL_INT *lda, const double *x,
    const DAAL_INT *incx, const double *beta, double *y, const DAAL_INT *incy);
void fpk_blas_sse42_xdsymm(const char *side, const char *uplo, const DAAL_INT *m,
    const DAAL_INT *n, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *b, const DAAL_INT *ldb, const double *beta, double *c,
    const DAAL_INT *ldc);
void fpk_blas_sse42_xdsyr(const char *uplo, const DAAL_INT *n,
    const double *alpha, const double *x, const DAAL_INT *incx, double *a,
    const DAAL_INT *lda);
void fpk_blas_sse42_xdsyrk(const char *uplo, const char *trans, const DAAL_INT *n,
    const DAAL_INT *k, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_sse42_xdtrmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const DAAL_INT *m, const DAAL_INT *n,
    const double *alpha, const double *a, const DAAL_INT *lda, double *b,
    const DAAL_INT *ldb);
void fpk_blas_sse42_xsaxpy(const DAAL_INT *n, const float *alpha, const float *x,
    const DAAL_INT *incx, float *y, const DAAL_INT *incy);
void fpk_blas_sse42_xscopy(const DAAL_INT *n, const float *x, const DAAL_INT *incx,
    float *y, const DAAL_INT *incy);
float fpk_blas_sse42_xsdot(const DAAL_INT *n, const float *x, const DAAL_INT *incx,
    const float *y, const DAAL_INT *incy);
void fpk_blas_sse42_xsgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_sse42_xsgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_sse42_xsgemv(const char *trans, const DAAL_INT *m, const DAAL_INT *n,
    const float *alpha, const float *a, const DAAL_INT *lda, const float *x,
    const DAAL_INT *incx, const float *beta, float *y, const DAAL_INT *incy);
void fpk_blas_sse42_xssymm(const char *side, const char *uplo, const DAAL_INT *m,
    const DAAL_INT *n, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *b, const DAAL_INT *ldb, const float *beta, float *c,
    const DAAL_INT *ldc);
void fpk_blas_sse42_xssyr(const char *uplo, const DAAL_INT *n, const float *alpha,
    const float *x, const DAAL_INT *incx, float *a, const DAAL_INT *lda);
void fpk_blas_sse42_xssyrk(const char *uplo, const char *trans, const DAAL_INT *n,
    const DAAL_INT *k, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_sse42_xstrmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const DAAL_INT *m, const DAAL_INT *n,
    const float *alpha, const float *a, const DAAL_INT *lda, float *b,
    const DAAL_INT *ldb);

void fpk_blas_avx_daxpy(const DAAL_INT *n, const double *alpha, const double *x,
    const DAAL_INT *incx, double *y, const DAAL_INT *incy);
void fpk_blas_avx_dcopy(const DAAL_INT *n, const double *x, const DAAL_INT *incx,
    double *y, const DAAL_INT *incy);
double fpk_blas_avx_ddot(const DAAL_INT *n, const double *x, const DAAL_INT *incx,
    const double *y, const DAAL_INT *incy);
void fpk_blas_avx_dgemm(const char *transa, const char *transb, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, const double *alpha, const double *a,
    const DAAL_INT *lda, const double *b, const DAAL_INT *ldb, const double *beta,
    double *c, const DAAL_INT *ldc);
void fpk_blas_avx_dgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_avx_dgemv(const char *trans, const DAAL_INT *m, const DAAL_INT *n,
    const double *alpha, const double *a, const DAAL_INT *lda, const double *x,
    const DAAL_INT *incx, const double *beta, double *y, const DAAL_INT *incy);
void fpk_blas_avx_dsymm(const char *side, const char *uplo, const DAAL_INT *m,
    const DAAL_INT *n, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *b, const DAAL_INT *ldb, const double *beta, double *c,
    const DAAL_INT *ldc);
void fpk_blas_avx_dsyr(const char *uplo, const DAAL_INT *n, const double *alpha,
    const double *x, const DAAL_INT *incx, double *a, const DAAL_INT *lda);
void fpk_blas_avx_dsyrk(const char *uplo, const char *trans, const DAAL_INT *n,
    const DAAL_INT *k, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_avx_dtrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const DAAL_INT *m, const DAAL_INT *n, const double *alpha,
    const double *a, const DAAL_INT *lda, double *b, const DAAL_INT *ldb);
void fpk_blas_avx_saxpy(const DAAL_INT *n, const float *alpha, const float *x,
    const DAAL_INT *incx, float *y, const DAAL_INT *incy);
void fpk_blas_avx_scopy(const DAAL_INT *n, const float *x, const DAAL_INT *incx,
    float *y, const DAAL_INT *incy);
float fpk_blas_avx_sdot(const DAAL_INT *n, const float *x, const DAAL_INT *incx,
    const float *y, const DAAL_INT *incy);
void fpk_blas_avx_sgemm(const char *transa, const char *transb, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, const float *alpha, const float *a,
    const DAAL_INT *lda, const float *b, const DAAL_INT *ldb, const float *beta,
    float *c, const DAAL_INT *ldc);
void fpk_blas_avx_sgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_avx_sgemv(const char *trans, const DAAL_INT *m, const DAAL_INT *n,
    const float *alpha, const float *a, const DAAL_INT *lda, const float *x,
    const DAAL_INT *incx, const float *beta, float *y, const DAAL_INT *incy);
void fpk_blas_avx_ssymm(const char *side, const char *uplo, const DAAL_INT *m,
    const DAAL_INT *n, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *b, const DAAL_INT *ldb, const float *beta, float *c,
    const DAAL_INT *ldc);
void fpk_blas_avx_ssyr(const char *uplo, const DAAL_INT *n, const float *alpha,
    const float *x, const DAAL_INT *incx, float *a, const DAAL_INT *lda);
void fpk_blas_avx_ssyrk(const char *uplo, const char *trans, const DAAL_INT *n,
    const DAAL_INT *k, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_avx_strmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const DAAL_INT *m, const DAAL_INT *n, const float *alpha,
    const float *a, const DAAL_INT *lda, float *b, const DAAL_INT *ldb);
void fpk_blas_avx_xdaxpy(const DAAL_INT *n, const double *alpha, const double *x,
    const DAAL_INT *incx, double *y, const DAAL_INT *incy);
void fpk_blas_avx_xdcopy(const DAAL_INT *n, const double *x, const DAAL_INT *incx,
    double *y, const DAAL_INT *incy);
double fpk_blas_avx_xddot(const DAAL_INT *n, const double *x, const DAAL_INT *incx,
    const double *y, const DAAL_INT *incy);
void fpk_blas_avx_xdgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_avx_xdgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_avx_xdgemv(const char *trans, const DAAL_INT *m, const DAAL_INT *n,
    const double *alpha, const double *a, const DAAL_INT *lda, const double *x,
    const DAAL_INT *incx, const double *beta, double *y, const DAAL_INT *incy);
void fpk_blas_avx_xdsymm(const char *side, const char *uplo, const DAAL_INT *m,
    const DAAL_INT *n, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *b, const DAAL_INT *ldb, const double *beta, double *c,
    const DAAL_INT *ldc);
void fpk_blas_avx_xdsyr(const char *uplo, const DAAL_INT *n, const double *alpha,
    const double *x, const DAAL_INT *incx, double *a, const DAAL_INT *lda);
void fpk_blas_avx_xdsyrk(const char *uplo, const char *trans, const DAAL_INT *n,
    const DAAL_INT *k, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_avx_xdtrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const DAAL_INT *m, const DAAL_INT *n, const double *alpha,
    const double *a, const DAAL_INT *lda, double *b, const DAAL_INT *ldb);
void fpk_blas_avx_xsaxpy(const DAAL_INT *n, const float *alpha, const float *x,
    const DAAL_INT *incx, float *y, const DAAL_INT *incy);
void fpk_blas_avx_xscopy(const DAAL_INT *n, const float *x, const DAAL_INT *incx,
    float *y, const DAAL_INT *incy);
float fpk_blas_avx_xsdot(const DAAL_INT *n, const float *x, const DAAL_INT *incx,
    const float *y, const DAAL_INT *incy);
void fpk_blas_avx_xsgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_avx_xsgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_avx_xsgemv(const char *trans, const DAAL_INT *m, const DAAL_INT *n,
    const float *alpha, const float *a, const DAAL_INT *lda, const float *x,
    const DAAL_INT *incx, const float *beta, float *y, const DAAL_INT *incy);
void fpk_blas_avx_xssymm(const char *side, const char *uplo, const DAAL_INT *m,
    const DAAL_INT *n, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *b, const DAAL_INT *ldb, const float *beta, float *c,
    const DAAL_INT *ldc);
void fpk_blas_avx_xssyr(const char *uplo, const DAAL_INT *n, const float *alpha,
    const float *x, const DAAL_INT *incx, float *a, const DAAL_INT *lda);
void fpk_blas_avx_xssyrk(const char *uplo, const char *trans, const DAAL_INT *n,
    const DAAL_INT *k, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_avx_xstrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const DAAL_INT *m, const DAAL_INT *n, const float *alpha,
    const float *a, const DAAL_INT *lda, float *b, const DAAL_INT *ldb);

void fpk_blas_avx2_daxpy(const DAAL_INT *n, const double *alpha, const double *x,
    const DAAL_INT *incx, double *y, const DAAL_INT *incy);
void fpk_blas_avx2_dcopy(const DAAL_INT *n, const double *x, const DAAL_INT *incx,
    double *y, const DAAL_INT *incy);
double fpk_blas_avx2_ddot(const DAAL_INT *n, const double *x, const DAAL_INT *incx,
    const double *y, const DAAL_INT *incy);
void fpk_blas_avx2_dgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_avx2_dgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_avx2_dgemv(const char *trans, const DAAL_INT *m, const DAAL_INT *n,
    const double *alpha, const double *a, const DAAL_INT *lda, const double *x,
    const DAAL_INT *incx, const double *beta, double *y, const DAAL_INT *incy);
void fpk_blas_avx2_dsymm(const char *side, const char *uplo, const DAAL_INT *m,
    const DAAL_INT *n, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *b, const DAAL_INT *ldb, const double *beta, double *c,
    const DAAL_INT *ldc);
void fpk_blas_avx2_dsyr(const char *uplo, const DAAL_INT *n, const double *alpha,
    const double *x, const DAAL_INT *incx, double *a, const DAAL_INT *lda);
void fpk_blas_avx2_dsyrk(const char *uplo, const char *trans, const DAAL_INT *n,
    const DAAL_INT *k, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_avx2_dtrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const DAAL_INT *m, const DAAL_INT *n, const double *alpha,
    const double *a, const DAAL_INT *lda, double *b, const DAAL_INT *ldb);
void fpk_blas_avx2_saxpy(const DAAL_INT *n, const float *alpha, const float *x,
    const DAAL_INT *incx, float *y, const DAAL_INT *incy);
void fpk_blas_avx2_scopy(const DAAL_INT *n, const float *x, const DAAL_INT *incx,
    float *y, const DAAL_INT *incy);
float fpk_blas_avx2_sdot(const DAAL_INT *n, const float *x, const DAAL_INT *incx,
    const float *y, const DAAL_INT *incy);
void fpk_blas_avx2_sgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_avx2_sgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_avx2_sgemv(const char *trans, const DAAL_INT *m, const DAAL_INT *n,
    const float *alpha, const float *a, const DAAL_INT *lda, const float *x,
    const DAAL_INT *incx, const float *beta, float *y, const DAAL_INT *incy);
void fpk_blas_avx2_ssymm(const char *side, const char *uplo, const DAAL_INT *m,
    const DAAL_INT *n, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *b, const DAAL_INT *ldb, const float *beta, float *c,
    const DAAL_INT *ldc);
void fpk_blas_avx2_ssyr(const char *uplo, const DAAL_INT *n, const float *alpha,
    const float *x, const DAAL_INT *incx, float *a, const DAAL_INT *lda);
void fpk_blas_avx2_ssyrk(const char *uplo, const char *trans, const DAAL_INT *n,
    const DAAL_INT *k, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_avx2_strmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const DAAL_INT *m, const DAAL_INT *n, const float *alpha,
    const float *a, const DAAL_INT *lda, float *b, const DAAL_INT *ldb);
void fpk_blas_avx2_xdaxpy(const DAAL_INT *n, const double *alpha, const double *x,
    const DAAL_INT *incx, double *y, const DAAL_INT *incy);
void fpk_blas_avx2_xdcopy(const DAAL_INT *n, const double *x, const DAAL_INT *incx,
    double *y, const DAAL_INT *incy);
double fpk_blas_avx2_xddot(const DAAL_INT *n, const double *x,
    const DAAL_INT *incx, const double *y, const DAAL_INT *incy);
void fpk_blas_avx2_xdgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_avx2_xdgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_avx2_xdgemv(const char *trans, const DAAL_INT *m, const DAAL_INT *n,
    const double *alpha, const double *a, const DAAL_INT *lda, const double *x,
    const DAAL_INT *incx, const double *beta, double *y, const DAAL_INT *incy);
void fpk_blas_avx2_xdsymm(const char *side, const char *uplo, const DAAL_INT *m,
    const DAAL_INT *n, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *b, const DAAL_INT *ldb, const double *beta, double *c,
    const DAAL_INT *ldc);
void fpk_blas_avx2_xdsyr(const char *uplo, const DAAL_INT *n, const double *alpha,
    const double *x, const DAAL_INT *incx, double *a, const DAAL_INT *lda);
void fpk_blas_avx2_xdsyrk(const char *uplo, const char *trans, const DAAL_INT *n,
    const DAAL_INT *k, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_avx2_xdtrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const DAAL_INT *m, const DAAL_INT *n, const double *alpha,
    const double *a, const DAAL_INT *lda, double *b, const DAAL_INT *ldb);
void fpk_blas_avx2_xsaxpy(const DAAL_INT *n, const float *alpha, const float *x,
    const DAAL_INT *incx, float *y, const DAAL_INT *incy);
void fpk_blas_avx2_xscopy(const DAAL_INT *n, const float *x, const DAAL_INT *incx,
    float *y, const DAAL_INT *incy);
float fpk_blas_avx2_xsdot(const DAAL_INT *n, const float *x, const DAAL_INT *incx,
    const float *y, const DAAL_INT *incy);
void fpk_blas_avx2_xsgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_avx2_xsgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_avx2_xsgemv(const char *trans, const DAAL_INT *m, const DAAL_INT *n,
    const float *alpha, const float *a, const DAAL_INT *lda, const float *x,
    const DAAL_INT *incx, const float *beta, float *y, const DAAL_INT *incy);
void fpk_blas_avx2_xssymm(const char *side, const char *uplo, const DAAL_INT *m,
    const DAAL_INT *n, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *b, const DAAL_INT *ldb, const float *beta, float *c,
    const DAAL_INT *ldc);
void fpk_blas_avx2_xssyr(const char *uplo, const DAAL_INT *n, const float *alpha,
    const float *x, const DAAL_INT *incx, float *a, const DAAL_INT *lda);
void fpk_blas_avx2_xssyrk(const char *uplo, const char *trans, const DAAL_INT *n,
    const DAAL_INT *k, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_avx2_xstrmm(const char *side, const char *uplo, const char *transa,
    const char *diag, const DAAL_INT *m, const DAAL_INT *n, const float *alpha,
    const float *a, const DAAL_INT *lda, float *b, const DAAL_INT *ldb);

void fpk_blas_avx512_daxpy(const DAAL_INT *n, const double *alpha,
    const double *x, const DAAL_INT *incx, double *y, const DAAL_INT *incy);
void fpk_blas_avx512_dcopy(const DAAL_INT *n, const double *x,
    const DAAL_INT *incx, double *y, const DAAL_INT *incy);
double fpk_blas_avx512_ddot(const DAAL_INT *n, const double *x,
    const DAAL_INT *incx, const double *y, const DAAL_INT *incy);
void fpk_blas_avx512_dgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_avx512_dgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_avx512_dgemv(const char *trans, const DAAL_INT *m, const DAAL_INT *n,
    const double *alpha, const double *a, const DAAL_INT *lda, const double *x,
    const DAAL_INT *incx, const double *beta, double *y, const DAAL_INT *incy);
void fpk_blas_avx512_dsymm(const char *side, const char *uplo, const DAAL_INT *m,
    const DAAL_INT *n, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *b, const DAAL_INT *ldb, const double *beta, double *c,
    const DAAL_INT *ldc);
void fpk_blas_avx512_dsyr(const char *uplo, const DAAL_INT *n,
    const double *alpha, const double *x, const DAAL_INT *incx, double *a,
    const DAAL_INT *lda);
void fpk_blas_avx512_dsyrk(const char *uplo, const char *trans, const DAAL_INT *n,
    const DAAL_INT *k, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_avx512_dtrmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const DAAL_INT *m, const DAAL_INT *n,
    const double *alpha, const double *a, const DAAL_INT *lda, double *b,
    const DAAL_INT *ldb);
void fpk_blas_avx512_saxpy(const DAAL_INT *n, const float *alpha, const float *x,
    const DAAL_INT *incx, float *y, const DAAL_INT *incy);
void fpk_blas_avx512_scopy(const DAAL_INT *n, const float *x, const DAAL_INT *incx,
    float *y, const DAAL_INT *incy);
float fpk_blas_avx512_sdot(const DAAL_INT *n, const float *x, const DAAL_INT *incx,
    const float *y, const DAAL_INT *incy);
void fpk_blas_avx512_sgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_avx512_sgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_avx512_sgemv(const char *trans, const DAAL_INT *m, const DAAL_INT *n,
    const float *alpha, const float *a, const DAAL_INT *lda, const float *x,
    const DAAL_INT *incx, const float *beta, float *y, const DAAL_INT *incy);
void fpk_blas_avx512_ssymm(const char *side, const char *uplo, const DAAL_INT *m,
    const DAAL_INT *n, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *b, const DAAL_INT *ldb, const float *beta, float *c,
    const DAAL_INT *ldc);
void fpk_blas_avx512_ssyr(const char *uplo, const DAAL_INT *n, const float *alpha,
    const float *x, const DAAL_INT *incx, float *a, const DAAL_INT *lda);
void fpk_blas_avx512_ssyrk(const char *uplo, const char *trans, const DAAL_INT *n,
    const DAAL_INT *k, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_avx512_strmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const DAAL_INT *m, const DAAL_INT *n,
    const float *alpha, const float *a, const DAAL_INT *lda, float *b,
    const DAAL_INT *ldb);
void fpk_blas_avx512_xdaxpy(const DAAL_INT *n, const double *alpha,
    const double *x, const DAAL_INT *incx, double *y, const DAAL_INT *incy);
void fpk_blas_avx512_xdcopy(const DAAL_INT *n, const double *x,
    const DAAL_INT *incx, double *y, const DAAL_INT *incy);
double fpk_blas_avx512_xddot(const DAAL_INT *n, const double *x,
    const DAAL_INT *incx, const double *y, const DAAL_INT *incy);
void fpk_blas_avx512_xdgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_avx512_xdgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_avx512_xdgemv(const char *trans, const DAAL_INT *m,
    const DAAL_INT *n, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *x, const DAAL_INT *incx, const double *beta, double *y,
    const DAAL_INT *incy);
void fpk_blas_avx512_xdsymm(const char *side, const char *uplo, const DAAL_INT *m,
    const DAAL_INT *n, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *b, const DAAL_INT *ldb, const double *beta, double *c,
    const DAAL_INT *ldc);
void fpk_blas_avx512_xdsyr(const char *uplo, const DAAL_INT *n,
    const double *alpha, const double *x, const DAAL_INT *incx, double *a,
    const DAAL_INT *lda);
void fpk_blas_avx512_xdsyrk(const char *uplo, const char *trans,
    const DAAL_INT *n, const DAAL_INT *k, const double *alpha, const double *a,
    const DAAL_INT *lda, const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_avx512_xdtrmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const DAAL_INT *m, const DAAL_INT *n,
    const double *alpha, const double *a, const DAAL_INT *lda, double *b,
    const DAAL_INT *ldb);
void fpk_blas_avx512_xsaxpy(const DAAL_INT *n, const float *alpha, const float *x,
    const DAAL_INT *incx, float *y, const DAAL_INT *incy);
void fpk_blas_avx512_xscopy(const DAAL_INT *n, const float *x,
    const DAAL_INT *incx, float *y, const DAAL_INT *incy);
float fpk_blas_avx512_xsdot(const DAAL_INT *n, const float *x,
    const DAAL_INT *incx, const float *y, const DAAL_INT *incy);
void fpk_blas_avx512_xsgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_avx512_xsgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_avx512_xsgemv(const char *trans, const DAAL_INT *m,
    const DAAL_INT *n, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *x, const DAAL_INT *incx, const float *beta, float *y,
    const DAAL_INT *incy);
void fpk_blas_avx512_xssymm(const char *side, const char *uplo, const DAAL_INT *m,
    const DAAL_INT *n, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *b, const DAAL_INT *ldb, const float *beta, float *c,
    const DAAL_INT *ldc);
void fpk_blas_avx512_xssyr(const char *uplo, const DAAL_INT *n,
    const float *alpha, const float *x, const DAAL_INT *incx, float *a,
    const DAAL_INT *lda);
void fpk_blas_avx512_xssyrk(const char *uplo, const char *trans,
    const DAAL_INT *n, const DAAL_INT *k, const float *alpha, const float *a,
    const DAAL_INT *lda, const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_avx512_xstrmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const DAAL_INT *m, const DAAL_INT *n,
    const float *alpha, const float *a, const DAAL_INT *lda, float *b,
    const DAAL_INT *ldb);

void fpk_blas_avx512_mic_daxpy(const DAAL_INT *n, const double *alpha,
    const double *x, const DAAL_INT *incx, double *y, const DAAL_INT *incy);
void fpk_blas_avx512_mic_dcopy(const DAAL_INT *n, const double *x,
    const DAAL_INT *incx, double *y, const DAAL_INT *incy);
double fpk_blas_avx512_mic_ddot(const DAAL_INT *n, const double *x,
    const DAAL_INT *incx, const double *y, const DAAL_INT *incy);
void fpk_blas_avx512_mic_dgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_avx512_mic_dgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_avx512_mic_dgemv(const char *trans, const DAAL_INT *m,
    const DAAL_INT *n, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *x, const DAAL_INT *incx, const double *beta, double *y,
    const DAAL_INT *incy);
void fpk_blas_avx512_mic_dsymm(const char *side, const char *uplo,
    const DAAL_INT *m, const DAAL_INT *n, const double *alpha, const double *a,
    const DAAL_INT *lda, const double *b, const DAAL_INT *ldb, const double *beta,
    double *c, const DAAL_INT *ldc);
void fpk_blas_avx512_mic_dsyr(const char *uplo, const DAAL_INT *n,
    const double *alpha, const double *x, const DAAL_INT *incx, double *a,
    const DAAL_INT *lda);
void fpk_blas_avx512_mic_dsyrk(const char *uplo, const char *trans,
    const DAAL_INT *n, const DAAL_INT *k, const double *alpha, const double *a,
    const DAAL_INT *lda, const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_avx512_mic_dtrmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const DAAL_INT *m, const DAAL_INT *n,
    const double *alpha, const double *a, const DAAL_INT *lda, double *b,
    const DAAL_INT *ldb);
void fpk_blas_avx512_mic_saxpy(const DAAL_INT *n, const float *alpha,
    const float *x, const DAAL_INT *incx, float *y, const DAAL_INT *incy);
void fpk_blas_avx512_mic_scopy(const DAAL_INT *n, const float *x,
    const DAAL_INT *incx, float *y, const DAAL_INT *incy);
float fpk_blas_avx512_mic_sdot(const DAAL_INT *n, const float *x,
    const DAAL_INT *incx, const float *y, const DAAL_INT *incy);
void fpk_blas_avx512_mic_sgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_avx512_mic_sgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_avx512_mic_sgemv(const char *trans, const DAAL_INT *m,
    const DAAL_INT *n, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *x, const DAAL_INT *incx, const float *beta, float *y,
    const DAAL_INT *incy);
void fpk_blas_avx512_mic_ssymm(const char *side, const char *uplo,
    const DAAL_INT *m, const DAAL_INT *n, const float *alpha, const float *a,
    const DAAL_INT *lda, const float *b, const DAAL_INT *ldb, const float *beta,
    float *c, const DAAL_INT *ldc);
void fpk_blas_avx512_mic_ssyr(const char *uplo, const DAAL_INT *n,
    const float *alpha, const float *x, const DAAL_INT *incx, float *a,
    const DAAL_INT *lda);
void fpk_blas_avx512_mic_ssyrk(const char *uplo, const char *trans,
    const DAAL_INT *n, const DAAL_INT *k, const float *alpha, const float *a,
    const DAAL_INT *lda, const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_avx512_mic_strmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const DAAL_INT *m, const DAAL_INT *n,
    const float *alpha, const float *a, const DAAL_INT *lda, float *b,
    const DAAL_INT *ldb);
void fpk_blas_avx512_mic_xdaxpy(const DAAL_INT *n, const double *alpha,
    const double *x, const DAAL_INT *incx, double *y, const DAAL_INT *incy);
void fpk_blas_avx512_mic_xdcopy(const DAAL_INT *n, const double *x,
    const DAAL_INT *incx, double *y, const DAAL_INT *incy);
double fpk_blas_avx512_mic_xddot(const DAAL_INT *n, const double *x,
    const DAAL_INT *incx, const double *y, const DAAL_INT *incy);
void fpk_blas_avx512_mic_xdgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_avx512_mic_xdgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const double *a, const DAAL_INT *lda, const double *b, const DAAL_INT *ldb,
    const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_avx512_mic_xdgemv(const char *trans, const DAAL_INT *m,
    const DAAL_INT *n, const double *alpha, const double *a, const DAAL_INT *lda,
    const double *x, const DAAL_INT *incx, const double *beta, double *y,
    const DAAL_INT *incy);
void fpk_blas_avx512_mic_xdsymm(const char *side, const char *uplo,
    const DAAL_INT *m, const DAAL_INT *n, const double *alpha, const double *a,
    const DAAL_INT *lda, const double *b, const DAAL_INT *ldb, const double *beta,
    double *c, const DAAL_INT *ldc);
void fpk_blas_avx512_mic_xdsyr(const char *uplo, const DAAL_INT *n,
    const double *alpha, const double *x, const DAAL_INT *incx, double *a,
    const DAAL_INT *lda);
void fpk_blas_avx512_mic_xdsyrk(const char *uplo, const char *trans,
    const DAAL_INT *n, const DAAL_INT *k, const double *alpha, const double *a,
    const DAAL_INT *lda, const double *beta, double *c, const DAAL_INT *ldc);
void fpk_blas_avx512_mic_xdtrmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const DAAL_INT *m, const DAAL_INT *n,
    const double *alpha, const double *a, const DAAL_INT *lda, double *b,
    const DAAL_INT *ldb);
void fpk_blas_avx512_mic_xsaxpy(const DAAL_INT *n, const float *alpha,
    const float *x, const DAAL_INT *incx, float *y, const DAAL_INT *incy);
void fpk_blas_avx512_mic_xscopy(const DAAL_INT *n, const float *x,
    const DAAL_INT *incx, float *y, const DAAL_INT *incy);
float fpk_blas_avx512_mic_xsdot(const DAAL_INT *n, const float *x,
    const DAAL_INT *incx, const float *y, const DAAL_INT *incy);
void fpk_blas_avx512_mic_xsgemm(const char *transa, const char *transb,
    const DAAL_INT *m, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_avx512_mic_xsgemmt(const char *uplo, const char *transa,
    const char *transb, const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const float *a, const DAAL_INT *lda, const float *b, const DAAL_INT *ldb,
    const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_avx512_mic_xsgemv(const char *trans, const DAAL_INT *m,
    const DAAL_INT *n, const float *alpha, const float *a, const DAAL_INT *lda,
    const float *x, const DAAL_INT *incx, const float *beta, float *y,
    const DAAL_INT *incy);
void fpk_blas_avx512_mic_xssymm(const char *side, const char *uplo,
    const DAAL_INT *m, const DAAL_INT *n, const float *alpha, const float *a,
    const DAAL_INT *lda, const float *b, const DAAL_INT *ldb, const float *beta,
    float *c, const DAAL_INT *ldc);
void fpk_blas_avx512_mic_xssyr(const char *uplo, const DAAL_INT *n,
    const float *alpha, const float *x, const DAAL_INT *incx, float *a,
    const DAAL_INT *lda);
void fpk_blas_avx512_mic_xssyrk(const char *uplo, const char *trans,
    const DAAL_INT *n, const DAAL_INT *k, const float *alpha, const float *a,
    const DAAL_INT *lda, const float *beta, float *c, const DAAL_INT *ldc);
void fpk_blas_avx512_mic_xstrmm(const char *side, const char *uplo,
    const char *transa, const char *diag, const DAAL_INT *m, const DAAL_INT *n,
    const float *alpha, const float *a, const DAAL_INT *lda, float *b,
    const DAAL_INT *ldb);


IppStatus fpk_dft_sse2_ippsSortRadixAscend_16s_I(Ipp16s *pSrcDst, Ipp16s *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse2_ippsSortRadixAscend_16u_I(Ipp16u *pSrcDst, Ipp16u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse2_ippsSortRadixAscend_32f_I(Ipp32f *pSrcDst, Ipp32f *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse2_ippsSortRadixAscend_32s_I(Ipp32s *pSrcDst, Ipp32s *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse2_ippsSortRadixAscend_32u_I(Ipp32u *pSrcDst, Ipp32u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse2_ippsSortRadixAscend_64f_I(Ipp64f *pSrcDst, Ipp64f *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse2_ippsSortRadixAscend_8u_I(Ipp8u *pSrcDst, Ipp8u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse2_ippsSortRadixDescend_16s_I(Ipp16s *pSrcDst, Ipp16s *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse2_ippsSortRadixDescend_16u_I(Ipp16u *pSrcDst, Ipp16u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse2_ippsSortRadixDescend_32f_I(Ipp32f *pSrcDst, Ipp32f *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse2_ippsSortRadixDescend_32s_I(Ipp32s *pSrcDst, Ipp32s *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse2_ippsSortRadixDescend_32u_I(Ipp32u *pSrcDst, Ipp32u *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse2_ippsSortRadixDescend_64f_I(Ipp64f *pSrcDst, Ipp64f *pTmp,
    Ipp32s len);
IppStatus fpk_dft_sse2_ippsSortRadixDescend_8u_I(Ipp8u *pSrcDst, Ipp8u *pTmp,
    Ipp32s len);

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

IppStatus fpk_dft_avx512_mic_ippsSortRadixAscend_16s_I(Ipp16s *pSrcDst,
    Ipp16s *pTmp, Ipp32s len);
IppStatus fpk_dft_avx512_mic_ippsSortRadixAscend_16u_I(Ipp16u *pSrcDst,
    Ipp16u *pTmp, Ipp32s len);
IppStatus fpk_dft_avx512_mic_ippsSortRadixAscend_32f_I(Ipp32f *pSrcDst,
    Ipp32f *pTmp, Ipp32s len);
IppStatus fpk_dft_avx512_mic_ippsSortRadixAscend_32s_I(Ipp32s *pSrcDst,
    Ipp32s *pTmp, Ipp32s len);
IppStatus fpk_dft_avx512_mic_ippsSortRadixAscend_32u_I(Ipp32u *pSrcDst,
    Ipp32u *pTmp, Ipp32s len);
IppStatus fpk_dft_avx512_mic_ippsSortRadixAscend_64f_I(Ipp64f *pSrcDst,
    Ipp64f *pTmp, Ipp32s len);
IppStatus fpk_dft_avx512_mic_ippsSortRadixAscend_8u_I(Ipp8u *pSrcDst,
    Ipp8u *pTmp, Ipp32s len);
IppStatus fpk_dft_avx512_mic_ippsSortRadixDescend_16s_I(Ipp16s *pSrcDst,
    Ipp16s *pTmp, Ipp32s len);
IppStatus fpk_dft_avx512_mic_ippsSortRadixDescend_16u_I(Ipp16u *pSrcDst,
    Ipp16u *pTmp, Ipp32s len);
IppStatus fpk_dft_avx512_mic_ippsSortRadixDescend_32f_I(Ipp32f *pSrcDst,
    Ipp32f *pTmp, Ipp32s len);
IppStatus fpk_dft_avx512_mic_ippsSortRadixDescend_32s_I(Ipp32s *pSrcDst,
    Ipp32s *pTmp, Ipp32s len);
IppStatus fpk_dft_avx512_mic_ippsSortRadixDescend_32u_I(Ipp32u *pSrcDst,
    Ipp32u *pTmp, Ipp32s len);
IppStatus fpk_dft_avx512_mic_ippsSortRadixDescend_64f_I(Ipp64f *pSrcDst,
    Ipp64f *pTmp, Ipp32s len);
IppStatus fpk_dft_avx512_mic_ippsSortRadixDescend_8u_I(Ipp8u *pSrcDst,
    Ipp8u *pTmp, Ipp32s len);


dnnError_t fpk_dnn_sse2_AllocateBuffer_F32(void **pPtr, dnnLayout_t layout);
dnnError_t fpk_dnn_sse2_AllocateBuffer_F64(void **pPtr, dnnLayout_t layout);
dnnError_t fpk_dnn_sse2_BatchNormalizationCreateBackwardData_F32(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps);
dnnError_t fpk_dnn_sse2_BatchNormalizationCreateBackwardData_F64(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, double eps);
dnnError_t fpk_dnn_sse2_BatchNormalizationCreateBackwardScaleShift_F32(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps);
dnnError_t fpk_dnn_sse2_BatchNormalizationCreateBackwardScaleShift_F64(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, double eps);
dnnError_t fpk_dnn_sse2_BatchNormalizationCreateForward_F32(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps);
dnnError_t fpk_dnn_sse2_BatchNormalizationCreateForward_F64(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, double eps);
dnnError_t fpk_dnn_sse2_ConcatCreate_F32(dnnPrimitive_t* pConcat,
    dnnPrimitiveAttributes_t attributes, const size_t nSrcTensors,
    dnnLayout_t *src);
dnnError_t fpk_dnn_sse2_ConcatCreate_F64(dnnPrimitive_t* pConcat,
    dnnPrimitiveAttributes_t attributes, const size_t nSrcTensors,
    dnnLayout_t *src);
dnnError_t fpk_dnn_sse2_ConversionCreate_F32(dnnPrimitive_t* pConversion,
    const dnnLayout_t from, const dnnLayout_t to);
dnnError_t fpk_dnn_sse2_ConversionCreate_F64(dnnPrimitive_t* pConversion,
    const dnnLayout_t from, const dnnLayout_t to);
dnnError_t fpk_dnn_sse2_ConversionExecute_F32(dnnPrimitive_t conversion,
    void *from, void *to);
dnnError_t fpk_dnn_sse2_ConversionExecute_F64(dnnPrimitive_t conversion,
    void *from, void *to);
dnnError_t fpk_dnn_sse2_ConvolutionCreateBackwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t dstSize[]);
dnnError_t fpk_dnn_sse2_ConvolutionCreateBackwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t dstSize[]);
dnnError_t fpk_dnn_sse2_ConvolutionCreateBackwardData_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse2_ConvolutionCreateBackwardData_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse2_ConvolutionCreateBackwardFilter_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse2_ConvolutionCreateBackwardFilter_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse2_ConvolutionCreateForwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse2_ConvolutionCreateForwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse2_ConvolutionCreateForward_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse2_ConvolutionCreateForward_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse2_Delete_F32(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_sse2_Delete_F64(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_sse2_ExecuteAsync_F32(dnnPrimitive_t primitive,
    void *resources[]);
dnnError_t fpk_dnn_sse2_ExecuteAsync_F64(dnnPrimitive_t primitive,
    void *resources[]);
dnnError_t fpk_dnn_sse2_Execute_F32(dnnPrimitive_t primitive, void *resources[]);
dnnError_t fpk_dnn_sse2_Execute_F64(dnnPrimitive_t primitive, void *resources[]);
dnnError_t fpk_dnn_sse2_GroupsConvolutionCreateBackwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t dstSize[]);
dnnError_t fpk_dnn_sse2_GroupsConvolutionCreateBackwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t dstSize[]);
dnnError_t fpk_dnn_sse2_GroupsConvolutionCreateBackwardData_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse2_GroupsConvolutionCreateBackwardData_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse2_GroupsConvolutionCreateBackwardFilter_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse2_GroupsConvolutionCreateBackwardFilter_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse2_GroupsConvolutionCreateForwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse2_GroupsConvolutionCreateForwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse2_GroupsConvolutionCreateForward_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse2_GroupsConvolutionCreateForward_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse2_InnerProductCreateBackwardBias_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t dstSize[]);
dnnError_t fpk_dnn_sse2_InnerProductCreateBackwardBias_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t dstSize[]);
dnnError_t fpk_dnn_sse2_InnerProductCreateBackwardData_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_sse2_InnerProductCreateBackwardData_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_sse2_InnerProductCreateBackwardFilter_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_sse2_InnerProductCreateBackwardFilter_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_sse2_InnerProductCreateForwardBias_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_sse2_InnerProductCreateForwardBias_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_sse2_InnerProductCreateForward_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_sse2_InnerProductCreateForward_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_sse2_LRNCreateBackward_F32(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, size_t kernel_size, float alpha, float beta,
    float k);
dnnError_t fpk_dnn_sse2_LRNCreateBackward_F64(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, size_t kernel_size, double alpha, double beta,
    double k);
dnnError_t fpk_dnn_sse2_LRNCreateForward_F32(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    size_t kernel_size, float alpha, float beta, float k);
dnnError_t fpk_dnn_sse2_LRNCreateForward_F64(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    size_t kernel_size, double alpha, double beta, double k);
int fpk_dnn_sse2_LayoutCompare_F32(const dnnLayout_t l1, const dnnLayout_t l2);
int fpk_dnn_sse2_LayoutCompare_F64(const dnnLayout_t l1, const dnnLayout_t l2);
dnnError_t fpk_dnn_sse2_LayoutCreateFromPrimitive_F32(dnnLayout_t *pLayout,
    const dnnPrimitive_t primitive, dnnResourceType_t type);
dnnError_t fpk_dnn_sse2_LayoutCreateFromPrimitive_F64(dnnLayout_t *pLayout,
    const dnnPrimitive_t primitive, dnnResourceType_t type);
dnnError_t fpk_dnn_sse2_LayoutCreate_F32(dnnLayout_t *pLayout, size_t dimension,
    const size_t size[], const size_t strides[]);
dnnError_t fpk_dnn_sse2_LayoutCreate_F64(dnnLayout_t *pLayout, size_t dimension,
    const size_t size[], const size_t strides[]);
dnnError_t fpk_dnn_sse2_LayoutDelete_F32(dnnLayout_t layout);
dnnError_t fpk_dnn_sse2_LayoutDelete_F64(dnnLayout_t layout);
size_t fpk_dnn_sse2_LayoutGetMemorySize_F32(const dnnLayout_t layout);
size_t fpk_dnn_sse2_LayoutGetMemorySize_F64(const dnnLayout_t layout);
dnnError_t fpk_dnn_sse2_PoolingCreateBackward_F32(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse2_PoolingCreateBackward_F64(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse2_PoolingCreateForward_F32(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse2_PoolingCreateForward_F64(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_sse2_ReLUCreateBackward_F32(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, float negativeSlope);
dnnError_t fpk_dnn_sse2_ReLUCreateBackward_F64(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, double negativeSlope);
dnnError_t fpk_dnn_sse2_ReLUCreateForward_F32(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    float negativeSlope);
dnnError_t fpk_dnn_sse2_ReLUCreateForward_F64(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    double negativeSlope);
dnnError_t fpk_dnn_sse2_ReleaseBuffer_F32(void *ptr);
dnnError_t fpk_dnn_sse2_ReleaseBuffer_F64(void *ptr);
dnnError_t fpk_dnn_sse2_ScaleCreate_F32(dnnPrimitive_t *pScale,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    float alpha);
dnnError_t fpk_dnn_sse2_ScaleCreate_F64(dnnPrimitive_t *pScale,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    double alpha);
dnnError_t fpk_dnn_sse2_SplitCreate_F32(dnnPrimitive_t *pSplit,
    dnnPrimitiveAttributes_t attributes, const size_t nDstTensors,
    dnnLayout_t layout, size_t dstChannelSize[]);
dnnError_t fpk_dnn_sse2_SplitCreate_F64(dnnPrimitive_t *pSplit,
    dnnPrimitiveAttributes_t attributes, const size_t nDstTensors,
    dnnLayout_t layout, size_t dstChannelSize[]);
dnnError_t fpk_dnn_sse2_SumCreate_F32(dnnPrimitive_t *pSum,
    dnnPrimitiveAttributes_t attributes, const size_t nSummands,
    dnnLayout_t layout, float *coefficients);
dnnError_t fpk_dnn_sse2_SumCreate_F64(dnnPrimitive_t *pSum,
    dnnPrimitiveAttributes_t attributes, const size_t nSummands,
    dnnLayout_t layout, double *coefficients);
dnnError_t fpk_dnn_sse2_WaitFor_F32(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_sse2_WaitFor_F64(dnnPrimitive_t primitive);

dnnError_t fpk_dnn_ssse3_AllocateBuffer_F32(void **pPtr, dnnLayout_t layout);
dnnError_t fpk_dnn_ssse3_AllocateBuffer_F64(void **pPtr, dnnLayout_t layout);
dnnError_t fpk_dnn_ssse3_BatchNormalizationCreateBackwardData_F32(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps);
dnnError_t fpk_dnn_ssse3_BatchNormalizationCreateBackwardData_F64(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, double eps);
dnnError_t fpk_dnn_ssse3_BatchNormalizationCreateBackwardScaleShift_F32(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps);
dnnError_t fpk_dnn_ssse3_BatchNormalizationCreateBackwardScaleShift_F64(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, double eps);
dnnError_t fpk_dnn_ssse3_BatchNormalizationCreateForward_F32(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps);
dnnError_t fpk_dnn_ssse3_BatchNormalizationCreateForward_F64(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, double eps);
dnnError_t fpk_dnn_ssse3_ConcatCreate_F32(dnnPrimitive_t* pConcat,
    dnnPrimitiveAttributes_t attributes, const size_t nSrcTensors,
    dnnLayout_t *src);
dnnError_t fpk_dnn_ssse3_ConcatCreate_F64(dnnPrimitive_t* pConcat,
    dnnPrimitiveAttributes_t attributes, const size_t nSrcTensors,
    dnnLayout_t *src);
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
dnnError_t fpk_dnn_ssse3_InnerProductCreateBackwardBias_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t dstSize[]);
dnnError_t fpk_dnn_ssse3_InnerProductCreateBackwardBias_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t dstSize[]);
dnnError_t fpk_dnn_ssse3_InnerProductCreateBackwardData_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_ssse3_InnerProductCreateBackwardData_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_ssse3_InnerProductCreateBackwardFilter_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_ssse3_InnerProductCreateBackwardFilter_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_ssse3_InnerProductCreateForwardBias_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_ssse3_InnerProductCreateForwardBias_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_ssse3_InnerProductCreateForward_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_ssse3_InnerProductCreateForward_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
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
dnnError_t fpk_dnn_ssse3_ScaleCreate_F32(dnnPrimitive_t *pScale,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    float alpha);
dnnError_t fpk_dnn_ssse3_ScaleCreate_F64(dnnPrimitive_t *pScale,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    double alpha);
dnnError_t fpk_dnn_ssse3_SplitCreate_F32(dnnPrimitive_t *pSplit,
    dnnPrimitiveAttributes_t attributes, const size_t nDstTensors,
    dnnLayout_t layout, size_t dstChannelSize[]);
dnnError_t fpk_dnn_ssse3_SplitCreate_F64(dnnPrimitive_t *pSplit,
    dnnPrimitiveAttributes_t attributes, const size_t nDstTensors,
    dnnLayout_t layout, size_t dstChannelSize[]);
dnnError_t fpk_dnn_ssse3_SumCreate_F32(dnnPrimitive_t *pSum,
    dnnPrimitiveAttributes_t attributes, const size_t nSummands,
    dnnLayout_t layout, float *coefficients);
dnnError_t fpk_dnn_ssse3_SumCreate_F64(dnnPrimitive_t *pSum,
    dnnPrimitiveAttributes_t attributes, const size_t nSummands,
    dnnLayout_t layout, double *coefficients);
dnnError_t fpk_dnn_ssse3_WaitFor_F32(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_ssse3_WaitFor_F64(dnnPrimitive_t primitive);

dnnError_t fpk_dnn_sse42_AllocateBuffer_F32(void **pPtr, dnnLayout_t layout);
dnnError_t fpk_dnn_sse42_AllocateBuffer_F64(void **pPtr, dnnLayout_t layout);
dnnError_t fpk_dnn_sse42_BatchNormalizationCreateBackwardData_F32(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps);
dnnError_t fpk_dnn_sse42_BatchNormalizationCreateBackwardData_F64(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, double eps);
dnnError_t fpk_dnn_sse42_BatchNormalizationCreateBackwardScaleShift_F32(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps);
dnnError_t fpk_dnn_sse42_BatchNormalizationCreateBackwardScaleShift_F64(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, double eps);
dnnError_t fpk_dnn_sse42_BatchNormalizationCreateForward_F32(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps);
dnnError_t fpk_dnn_sse42_BatchNormalizationCreateForward_F64(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, double eps);
dnnError_t fpk_dnn_sse42_ConcatCreate_F32(dnnPrimitive_t* pConcat,
    dnnPrimitiveAttributes_t attributes, const size_t nSrcTensors,
    dnnLayout_t *src);
dnnError_t fpk_dnn_sse42_ConcatCreate_F64(dnnPrimitive_t* pConcat,
    dnnPrimitiveAttributes_t attributes, const size_t nSrcTensors,
    dnnLayout_t *src);
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
dnnError_t fpk_dnn_sse42_InnerProductCreateBackwardBias_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t dstSize[]);
dnnError_t fpk_dnn_sse42_InnerProductCreateBackwardBias_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t dstSize[]);
dnnError_t fpk_dnn_sse42_InnerProductCreateBackwardData_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_sse42_InnerProductCreateBackwardData_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_sse42_InnerProductCreateBackwardFilter_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_sse42_InnerProductCreateBackwardFilter_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_sse42_InnerProductCreateForwardBias_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_sse42_InnerProductCreateForwardBias_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_sse42_InnerProductCreateForward_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_sse42_InnerProductCreateForward_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
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
dnnError_t fpk_dnn_sse42_ScaleCreate_F32(dnnPrimitive_t *pScale,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    float alpha);
dnnError_t fpk_dnn_sse42_ScaleCreate_F64(dnnPrimitive_t *pScale,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    double alpha);
dnnError_t fpk_dnn_sse42_SplitCreate_F32(dnnPrimitive_t *pSplit,
    dnnPrimitiveAttributes_t attributes, const size_t nDstTensors,
    dnnLayout_t layout, size_t dstChannelSize[]);
dnnError_t fpk_dnn_sse42_SplitCreate_F64(dnnPrimitive_t *pSplit,
    dnnPrimitiveAttributes_t attributes, const size_t nDstTensors,
    dnnLayout_t layout, size_t dstChannelSize[]);
dnnError_t fpk_dnn_sse42_SumCreate_F32(dnnPrimitive_t *pSum,
    dnnPrimitiveAttributes_t attributes, const size_t nSummands,
    dnnLayout_t layout, float *coefficients);
dnnError_t fpk_dnn_sse42_SumCreate_F64(dnnPrimitive_t *pSum,
    dnnPrimitiveAttributes_t attributes, const size_t nSummands,
    dnnLayout_t layout, double *coefficients);
dnnError_t fpk_dnn_sse42_WaitFor_F32(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_sse42_WaitFor_F64(dnnPrimitive_t primitive);

dnnError_t fpk_dnn_avx_AllocateBuffer_F32(void **pPtr, dnnLayout_t layout);
dnnError_t fpk_dnn_avx_AllocateBuffer_F64(void **pPtr, dnnLayout_t layout);
dnnError_t fpk_dnn_avx_BatchNormalizationCreateBackwardData_F32(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps);
dnnError_t fpk_dnn_avx_BatchNormalizationCreateBackwardData_F64(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, double eps);
dnnError_t fpk_dnn_avx_BatchNormalizationCreateBackwardScaleShift_F32(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps);
dnnError_t fpk_dnn_avx_BatchNormalizationCreateBackwardScaleShift_F64(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, double eps);
dnnError_t fpk_dnn_avx_BatchNormalizationCreateForward_F32(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps);
dnnError_t fpk_dnn_avx_BatchNormalizationCreateForward_F64(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, double eps);
dnnError_t fpk_dnn_avx_ConcatCreate_F32(dnnPrimitive_t* pConcat,
    dnnPrimitiveAttributes_t attributes, const size_t nSrcTensors,
    dnnLayout_t *src);
dnnError_t fpk_dnn_avx_ConcatCreate_F64(dnnPrimitive_t* pConcat,
    dnnPrimitiveAttributes_t attributes, const size_t nSrcTensors,
    dnnLayout_t *src);
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
dnnError_t fpk_dnn_avx_InnerProductCreateBackwardBias_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t dstSize[]);
dnnError_t fpk_dnn_avx_InnerProductCreateBackwardBias_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t dstSize[]);
dnnError_t fpk_dnn_avx_InnerProductCreateBackwardData_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx_InnerProductCreateBackwardData_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx_InnerProductCreateBackwardFilter_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx_InnerProductCreateBackwardFilter_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx_InnerProductCreateForwardBias_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx_InnerProductCreateForwardBias_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx_InnerProductCreateForward_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx_InnerProductCreateForward_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
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
dnnError_t fpk_dnn_avx_ScaleCreate_F32(dnnPrimitive_t *pScale,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    float alpha);
dnnError_t fpk_dnn_avx_ScaleCreate_F64(dnnPrimitive_t *pScale,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    double alpha);
dnnError_t fpk_dnn_avx_SplitCreate_F32(dnnPrimitive_t *pSplit,
    dnnPrimitiveAttributes_t attributes, const size_t nDstTensors,
    dnnLayout_t layout, size_t dstChannelSize[]);
dnnError_t fpk_dnn_avx_SplitCreate_F64(dnnPrimitive_t *pSplit,
    dnnPrimitiveAttributes_t attributes, const size_t nDstTensors,
    dnnLayout_t layout, size_t dstChannelSize[]);
dnnError_t fpk_dnn_avx_SumCreate_F32(dnnPrimitive_t *pSum,
    dnnPrimitiveAttributes_t attributes, const size_t nSummands,
    dnnLayout_t layout, float *coefficients);
dnnError_t fpk_dnn_avx_SumCreate_F64(dnnPrimitive_t *pSum,
    dnnPrimitiveAttributes_t attributes, const size_t nSummands,
    dnnLayout_t layout, double *coefficients);
dnnError_t fpk_dnn_avx_WaitFor_F32(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_avx_WaitFor_F64(dnnPrimitive_t primitive);

dnnError_t fpk_dnn_avx2_AllocateBuffer_F32(void **pPtr, dnnLayout_t layout);
dnnError_t fpk_dnn_avx2_AllocateBuffer_F64(void **pPtr, dnnLayout_t layout);
dnnError_t fpk_dnn_avx2_BatchNormalizationCreateBackwardData_F32(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps);
dnnError_t fpk_dnn_avx2_BatchNormalizationCreateBackwardData_F64(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, double eps);
dnnError_t fpk_dnn_avx2_BatchNormalizationCreateBackwardScaleShift_F32(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps);
dnnError_t fpk_dnn_avx2_BatchNormalizationCreateBackwardScaleShift_F64(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, double eps);
dnnError_t fpk_dnn_avx2_BatchNormalizationCreateForward_F32(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps);
dnnError_t fpk_dnn_avx2_BatchNormalizationCreateForward_F64(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, double eps);
dnnError_t fpk_dnn_avx2_ConcatCreate_F32(dnnPrimitive_t* pConcat,
    dnnPrimitiveAttributes_t attributes, const size_t nSrcTensors,
    dnnLayout_t *src);
dnnError_t fpk_dnn_avx2_ConcatCreate_F64(dnnPrimitive_t* pConcat,
    dnnPrimitiveAttributes_t attributes, const size_t nSrcTensors,
    dnnLayout_t *src);
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
dnnError_t fpk_dnn_avx2_InnerProductCreateBackwardBias_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t dstSize[]);
dnnError_t fpk_dnn_avx2_InnerProductCreateBackwardBias_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t dstSize[]);
dnnError_t fpk_dnn_avx2_InnerProductCreateBackwardData_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx2_InnerProductCreateBackwardData_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx2_InnerProductCreateBackwardFilter_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx2_InnerProductCreateBackwardFilter_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx2_InnerProductCreateForwardBias_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx2_InnerProductCreateForwardBias_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx2_InnerProductCreateForward_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx2_InnerProductCreateForward_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
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
dnnError_t fpk_dnn_avx2_ScaleCreate_F32(dnnPrimitive_t *pScale,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    float alpha);
dnnError_t fpk_dnn_avx2_ScaleCreate_F64(dnnPrimitive_t *pScale,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    double alpha);
dnnError_t fpk_dnn_avx2_SplitCreate_F32(dnnPrimitive_t *pSplit,
    dnnPrimitiveAttributes_t attributes, const size_t nDstTensors,
    dnnLayout_t layout, size_t dstChannelSize[]);
dnnError_t fpk_dnn_avx2_SplitCreate_F64(dnnPrimitive_t *pSplit,
    dnnPrimitiveAttributes_t attributes, const size_t nDstTensors,
    dnnLayout_t layout, size_t dstChannelSize[]);
dnnError_t fpk_dnn_avx2_SumCreate_F32(dnnPrimitive_t *pSum,
    dnnPrimitiveAttributes_t attributes, const size_t nSummands,
    dnnLayout_t layout, float *coefficients);
dnnError_t fpk_dnn_avx2_SumCreate_F64(dnnPrimitive_t *pSum,
    dnnPrimitiveAttributes_t attributes, const size_t nSummands,
    dnnLayout_t layout, double *coefficients);
dnnError_t fpk_dnn_avx2_WaitFor_F32(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_avx2_WaitFor_F64(dnnPrimitive_t primitive);

dnnError_t fpk_dnn_avx512_AllocateBuffer_F32(void **pPtr, dnnLayout_t layout);
dnnError_t fpk_dnn_avx512_AllocateBuffer_F64(void **pPtr, dnnLayout_t layout);
dnnError_t fpk_dnn_avx512_BatchNormalizationCreateBackwardData_F32(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps);
dnnError_t fpk_dnn_avx512_BatchNormalizationCreateBackwardData_F64(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, double eps);
dnnError_t fpk_dnn_avx512_BatchNormalizationCreateBackwardScaleShift_F32(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps);
dnnError_t fpk_dnn_avx512_BatchNormalizationCreateBackwardScaleShift_F64(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, double eps);
dnnError_t fpk_dnn_avx512_BatchNormalizationCreateForward_F32(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps);
dnnError_t fpk_dnn_avx512_BatchNormalizationCreateForward_F64(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, double eps);
dnnError_t fpk_dnn_avx512_ConcatCreate_F32(dnnPrimitive_t* pConcat,
    dnnPrimitiveAttributes_t attributes, const size_t nSrcTensors,
    dnnLayout_t *src);
dnnError_t fpk_dnn_avx512_ConcatCreate_F64(dnnPrimitive_t* pConcat,
    dnnPrimitiveAttributes_t attributes, const size_t nSrcTensors,
    dnnLayout_t *src);
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
dnnError_t fpk_dnn_avx512_InnerProductCreateBackwardBias_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t dstSize[]);
dnnError_t fpk_dnn_avx512_InnerProductCreateBackwardBias_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t dstSize[]);
dnnError_t fpk_dnn_avx512_InnerProductCreateBackwardData_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx512_InnerProductCreateBackwardData_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx512_InnerProductCreateBackwardFilter_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx512_InnerProductCreateBackwardFilter_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx512_InnerProductCreateForwardBias_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx512_InnerProductCreateForwardBias_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx512_InnerProductCreateForward_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx512_InnerProductCreateForward_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
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
dnnError_t fpk_dnn_avx512_ScaleCreate_F32(dnnPrimitive_t *pScale,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    float alpha);
dnnError_t fpk_dnn_avx512_ScaleCreate_F64(dnnPrimitive_t *pScale,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    double alpha);
dnnError_t fpk_dnn_avx512_SplitCreate_F32(dnnPrimitive_t *pSplit,
    dnnPrimitiveAttributes_t attributes, const size_t nDstTensors,
    dnnLayout_t layout, size_t dstChannelSize[]);
dnnError_t fpk_dnn_avx512_SplitCreate_F64(dnnPrimitive_t *pSplit,
    dnnPrimitiveAttributes_t attributes, const size_t nDstTensors,
    dnnLayout_t layout, size_t dstChannelSize[]);
dnnError_t fpk_dnn_avx512_SumCreate_F32(dnnPrimitive_t *pSum,
    dnnPrimitiveAttributes_t attributes, const size_t nSummands,
    dnnLayout_t layout, float *coefficients);
dnnError_t fpk_dnn_avx512_SumCreate_F64(dnnPrimitive_t *pSum,
    dnnPrimitiveAttributes_t attributes, const size_t nSummands,
    dnnLayout_t layout, double *coefficients);
dnnError_t fpk_dnn_avx512_WaitFor_F32(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_avx512_WaitFor_F64(dnnPrimitive_t primitive);

dnnError_t fpk_dnn_avx512_mic_AllocateBuffer_F32(void **pPtr,
    dnnLayout_t layout);
dnnError_t fpk_dnn_avx512_mic_AllocateBuffer_F64(void **pPtr,
    dnnLayout_t layout);
dnnError_t fpk_dnn_avx512_mic_BatchNormalizationCreateBackwardData_F32(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps);
dnnError_t fpk_dnn_avx512_mic_BatchNormalizationCreateBackwardData_F64(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, double eps);
dnnError_t fpk_dnn_avx512_mic_BatchNormalizationCreateBackwardScaleShift_F32(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps);
dnnError_t fpk_dnn_avx512_mic_BatchNormalizationCreateBackwardScaleShift_F64(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, double eps);
dnnError_t fpk_dnn_avx512_mic_BatchNormalizationCreateForward_F32(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps);
dnnError_t fpk_dnn_avx512_mic_BatchNormalizationCreateForward_F64(
    dnnPrimitive_t* pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, double eps);
dnnError_t fpk_dnn_avx512_mic_ConcatCreate_F32(dnnPrimitive_t* pConcat,
    dnnPrimitiveAttributes_t attributes, const size_t nSrcTensors,
    dnnLayout_t *src);
dnnError_t fpk_dnn_avx512_mic_ConcatCreate_F64(dnnPrimitive_t* pConcat,
    dnnPrimitiveAttributes_t attributes, const size_t nSrcTensors,
    dnnLayout_t *src);
dnnError_t fpk_dnn_avx512_mic_ConversionCreate_F32(dnnPrimitive_t* pConversion,
    const dnnLayout_t from, const dnnLayout_t to);
dnnError_t fpk_dnn_avx512_mic_ConversionCreate_F64(dnnPrimitive_t* pConversion,
    const dnnLayout_t from, const dnnLayout_t to);
dnnError_t fpk_dnn_avx512_mic_ConversionExecute_F32(dnnPrimitive_t conversion,
    void *from, void *to);
dnnError_t fpk_dnn_avx512_mic_ConversionExecute_F64(dnnPrimitive_t conversion,
    void *from, void *to);
dnnError_t fpk_dnn_avx512_mic_ConvolutionCreateBackwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t dstSize[]);
dnnError_t fpk_dnn_avx512_mic_ConvolutionCreateBackwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t dstSize[]);
dnnError_t fpk_dnn_avx512_mic_ConvolutionCreateBackwardData_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_mic_ConvolutionCreateBackwardData_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_mic_ConvolutionCreateBackwardFilter_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_mic_ConvolutionCreateBackwardFilter_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_mic_ConvolutionCreateForwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_mic_ConvolutionCreateForwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_mic_ConvolutionCreateForward_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_mic_ConvolutionCreateForward_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t dimension, const size_t srcSize[],
    const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_mic_Delete_F32(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_avx512_mic_Delete_F64(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_avx512_mic_ExecuteAsync_F32(dnnPrimitive_t primitive,
    void *resources[]);
dnnError_t fpk_dnn_avx512_mic_ExecuteAsync_F64(dnnPrimitive_t primitive,
    void *resources[]);
dnnError_t fpk_dnn_avx512_mic_Execute_F32(dnnPrimitive_t primitive,
    void *resources[]);
dnnError_t fpk_dnn_avx512_mic_Execute_F64(dnnPrimitive_t primitive,
    void *resources[]);
dnnError_t fpk_dnn_avx512_mic_GroupsConvolutionCreateBackwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t dstSize[]);
dnnError_t fpk_dnn_avx512_mic_GroupsConvolutionCreateBackwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t dstSize[]);
dnnError_t fpk_dnn_avx512_mic_GroupsConvolutionCreateBackwardData_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_mic_GroupsConvolutionCreateBackwardData_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_mic_GroupsConvolutionCreateBackwardFilter_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_mic_GroupsConvolutionCreateBackwardFilter_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_mic_GroupsConvolutionCreateForwardBias_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_mic_GroupsConvolutionCreateForwardBias_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_mic_GroupsConvolutionCreateForward_F32(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_mic_GroupsConvolutionCreateForward_F64(
    dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_mic_InnerProductCreateBackwardBias_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t dstSize[]);
dnnError_t fpk_dnn_avx512_mic_InnerProductCreateBackwardBias_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t dstSize[]);
dnnError_t fpk_dnn_avx512_mic_InnerProductCreateBackwardData_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx512_mic_InnerProductCreateBackwardData_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx512_mic_InnerProductCreateBackwardFilter_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx512_mic_InnerProductCreateBackwardFilter_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx512_mic_InnerProductCreateForwardBias_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx512_mic_InnerProductCreateForwardBias_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx512_mic_InnerProductCreateForward_F32(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx512_mic_InnerProductCreateForward_F64(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimensions, const size_t srcSize[], size_t outputChannels);
dnnError_t fpk_dnn_avx512_mic_LRNCreateBackward_F32(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, size_t kernel_size, float alpha, float beta,
    float k);
dnnError_t fpk_dnn_avx512_mic_LRNCreateBackward_F64(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, size_t kernel_size, double alpha, double beta,
    double k);
dnnError_t fpk_dnn_avx512_mic_LRNCreateForward_F32(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    size_t kernel_size, float alpha, float beta, float k);
dnnError_t fpk_dnn_avx512_mic_LRNCreateForward_F64(dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    size_t kernel_size, double alpha, double beta, double k);
int fpk_dnn_avx512_mic_LayoutCompare_F32(const dnnLayout_t l1,
    const dnnLayout_t l2);
int fpk_dnn_avx512_mic_LayoutCompare_F64(const dnnLayout_t l1,
    const dnnLayout_t l2);
dnnError_t fpk_dnn_avx512_mic_LayoutCreateFromPrimitive_F32(dnnLayout_t *pLayout,
    const dnnPrimitive_t primitive, dnnResourceType_t type);
dnnError_t fpk_dnn_avx512_mic_LayoutCreateFromPrimitive_F64(dnnLayout_t *pLayout,
    const dnnPrimitive_t primitive, dnnResourceType_t type);
dnnError_t fpk_dnn_avx512_mic_LayoutCreate_F32(dnnLayout_t *pLayout,
    size_t dimension, const size_t size[], const size_t strides[]);
dnnError_t fpk_dnn_avx512_mic_LayoutCreate_F64(dnnLayout_t *pLayout,
    size_t dimension, const size_t size[], const size_t strides[]);
dnnError_t fpk_dnn_avx512_mic_LayoutDelete_F32(dnnLayout_t layout);
dnnError_t fpk_dnn_avx512_mic_LayoutDelete_F64(dnnLayout_t layout);
size_t fpk_dnn_avx512_mic_LayoutGetMemorySize_F32(const dnnLayout_t layout);
size_t fpk_dnn_avx512_mic_LayoutGetMemorySize_F64(const dnnLayout_t layout);
dnnError_t fpk_dnn_avx512_mic_PoolingCreateBackward_F32(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_mic_PoolingCreateBackward_F64(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_mic_PoolingCreateForward_F32(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_mic_PoolingCreateForward_F64(dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t op,
    const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType);
dnnError_t fpk_dnn_avx512_mic_ReLUCreateBackward_F32(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, float negativeSlope);
dnnError_t fpk_dnn_avx512_mic_ReLUCreateBackward_F64(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout, double negativeSlope);
dnnError_t fpk_dnn_avx512_mic_ReLUCreateForward_F32(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    float negativeSlope);
dnnError_t fpk_dnn_avx512_mic_ReLUCreateForward_F64(dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    double negativeSlope);
dnnError_t fpk_dnn_avx512_mic_ReleaseBuffer_F32(void *ptr);
dnnError_t fpk_dnn_avx512_mic_ReleaseBuffer_F64(void *ptr);
dnnError_t fpk_dnn_avx512_mic_ScaleCreate_F32(dnnPrimitive_t *pScale,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    float alpha);
dnnError_t fpk_dnn_avx512_mic_ScaleCreate_F64(dnnPrimitive_t *pScale,
    dnnPrimitiveAttributes_t attributes, const dnnLayout_t dataLayout,
    double alpha);
dnnError_t fpk_dnn_avx512_mic_SplitCreate_F32(dnnPrimitive_t *pSplit,
    dnnPrimitiveAttributes_t attributes, const size_t nDstTensors,
    dnnLayout_t layout, size_t dstChannelSize[]);
dnnError_t fpk_dnn_avx512_mic_SplitCreate_F64(dnnPrimitive_t *pSplit,
    dnnPrimitiveAttributes_t attributes, const size_t nDstTensors,
    dnnLayout_t layout, size_t dstChannelSize[]);
dnnError_t fpk_dnn_avx512_mic_SumCreate_F32(dnnPrimitive_t *pSum,
    dnnPrimitiveAttributes_t attributes, const size_t nSummands,
    dnnLayout_t layout, float *coefficients);
dnnError_t fpk_dnn_avx512_mic_SumCreate_F64(dnnPrimitive_t *pSum,
    dnnPrimitiveAttributes_t attributes, const size_t nSummands,
    dnnLayout_t layout, double *coefficients);
dnnError_t fpk_dnn_avx512_mic_WaitFor_F32(dnnPrimitive_t primitive);
dnnError_t fpk_dnn_avx512_mic_WaitFor_F64(dnnPrimitive_t primitive);


void fpk_feast_sse2_cfeast_hbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex8* a, const DAAL_INT* lda, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_sse2_cfeast_hbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex8* a, const DAAL_INT* lda,
    const DAAL_INT* klb, const MKL_Complex8* b, const DAAL_INT* ldb, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_sse2_cfeast_hcsrev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* sa, const DAAL_INT* isa, const DAAL_INT* jsa, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_sse2_cfeast_hcsrgv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* sa, const DAAL_INT* isa, const DAAL_INT* jsa,
    const MKL_Complex8* sb, const DAAL_INT* isb, const DAAL_INT* jsb, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_sse2_cfeast_heev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* a, const DAAL_INT* lda, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    MKL_Complex8* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_sse2_cfeast_hegv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* a, const DAAL_INT* lda, const MKL_Complex8* b,
    const DAAL_INT* ldb, DAAL_INT* fpm, float* epsout, DAAL_INT* loop,
    const float* emin, const float* emax, DAAL_INT* m0, float* e, MKL_Complex8* x,
    DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_sse2_cfeast_hrci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex8* ze, MKL_Complex8* work, MKL_Complex8* workc, MKL_Complex8* zaq,
    MKL_Complex8* zsq, DAAL_INT* fpm, float* epsout, DAAL_INT* loop,
    const float* emin, const float* emax, DAAL_INT* m0, float* lambda,
    MKL_Complex8* q, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_sse2_dfeast_sbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const double* a, const DAAL_INT* lda, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_sse2_dfeast_sbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const double* a, const DAAL_INT* lda, const DAAL_INT* klb,
    const double* b, const DAAL_INT* ldb, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_sse2_dfeast_scsrev(const char* uplo, const DAAL_INT* n,
    const double* sa, const DAAL_INT* isa, const DAAL_INT* jsa, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_sse2_dfeast_scsrgv(const char* uplo, const DAAL_INT* n,
    const double* sa, const DAAL_INT* isa, const DAAL_INT* jsa, const double* sb,
    const DAAL_INT* isb, const DAAL_INT* jsb, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_sse2_dfeast_srci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex16* ze, double* work, MKL_Complex16* workc, double* aq,
    double* sq, DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* lambda, double* q, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_sse2_dfeast_syev(const char* uplo, const DAAL_INT* n,
    const double* a, const DAAL_INT* lda, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_sse2_dfeast_sygv(const char* uplo, const DAAL_INT* n,
    const double* a, const DAAL_INT* lda, const double* b, const DAAL_INT* ldb,
    DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* e, double* x, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_sse2_feastinit(DAAL_INT* fpm);
void fpk_feast_sse2_sfeast_sbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const float* a, const DAAL_INT* lda, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_sse2_sfeast_sbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const float* a, const DAAL_INT* lda, const DAAL_INT* klb,
    const float* b, const DAAL_INT* ldb, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_sse2_sfeast_scsrev(const char* uplo, const DAAL_INT* n,
    const float* sa, const DAAL_INT* isa, const DAAL_INT* jsa, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_sse2_sfeast_scsrgv(const char* uplo, const DAAL_INT* n,
    const float* sa, const DAAL_INT* isa, const DAAL_INT* jsa, const float* sb,
    const DAAL_INT* isb, const DAAL_INT* jsb, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_sse2_sfeast_srci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex8* ze, float* work, MKL_Complex8* workc, float* aq, float* sq,
    DAAL_INT* fpm, float* epsout, DAAL_INT* loop, const float* emin,
    const float* emax, DAAL_INT* m0, float* lambda, float* q, DAAL_INT* m,
    float* res, DAAL_INT* info);
void fpk_feast_sse2_sfeast_syev(const char* uplo, const DAAL_INT* n,
    const float* a, const DAAL_INT* lda, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_sse2_sfeast_sygv(const char* uplo, const DAAL_INT* n,
    const float* a, const DAAL_INT* lda, const float* b, const DAAL_INT* ldb,
    DAAL_INT* fpm, float* epsout, DAAL_INT* loop, const float* emin,
    const float* emax, DAAL_INT* m0, float* e, float* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_sse2_zfeast_hbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex16* a, const DAAL_INT* lda, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m, double* res,
    DAAL_INT* info);
void fpk_feast_sse2_zfeast_hbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex16* a, const DAAL_INT* lda,
    const DAAL_INT* klb, const MKL_Complex16* b, const DAAL_INT* ldb, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m, double* res,
    DAAL_INT* info);
void fpk_feast_sse2_zfeast_hcsrev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* sa, const DAAL_INT* isa, const DAAL_INT* jsa,
    DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_sse2_zfeast_hcsrgv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* sa, const DAAL_INT* isa, const DAAL_INT* jsa,
    const MKL_Complex16* sb, const DAAL_INT* isb, const DAAL_INT* jsb,
    DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_sse2_zfeast_heev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* a, const DAAL_INT* lda, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, MKL_Complex16* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_sse2_zfeast_hegv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* a, const DAAL_INT* lda, const MKL_Complex16* b,
    const DAAL_INT* ldb, DAAL_INT* fpm, double* epsout, DAAL_INT* loop,
    const double* emin, const double* emax, DAAL_INT* m0, double* e,
    MKL_Complex16* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_sse2_zfeast_hrci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex16* ze, MKL_Complex16* work, MKL_Complex16* workc,
    MKL_Complex16* zaq, MKL_Complex16* zsq, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* lambda, MKL_Complex16* q, DAAL_INT* m, double* res, DAAL_INT* info);

void fpk_feast_ssse3_cfeast_hbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex8* a, const DAAL_INT* lda, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_ssse3_cfeast_hbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex8* a, const DAAL_INT* lda,
    const DAAL_INT* klb, const MKL_Complex8* b, const DAAL_INT* ldb, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_ssse3_cfeast_hcsrev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* sa, const DAAL_INT* isa, const DAAL_INT* jsa, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_ssse3_cfeast_hcsrgv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* sa, const DAAL_INT* isa, const DAAL_INT* jsa,
    const MKL_Complex8* sb, const DAAL_INT* isb, const DAAL_INT* jsb, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_ssse3_cfeast_heev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* a, const DAAL_INT* lda, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    MKL_Complex8* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_ssse3_cfeast_hegv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* a, const DAAL_INT* lda, const MKL_Complex8* b,
    const DAAL_INT* ldb, DAAL_INT* fpm, float* epsout, DAAL_INT* loop,
    const float* emin, const float* emax, DAAL_INT* m0, float* e, MKL_Complex8* x,
    DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_ssse3_cfeast_hrci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex8* ze, MKL_Complex8* work, MKL_Complex8* workc, MKL_Complex8* zaq,
    MKL_Complex8* zsq, DAAL_INT* fpm, float* epsout, DAAL_INT* loop,
    const float* emin, const float* emax, DAAL_INT* m0, float* lambda,
    MKL_Complex8* q, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_ssse3_dfeast_sbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const double* a, const DAAL_INT* lda, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_ssse3_dfeast_sbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const double* a, const DAAL_INT* lda, const DAAL_INT* klb,
    const double* b, const DAAL_INT* ldb, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_ssse3_dfeast_scsrev(const char* uplo, const DAAL_INT* n,
    const double* sa, const DAAL_INT* isa, const DAAL_INT* jsa, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_ssse3_dfeast_scsrgv(const char* uplo, const DAAL_INT* n,
    const double* sa, const DAAL_INT* isa, const DAAL_INT* jsa, const double* sb,
    const DAAL_INT* isb, const DAAL_INT* jsb, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_ssse3_dfeast_srci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex16* ze, double* work, MKL_Complex16* workc, double* aq,
    double* sq, DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* lambda, double* q, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_ssse3_dfeast_syev(const char* uplo, const DAAL_INT* n,
    const double* a, const DAAL_INT* lda, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_ssse3_dfeast_sygv(const char* uplo, const DAAL_INT* n,
    const double* a, const DAAL_INT* lda, const double* b, const DAAL_INT* ldb,
    DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* e, double* x, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_ssse3_feastinit(DAAL_INT* fpm);
void fpk_feast_ssse3_sfeast_sbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const float* a, const DAAL_INT* lda, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_ssse3_sfeast_sbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const float* a, const DAAL_INT* lda, const DAAL_INT* klb,
    const float* b, const DAAL_INT* ldb, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_ssse3_sfeast_scsrev(const char* uplo, const DAAL_INT* n,
    const float* sa, const DAAL_INT* isa, const DAAL_INT* jsa, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_ssse3_sfeast_scsrgv(const char* uplo, const DAAL_INT* n,
    const float* sa, const DAAL_INT* isa, const DAAL_INT* jsa, const float* sb,
    const DAAL_INT* isb, const DAAL_INT* jsb, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_ssse3_sfeast_srci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex8* ze, float* work, MKL_Complex8* workc, float* aq, float* sq,
    DAAL_INT* fpm, float* epsout, DAAL_INT* loop, const float* emin,
    const float* emax, DAAL_INT* m0, float* lambda, float* q, DAAL_INT* m,
    float* res, DAAL_INT* info);
void fpk_feast_ssse3_sfeast_syev(const char* uplo, const DAAL_INT* n,
    const float* a, const DAAL_INT* lda, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_ssse3_sfeast_sygv(const char* uplo, const DAAL_INT* n,
    const float* a, const DAAL_INT* lda, const float* b, const DAAL_INT* ldb,
    DAAL_INT* fpm, float* epsout, DAAL_INT* loop, const float* emin,
    const float* emax, DAAL_INT* m0, float* e, float* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_ssse3_zfeast_hbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex16* a, const DAAL_INT* lda, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m, double* res,
    DAAL_INT* info);
void fpk_feast_ssse3_zfeast_hbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex16* a, const DAAL_INT* lda,
    const DAAL_INT* klb, const MKL_Complex16* b, const DAAL_INT* ldb, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m, double* res,
    DAAL_INT* info);
void fpk_feast_ssse3_zfeast_hcsrev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* sa, const DAAL_INT* isa, const DAAL_INT* jsa,
    DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_ssse3_zfeast_hcsrgv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* sa, const DAAL_INT* isa, const DAAL_INT* jsa,
    const MKL_Complex16* sb, const DAAL_INT* isb, const DAAL_INT* jsb,
    DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_ssse3_zfeast_heev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* a, const DAAL_INT* lda, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, MKL_Complex16* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_ssse3_zfeast_hegv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* a, const DAAL_INT* lda, const MKL_Complex16* b,
    const DAAL_INT* ldb, DAAL_INT* fpm, double* epsout, DAAL_INT* loop,
    const double* emin, const double* emax, DAAL_INT* m0, double* e,
    MKL_Complex16* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_ssse3_zfeast_hrci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex16* ze, MKL_Complex16* work, MKL_Complex16* workc,
    MKL_Complex16* zaq, MKL_Complex16* zsq, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* lambda, MKL_Complex16* q, DAAL_INT* m, double* res, DAAL_INT* info);

void fpk_feast_sse42_cfeast_hbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex8* a, const DAAL_INT* lda, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_sse42_cfeast_hbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex8* a, const DAAL_INT* lda,
    const DAAL_INT* klb, const MKL_Complex8* b, const DAAL_INT* ldb, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_sse42_cfeast_hcsrev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* sa, const DAAL_INT* isa, const DAAL_INT* jsa, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_sse42_cfeast_hcsrgv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* sa, const DAAL_INT* isa, const DAAL_INT* jsa,
    const MKL_Complex8* sb, const DAAL_INT* isb, const DAAL_INT* jsb, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_sse42_cfeast_heev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* a, const DAAL_INT* lda, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    MKL_Complex8* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_sse42_cfeast_hegv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* a, const DAAL_INT* lda, const MKL_Complex8* b,
    const DAAL_INT* ldb, DAAL_INT* fpm, float* epsout, DAAL_INT* loop,
    const float* emin, const float* emax, DAAL_INT* m0, float* e, MKL_Complex8* x,
    DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_sse42_cfeast_hrci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex8* ze, MKL_Complex8* work, MKL_Complex8* workc, MKL_Complex8* zaq,
    MKL_Complex8* zsq, DAAL_INT* fpm, float* epsout, DAAL_INT* loop,
    const float* emin, const float* emax, DAAL_INT* m0, float* lambda,
    MKL_Complex8* q, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_sse42_dfeast_sbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const double* a, const DAAL_INT* lda, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_sse42_dfeast_sbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const double* a, const DAAL_INT* lda, const DAAL_INT* klb,
    const double* b, const DAAL_INT* ldb, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_sse42_dfeast_scsrev(const char* uplo, const DAAL_INT* n,
    const double* sa, const DAAL_INT* isa, const DAAL_INT* jsa, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_sse42_dfeast_scsrgv(const char* uplo, const DAAL_INT* n,
    const double* sa, const DAAL_INT* isa, const DAAL_INT* jsa, const double* sb,
    const DAAL_INT* isb, const DAAL_INT* jsb, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_sse42_dfeast_srci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex16* ze, double* work, MKL_Complex16* workc, double* aq,
    double* sq, DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* lambda, double* q, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_sse42_dfeast_syev(const char* uplo, const DAAL_INT* n,
    const double* a, const DAAL_INT* lda, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_sse42_dfeast_sygv(const char* uplo, const DAAL_INT* n,
    const double* a, const DAAL_INT* lda, const double* b, const DAAL_INT* ldb,
    DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* e, double* x, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_sse42_feastinit(DAAL_INT* fpm);
void fpk_feast_sse42_sfeast_sbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const float* a, const DAAL_INT* lda, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_sse42_sfeast_sbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const float* a, const DAAL_INT* lda, const DAAL_INT* klb,
    const float* b, const DAAL_INT* ldb, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_sse42_sfeast_scsrev(const char* uplo, const DAAL_INT* n,
    const float* sa, const DAAL_INT* isa, const DAAL_INT* jsa, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_sse42_sfeast_scsrgv(const char* uplo, const DAAL_INT* n,
    const float* sa, const DAAL_INT* isa, const DAAL_INT* jsa, const float* sb,
    const DAAL_INT* isb, const DAAL_INT* jsb, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_sse42_sfeast_srci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex8* ze, float* work, MKL_Complex8* workc, float* aq, float* sq,
    DAAL_INT* fpm, float* epsout, DAAL_INT* loop, const float* emin,
    const float* emax, DAAL_INT* m0, float* lambda, float* q, DAAL_INT* m,
    float* res, DAAL_INT* info);
void fpk_feast_sse42_sfeast_syev(const char* uplo, const DAAL_INT* n,
    const float* a, const DAAL_INT* lda, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_sse42_sfeast_sygv(const char* uplo, const DAAL_INT* n,
    const float* a, const DAAL_INT* lda, const float* b, const DAAL_INT* ldb,
    DAAL_INT* fpm, float* epsout, DAAL_INT* loop, const float* emin,
    const float* emax, DAAL_INT* m0, float* e, float* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_sse42_zfeast_hbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex16* a, const DAAL_INT* lda, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m, double* res,
    DAAL_INT* info);
void fpk_feast_sse42_zfeast_hbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex16* a, const DAAL_INT* lda,
    const DAAL_INT* klb, const MKL_Complex16* b, const DAAL_INT* ldb, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m, double* res,
    DAAL_INT* info);
void fpk_feast_sse42_zfeast_hcsrev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* sa, const DAAL_INT* isa, const DAAL_INT* jsa,
    DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_sse42_zfeast_hcsrgv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* sa, const DAAL_INT* isa, const DAAL_INT* jsa,
    const MKL_Complex16* sb, const DAAL_INT* isb, const DAAL_INT* jsb,
    DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_sse42_zfeast_heev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* a, const DAAL_INT* lda, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, MKL_Complex16* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_sse42_zfeast_hegv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* a, const DAAL_INT* lda, const MKL_Complex16* b,
    const DAAL_INT* ldb, DAAL_INT* fpm, double* epsout, DAAL_INT* loop,
    const double* emin, const double* emax, DAAL_INT* m0, double* e,
    MKL_Complex16* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_sse42_zfeast_hrci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex16* ze, MKL_Complex16* work, MKL_Complex16* workc,
    MKL_Complex16* zaq, MKL_Complex16* zsq, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* lambda, MKL_Complex16* q, DAAL_INT* m, double* res, DAAL_INT* info);

void fpk_feast_avx_cfeast_hbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex8* a, const DAAL_INT* lda, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_avx_cfeast_hbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex8* a, const DAAL_INT* lda,
    const DAAL_INT* klb, const MKL_Complex8* b, const DAAL_INT* ldb, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_avx_cfeast_hcsrev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* sa, const DAAL_INT* isa, const DAAL_INT* jsa, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_avx_cfeast_hcsrgv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* sa, const DAAL_INT* isa, const DAAL_INT* jsa,
    const MKL_Complex8* sb, const DAAL_INT* isb, const DAAL_INT* jsb, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_avx_cfeast_heev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* a, const DAAL_INT* lda, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    MKL_Complex8* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx_cfeast_hegv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* a, const DAAL_INT* lda, const MKL_Complex8* b,
    const DAAL_INT* ldb, DAAL_INT* fpm, float* epsout, DAAL_INT* loop,
    const float* emin, const float* emax, DAAL_INT* m0, float* e, MKL_Complex8* x,
    DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx_cfeast_hrci(DAAL_INT* ijob, const DAAL_INT* n, MKL_Complex8* ze,
    MKL_Complex8* work, MKL_Complex8* workc, MKL_Complex8* zaq,
    MKL_Complex8* zsq, DAAL_INT* fpm, float* epsout, DAAL_INT* loop,
    const float* emin, const float* emax, DAAL_INT* m0, float* lambda,
    MKL_Complex8* q, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx_dfeast_sbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const double* a, const DAAL_INT* lda, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx_dfeast_sbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const double* a, const DAAL_INT* lda, const DAAL_INT* klb,
    const double* b, const DAAL_INT* ldb, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx_dfeast_scsrev(const char* uplo, const DAAL_INT* n,
    const double* sa, const DAAL_INT* isa, const DAAL_INT* jsa, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx_dfeast_scsrgv(const char* uplo, const DAAL_INT* n,
    const double* sa, const DAAL_INT* isa, const DAAL_INT* jsa, const double* sb,
    const DAAL_INT* isb, const DAAL_INT* jsb, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx_dfeast_srci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex16* ze, double* work, MKL_Complex16* workc, double* aq,
    double* sq, DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* lambda, double* q, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_avx_dfeast_syev(const char* uplo, const DAAL_INT* n,
    const double* a, const DAAL_INT* lda, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx_dfeast_sygv(const char* uplo, const DAAL_INT* n,
    const double* a, const DAAL_INT* lda, const double* b, const DAAL_INT* ldb,
    DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* e, double* x, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_avx_feastinit(DAAL_INT* fpm);
void fpk_feast_avx_sfeast_sbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const float* a, const DAAL_INT* lda, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx_sfeast_sbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const float* a, const DAAL_INT* lda, const DAAL_INT* klb,
    const float* b, const DAAL_INT* ldb, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx_sfeast_scsrev(const char* uplo, const DAAL_INT* n,
    const float* sa, const DAAL_INT* isa, const DAAL_INT* jsa, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx_sfeast_scsrgv(const char* uplo, const DAAL_INT* n,
    const float* sa, const DAAL_INT* isa, const DAAL_INT* jsa, const float* sb,
    const DAAL_INT* isb, const DAAL_INT* jsb, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx_sfeast_srci(DAAL_INT* ijob, const DAAL_INT* n, MKL_Complex8* ze,
    float* work, MKL_Complex8* workc, float* aq, float* sq, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* lambda, float* q, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx_sfeast_syev(const char* uplo, const DAAL_INT* n,
    const float* a, const DAAL_INT* lda, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx_sfeast_sygv(const char* uplo, const DAAL_INT* n,
    const float* a, const DAAL_INT* lda, const float* b, const DAAL_INT* ldb,
    DAAL_INT* fpm, float* epsout, DAAL_INT* loop, const float* emin,
    const float* emax, DAAL_INT* m0, float* e, float* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_avx_zfeast_hbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex16* a, const DAAL_INT* lda, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m, double* res,
    DAAL_INT* info);
void fpk_feast_avx_zfeast_hbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex16* a, const DAAL_INT* lda,
    const DAAL_INT* klb, const MKL_Complex16* b, const DAAL_INT* ldb, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m, double* res,
    DAAL_INT* info);
void fpk_feast_avx_zfeast_hcsrev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* sa, const DAAL_INT* isa, const DAAL_INT* jsa,
    DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_avx_zfeast_hcsrgv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* sa, const DAAL_INT* isa, const DAAL_INT* jsa,
    const MKL_Complex16* sb, const DAAL_INT* isb, const DAAL_INT* jsb,
    DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_avx_zfeast_heev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* a, const DAAL_INT* lda, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, MKL_Complex16* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx_zfeast_hegv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* a, const DAAL_INT* lda, const MKL_Complex16* b,
    const DAAL_INT* ldb, DAAL_INT* fpm, double* epsout, DAAL_INT* loop,
    const double* emin, const double* emax, DAAL_INT* m0, double* e,
    MKL_Complex16* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx_zfeast_hrci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex16* ze, MKL_Complex16* work, MKL_Complex16* workc,
    MKL_Complex16* zaq, MKL_Complex16* zsq, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* lambda, MKL_Complex16* q, DAAL_INT* m, double* res, DAAL_INT* info);

void fpk_feast_avx2_cfeast_hbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex8* a, const DAAL_INT* lda, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_avx2_cfeast_hbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex8* a, const DAAL_INT* lda,
    const DAAL_INT* klb, const MKL_Complex8* b, const DAAL_INT* ldb, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_avx2_cfeast_hcsrev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* sa, const DAAL_INT* isa, const DAAL_INT* jsa, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_avx2_cfeast_hcsrgv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* sa, const DAAL_INT* isa, const DAAL_INT* jsa,
    const MKL_Complex8* sb, const DAAL_INT* isb, const DAAL_INT* jsb, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_avx2_cfeast_heev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* a, const DAAL_INT* lda, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    MKL_Complex8* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx2_cfeast_hegv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* a, const DAAL_INT* lda, const MKL_Complex8* b,
    const DAAL_INT* ldb, DAAL_INT* fpm, float* epsout, DAAL_INT* loop,
    const float* emin, const float* emax, DAAL_INT* m0, float* e, MKL_Complex8* x,
    DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx2_cfeast_hrci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex8* ze, MKL_Complex8* work, MKL_Complex8* workc, MKL_Complex8* zaq,
    MKL_Complex8* zsq, DAAL_INT* fpm, float* epsout, DAAL_INT* loop,
    const float* emin, const float* emax, DAAL_INT* m0, float* lambda,
    MKL_Complex8* q, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx2_dfeast_sbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const double* a, const DAAL_INT* lda, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx2_dfeast_sbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const double* a, const DAAL_INT* lda, const DAAL_INT* klb,
    const double* b, const DAAL_INT* ldb, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx2_dfeast_scsrev(const char* uplo, const DAAL_INT* n,
    const double* sa, const DAAL_INT* isa, const DAAL_INT* jsa, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx2_dfeast_scsrgv(const char* uplo, const DAAL_INT* n,
    const double* sa, const DAAL_INT* isa, const DAAL_INT* jsa, const double* sb,
    const DAAL_INT* isb, const DAAL_INT* jsb, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx2_dfeast_srci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex16* ze, double* work, MKL_Complex16* workc, double* aq,
    double* sq, DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* lambda, double* q, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_avx2_dfeast_syev(const char* uplo, const DAAL_INT* n,
    const double* a, const DAAL_INT* lda, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx2_dfeast_sygv(const char* uplo, const DAAL_INT* n,
    const double* a, const DAAL_INT* lda, const double* b, const DAAL_INT* ldb,
    DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* e, double* x, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_avx2_feastinit(DAAL_INT* fpm);
void fpk_feast_avx2_sfeast_sbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const float* a, const DAAL_INT* lda, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx2_sfeast_sbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const float* a, const DAAL_INT* lda, const DAAL_INT* klb,
    const float* b, const DAAL_INT* ldb, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx2_sfeast_scsrev(const char* uplo, const DAAL_INT* n,
    const float* sa, const DAAL_INT* isa, const DAAL_INT* jsa, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx2_sfeast_scsrgv(const char* uplo, const DAAL_INT* n,
    const float* sa, const DAAL_INT* isa, const DAAL_INT* jsa, const float* sb,
    const DAAL_INT* isb, const DAAL_INT* jsb, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx2_sfeast_srci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex8* ze, float* work, MKL_Complex8* workc, float* aq, float* sq,
    DAAL_INT* fpm, float* epsout, DAAL_INT* loop, const float* emin,
    const float* emax, DAAL_INT* m0, float* lambda, float* q, DAAL_INT* m,
    float* res, DAAL_INT* info);
void fpk_feast_avx2_sfeast_syev(const char* uplo, const DAAL_INT* n,
    const float* a, const DAAL_INT* lda, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx2_sfeast_sygv(const char* uplo, const DAAL_INT* n,
    const float* a, const DAAL_INT* lda, const float* b, const DAAL_INT* ldb,
    DAAL_INT* fpm, float* epsout, DAAL_INT* loop, const float* emin,
    const float* emax, DAAL_INT* m0, float* e, float* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_avx2_zfeast_hbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex16* a, const DAAL_INT* lda, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m, double* res,
    DAAL_INT* info);
void fpk_feast_avx2_zfeast_hbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex16* a, const DAAL_INT* lda,
    const DAAL_INT* klb, const MKL_Complex16* b, const DAAL_INT* ldb, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m, double* res,
    DAAL_INT* info);
void fpk_feast_avx2_zfeast_hcsrev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* sa, const DAAL_INT* isa, const DAAL_INT* jsa,
    DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_avx2_zfeast_hcsrgv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* sa, const DAAL_INT* isa, const DAAL_INT* jsa,
    const MKL_Complex16* sb, const DAAL_INT* isb, const DAAL_INT* jsb,
    DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_avx2_zfeast_heev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* a, const DAAL_INT* lda, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, MKL_Complex16* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx2_zfeast_hegv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* a, const DAAL_INT* lda, const MKL_Complex16* b,
    const DAAL_INT* ldb, DAAL_INT* fpm, double* epsout, DAAL_INT* loop,
    const double* emin, const double* emax, DAAL_INT* m0, double* e,
    MKL_Complex16* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx2_zfeast_hrci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex16* ze, MKL_Complex16* work, MKL_Complex16* workc,
    MKL_Complex16* zaq, MKL_Complex16* zsq, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* lambda, MKL_Complex16* q, DAAL_INT* m, double* res, DAAL_INT* info);

void fpk_feast_avx512_cfeast_hbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex8* a, const DAAL_INT* lda, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_avx512_cfeast_hbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex8* a, const DAAL_INT* lda,
    const DAAL_INT* klb, const MKL_Complex8* b, const DAAL_INT* ldb, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_avx512_cfeast_hcsrev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* sa, const DAAL_INT* isa, const DAAL_INT* jsa, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_avx512_cfeast_hcsrgv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* sa, const DAAL_INT* isa, const DAAL_INT* jsa,
    const MKL_Complex8* sb, const DAAL_INT* isb, const DAAL_INT* jsb, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_avx512_cfeast_heev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* a, const DAAL_INT* lda, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    MKL_Complex8* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx512_cfeast_hegv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* a, const DAAL_INT* lda, const MKL_Complex8* b,
    const DAAL_INT* ldb, DAAL_INT* fpm, float* epsout, DAAL_INT* loop,
    const float* emin, const float* emax, DAAL_INT* m0, float* e, MKL_Complex8* x,
    DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx512_cfeast_hrci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex8* ze, MKL_Complex8* work, MKL_Complex8* workc, MKL_Complex8* zaq,
    MKL_Complex8* zsq, DAAL_INT* fpm, float* epsout, DAAL_INT* loop,
    const float* emin, const float* emax, DAAL_INT* m0, float* lambda,
    MKL_Complex8* q, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx512_dfeast_sbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const double* a, const DAAL_INT* lda, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx512_dfeast_sbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const double* a, const DAAL_INT* lda, const DAAL_INT* klb,
    const double* b, const DAAL_INT* ldb, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx512_dfeast_scsrev(const char* uplo, const DAAL_INT* n,
    const double* sa, const DAAL_INT* isa, const DAAL_INT* jsa, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx512_dfeast_scsrgv(const char* uplo, const DAAL_INT* n,
    const double* sa, const DAAL_INT* isa, const DAAL_INT* jsa, const double* sb,
    const DAAL_INT* isb, const DAAL_INT* jsb, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx512_dfeast_srci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex16* ze, double* work, MKL_Complex16* workc, double* aq,
    double* sq, DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* lambda, double* q, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_avx512_dfeast_syev(const char* uplo, const DAAL_INT* n,
    const double* a, const DAAL_INT* lda, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx512_dfeast_sygv(const char* uplo, const DAAL_INT* n,
    const double* a, const DAAL_INT* lda, const double* b, const DAAL_INT* ldb,
    DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* e, double* x, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_avx512_feastinit(DAAL_INT* fpm);
void fpk_feast_avx512_sfeast_sbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const float* a, const DAAL_INT* lda, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx512_sfeast_sbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const float* a, const DAAL_INT* lda, const DAAL_INT* klb,
    const float* b, const DAAL_INT* ldb, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx512_sfeast_scsrev(const char* uplo, const DAAL_INT* n,
    const float* sa, const DAAL_INT* isa, const DAAL_INT* jsa, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx512_sfeast_scsrgv(const char* uplo, const DAAL_INT* n,
    const float* sa, const DAAL_INT* isa, const DAAL_INT* jsa, const float* sb,
    const DAAL_INT* isb, const DAAL_INT* jsb, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx512_sfeast_srci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex8* ze, float* work, MKL_Complex8* workc, float* aq, float* sq,
    DAAL_INT* fpm, float* epsout, DAAL_INT* loop, const float* emin,
    const float* emax, DAAL_INT* m0, float* lambda, float* q, DAAL_INT* m,
    float* res, DAAL_INT* info);
void fpk_feast_avx512_sfeast_syev(const char* uplo, const DAAL_INT* n,
    const float* a, const DAAL_INT* lda, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx512_sfeast_sygv(const char* uplo, const DAAL_INT* n,
    const float* a, const DAAL_INT* lda, const float* b, const DAAL_INT* ldb,
    DAAL_INT* fpm, float* epsout, DAAL_INT* loop, const float* emin,
    const float* emax, DAAL_INT* m0, float* e, float* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_avx512_zfeast_hbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex16* a, const DAAL_INT* lda, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m, double* res,
    DAAL_INT* info);
void fpk_feast_avx512_zfeast_hbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex16* a, const DAAL_INT* lda,
    const DAAL_INT* klb, const MKL_Complex16* b, const DAAL_INT* ldb, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m, double* res,
    DAAL_INT* info);
void fpk_feast_avx512_zfeast_hcsrev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* sa, const DAAL_INT* isa, const DAAL_INT* jsa,
    DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_avx512_zfeast_hcsrgv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* sa, const DAAL_INT* isa, const DAAL_INT* jsa,
    const MKL_Complex16* sb, const DAAL_INT* isb, const DAAL_INT* jsb,
    DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_avx512_zfeast_heev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* a, const DAAL_INT* lda, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, MKL_Complex16* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx512_zfeast_hegv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* a, const DAAL_INT* lda, const MKL_Complex16* b,
    const DAAL_INT* ldb, DAAL_INT* fpm, double* epsout, DAAL_INT* loop,
    const double* emin, const double* emax, DAAL_INT* m0, double* e,
    MKL_Complex16* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx512_zfeast_hrci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex16* ze, MKL_Complex16* work, MKL_Complex16* workc,
    MKL_Complex16* zaq, MKL_Complex16* zsq, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* lambda, MKL_Complex16* q, DAAL_INT* m, double* res, DAAL_INT* info);

void fpk_feast_avx512_mic_cfeast_hbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex8* a, const DAAL_INT* lda, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_avx512_mic_cfeast_hbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex8* a, const DAAL_INT* lda,
    const DAAL_INT* klb, const MKL_Complex8* b, const DAAL_INT* ldb, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_avx512_mic_cfeast_hcsrev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* sa, const DAAL_INT* isa, const DAAL_INT* jsa, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_avx512_mic_cfeast_hcsrgv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* sa, const DAAL_INT* isa, const DAAL_INT* jsa,
    const MKL_Complex8* sb, const DAAL_INT* isb, const DAAL_INT* jsb, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, MKL_Complex8* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_avx512_mic_cfeast_heev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* a, const DAAL_INT* lda, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    MKL_Complex8* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx512_mic_cfeast_hegv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex8* a, const DAAL_INT* lda, const MKL_Complex8* b,
    const DAAL_INT* ldb, DAAL_INT* fpm, float* epsout, DAAL_INT* loop,
    const float* emin, const float* emax, DAAL_INT* m0, float* e, MKL_Complex8* x,
    DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx512_mic_cfeast_hrci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex8* ze, MKL_Complex8* work, MKL_Complex8* workc, MKL_Complex8* zaq,
    MKL_Complex8* zsq, DAAL_INT* fpm, float* epsout, DAAL_INT* loop,
    const float* emin, const float* emax, DAAL_INT* m0, float* lambda,
    MKL_Complex8* q, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx512_mic_dfeast_sbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const double* a, const DAAL_INT* lda, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx512_mic_dfeast_sbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const double* a, const DAAL_INT* lda, const DAAL_INT* klb,
    const double* b, const DAAL_INT* ldb, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx512_mic_dfeast_scsrev(const char* uplo, const DAAL_INT* n,
    const double* sa, const DAAL_INT* isa, const DAAL_INT* jsa, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx512_mic_dfeast_scsrgv(const char* uplo, const DAAL_INT* n,
    const double* sa, const DAAL_INT* isa, const DAAL_INT* jsa, const double* sb,
    const DAAL_INT* isb, const DAAL_INT* jsb, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx512_mic_dfeast_srci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex16* ze, double* work, MKL_Complex16* workc, double* aq,
    double* sq, DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* lambda, double* q, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_avx512_mic_dfeast_syev(const char* uplo, const DAAL_INT* n,
    const double* a, const DAAL_INT* lda, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, double* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx512_mic_dfeast_sygv(const char* uplo, const DAAL_INT* n,
    const double* a, const DAAL_INT* lda, const double* b, const DAAL_INT* ldb,
    DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* e, double* x, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_avx512_mic_feastinit(DAAL_INT* fpm);
void fpk_feast_avx512_mic_sfeast_sbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const float* a, const DAAL_INT* lda, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx512_mic_sfeast_sbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const float* a, const DAAL_INT* lda, const DAAL_INT* klb,
    const float* b, const DAAL_INT* ldb, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx512_mic_sfeast_scsrev(const char* uplo, const DAAL_INT* n,
    const float* sa, const DAAL_INT* isa, const DAAL_INT* jsa, DAAL_INT* fpm,
    float* epsout, DAAL_INT* loop, const float* emin, const float* emax,
    DAAL_INT* m0, float* e, float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx512_mic_sfeast_scsrgv(const char* uplo, const DAAL_INT* n,
    const float* sa, const DAAL_INT* isa, const DAAL_INT* jsa, const float* sb,
    const DAAL_INT* isb, const DAAL_INT* jsb, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx512_mic_sfeast_srci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex8* ze, float* work, MKL_Complex8* workc, float* aq, float* sq,
    DAAL_INT* fpm, float* epsout, DAAL_INT* loop, const float* emin,
    const float* emax, DAAL_INT* m0, float* lambda, float* q, DAAL_INT* m,
    float* res, DAAL_INT* info);
void fpk_feast_avx512_mic_sfeast_syev(const char* uplo, const DAAL_INT* n,
    const float* a, const DAAL_INT* lda, DAAL_INT* fpm, float* epsout,
    DAAL_INT* loop, const float* emin, const float* emax, DAAL_INT* m0, float* e,
    float* x, DAAL_INT* m, float* res, DAAL_INT* info);
void fpk_feast_avx512_mic_sfeast_sygv(const char* uplo, const DAAL_INT* n,
    const float* a, const DAAL_INT* lda, const float* b, const DAAL_INT* ldb,
    DAAL_INT* fpm, float* epsout, DAAL_INT* loop, const float* emin,
    const float* emax, DAAL_INT* m0, float* e, float* x, DAAL_INT* m, float* res,
    DAAL_INT* info);
void fpk_feast_avx512_mic_zfeast_hbev(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex16* a, const DAAL_INT* lda, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m, double* res,
    DAAL_INT* info);
void fpk_feast_avx512_mic_zfeast_hbgv(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* kla, const MKL_Complex16* a, const DAAL_INT* lda,
    const DAAL_INT* klb, const MKL_Complex16* b, const DAAL_INT* ldb, DAAL_INT* fpm,
    double* epsout, DAAL_INT* loop, const double* emin, const double* emax,
    DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m, double* res,
    DAAL_INT* info);
void fpk_feast_avx512_mic_zfeast_hcsrev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* sa, const DAAL_INT* isa, const DAAL_INT* jsa,
    DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_avx512_mic_zfeast_hcsrgv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* sa, const DAAL_INT* isa, const DAAL_INT* jsa,
    const MKL_Complex16* sb, const DAAL_INT* isb, const DAAL_INT* jsb,
    DAAL_INT* fpm, double* epsout, DAAL_INT* loop, const double* emin,
    const double* emax, DAAL_INT* m0, double* e, MKL_Complex16* x, DAAL_INT* m,
    double* res, DAAL_INT* info);
void fpk_feast_avx512_mic_zfeast_heev(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* a, const DAAL_INT* lda, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* e, MKL_Complex16* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx512_mic_zfeast_hegv(const char* uplo, const DAAL_INT* n,
    const MKL_Complex16* a, const DAAL_INT* lda, const MKL_Complex16* b,
    const DAAL_INT* ldb, DAAL_INT* fpm, double* epsout, DAAL_INT* loop,
    const double* emin, const double* emax, DAAL_INT* m0, double* e,
    MKL_Complex16* x, DAAL_INT* m, double* res, DAAL_INT* info);
void fpk_feast_avx512_mic_zfeast_hrci(DAAL_INT* ijob, const DAAL_INT* n,
    MKL_Complex16* ze, MKL_Complex16* work, MKL_Complex16* workc,
    MKL_Complex16* zaq, MKL_Complex16* zsq, DAAL_INT* fpm, double* epsout,
    DAAL_INT* loop, const double* emin, const double* emax, DAAL_INT* m0,
    double* lambda, MKL_Complex16* q, DAAL_INT* m, double* res, DAAL_INT* info);


void fpk_lapack_sse2_dgels(const char* trans, const DAAL_INT* m, const DAAL_INT* n,
    const DAAL_INT* nrhs, double* a, const DAAL_INT* lda, double* b,
    const DAAL_INT* ldb, double* work, const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_sse2_dgeqp3(const DAAL_INT* m, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* jpvt, double* tau, double* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_sse2_dgeqrf(const DAAL_INT* m, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, double* tau, double* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_sse2_dgerqf(const DAAL_INT* m, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, double* tau, double* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_sse2_dgesvd(const char* jobu, const char* jobvt,
    const DAAL_INT* m, const DAAL_INT* n, double* a, const DAAL_INT* lda, double* s,
    double* u, const DAAL_INT* ldu, double* vt, const DAAL_INT* ldvt, double* work,
    const DAAL_INT* lwork, DAAL_INT* info );
double fpk_lapack_sse2_dlange(const char* norm, const DAAL_INT* m,
    const DAAL_INT* n, const double* a, const DAAL_INT* lda, double* work );
void fpk_lapack_sse2_dlarnv(const DAAL_INT* idist, DAAL_INT* iseed,
    const DAAL_INT* n, double* x );
void fpk_lapack_sse2_dorgqr(const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k,
    double* a, const DAAL_INT* lda, const double* tau, double* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_sse2_dorgrq(const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k,
    double* a, const DAAL_INT* lda, const double* tau, double* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_sse2_dormqr(const char* side, const char* trans,
    const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k, const double* a,
    const DAAL_INT* lda, const double* tau, double* c, const DAAL_INT* ldc,
    double* work, const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_sse2_dormrq(const char* side, const char* trans,
    const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k, const double* a,
    const DAAL_INT* lda, const double* tau, double* c, const DAAL_INT* ldc,
    double* work, const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_sse2_dpftrf(const char* transr, const char* uplo,
    const DAAL_INT* n, double* a, DAAL_INT* info );
void fpk_lapack_sse2_dpotrf(const char* uplo, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_sse2_dpotri(const char* uplo, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_sse2_dpotrs(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* nrhs, const double* a, const DAAL_INT* lda, double* b,
    const DAAL_INT* ldb, DAAL_INT* info );
void fpk_lapack_sse2_dpptrf(const char* uplo, const DAAL_INT* n, double* ap,
    DAAL_INT* info );
void fpk_lapack_sse2_dpstrf(const char* uplo, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* piv, DAAL_INT* rank, const double* tol,
    double* work, DAAL_INT* info );
void fpk_lapack_sse2_dspevd(const char* jobz, const char* uplo, const DAAL_INT* n,
    double* ap, double* w, double* z, const DAAL_INT* ldz, double* work,
    const DAAL_INT* lwork, DAAL_INT* iwork, const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_sse2_dsyev(const char* jobz, const char* uplo, const DAAL_INT* n,
    double* a, const DAAL_INT* lda, double* w, double* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_sse2_dsyevd(const char* jobz, const char* uplo, const DAAL_INT* n,
    double* a, const DAAL_INT* lda, double* w, double* work, const DAAL_INT* lwork,
    DAAL_INT* iwork, const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_sse2_dsyevr(const char* jobz, const char* range,
    const char* uplo, const DAAL_INT* n, double* a, const DAAL_INT* lda,
    const double* vl, const double* vu, const DAAL_INT* il, const DAAL_INT* iu,
    const double* abstol, DAAL_INT* m, double* w, double* z, const DAAL_INT* ldz,
    DAAL_INT* isuppz, double* work, const DAAL_INT* lwork, DAAL_INT* iwork,
    const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_sse2_dtrtrs(const char* uplo, const char* trans,
    const char* diag, const DAAL_INT* n, const DAAL_INT* nrhs, const double* a,
    const DAAL_INT* lda, double* b, const DAAL_INT* ldb, DAAL_INT* info );
void fpk_lapack_sse2_sgels(const char* trans, const DAAL_INT* m, const DAAL_INT* n,
    const DAAL_INT* nrhs, float* a, const DAAL_INT* lda, float* b,
    const DAAL_INT* ldb, float* work, const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_sse2_sgeqp3(const DAAL_INT* m, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* jpvt, float* tau, float* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_sse2_sgeqrf(const DAAL_INT* m, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, float* tau, float* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_sse2_sgerqf(const DAAL_INT* m, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, float* tau, float* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_sse2_sgesvd(const char* jobu, const char* jobvt,
    const DAAL_INT* m, const DAAL_INT* n, float* a, const DAAL_INT* lda, float* s,
    float* u, const DAAL_INT* ldu, float* vt, const DAAL_INT* ldvt, float* work,
    const DAAL_INT* lwork, DAAL_INT* info );
float fpk_lapack_sse2_slange(const char* norm, const DAAL_INT* m,
    const DAAL_INT* n, const float* a, const DAAL_INT* lda, float* work );
void fpk_lapack_sse2_slarnv(const DAAL_INT* idist, DAAL_INT* iseed,
    const DAAL_INT* n, float* x );
void fpk_lapack_sse2_sorgqr(const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k,
    float* a, const DAAL_INT* lda, const float* tau, float* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_sse2_sorgrq(const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k,
    float* a, const DAAL_INT* lda, const float* tau, float* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_sse2_sormqr(const char* side, const char* trans,
    const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k, const float* a,
    const DAAL_INT* lda, const float* tau, float* c, const DAAL_INT* ldc,
    float* work, const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_sse2_sormrq(const char* side, const char* trans,
    const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k, const float* a,
    const DAAL_INT* lda, const float* tau, float* c, const DAAL_INT* ldc,
    float* work, const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_sse2_spftrf(const char* transr, const char* uplo,
    const DAAL_INT* n, float* a, DAAL_INT* info );
void fpk_lapack_sse2_spotrf(const char* uplo, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_sse2_spotri(const char* uplo, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_sse2_spotrs(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* nrhs, const float* a, const DAAL_INT* lda, float* b,
    const DAAL_INT* ldb, DAAL_INT* info );
void fpk_lapack_sse2_spptrf(const char* uplo, const DAAL_INT* n, float* ap,
    DAAL_INT* info );
void fpk_lapack_sse2_spstrf(const char* uplo, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* piv, DAAL_INT* rank, const float* tol,
    float* work, DAAL_INT* info );
void fpk_lapack_sse2_sspevd(const char* jobz, const char* uplo, const DAAL_INT* n,
    float* ap, float* w, float* z, const DAAL_INT* ldz, float* work,
    const DAAL_INT* lwork, DAAL_INT* iwork, const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_sse2_ssyev(const char* jobz, const char* uplo, const DAAL_INT* n,
    float* a, const DAAL_INT* lda, float* w, float* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_sse2_ssyevd(const char* jobz, const char* uplo, const DAAL_INT* n,
    float* a, const DAAL_INT* lda, float* w, float* work, const DAAL_INT* lwork,
    DAAL_INT* iwork, const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_sse2_ssyevr(const char* jobz, const char* range,
    const char* uplo, const DAAL_INT* n, float* a, const DAAL_INT* lda,
    const float* vl, const float* vu, const DAAL_INT* il, const DAAL_INT* iu,
    const float* abstol, DAAL_INT* m, float* w, float* z, const DAAL_INT* ldz,
    DAAL_INT* isuppz, float* work, const DAAL_INT* lwork, DAAL_INT* iwork,
    const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_sse2_strtrs(const char* uplo, const char* trans,
    const char* diag, const DAAL_INT* n, const DAAL_INT* nrhs, const float* a,
    const DAAL_INT* lda, float* b, const DAAL_INT* ldb, DAAL_INT* info );

void fpk_lapack_ssse3_dgels(const char* trans, const DAAL_INT* m,
    const DAAL_INT* n, const DAAL_INT* nrhs, double* a, const DAAL_INT* lda,
    double* b, const DAAL_INT* ldb, double* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_ssse3_dgeqp3(const DAAL_INT* m, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* jpvt, double* tau, double* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_ssse3_dgeqrf(const DAAL_INT* m, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, double* tau, double* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_ssse3_dgerqf(const DAAL_INT* m, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, double* tau, double* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_ssse3_dgesvd(const char* jobu, const char* jobvt,
    const DAAL_INT* m, const DAAL_INT* n, double* a, const DAAL_INT* lda, double* s,
    double* u, const DAAL_INT* ldu, double* vt, const DAAL_INT* ldvt, double* work,
    const DAAL_INT* lwork, DAAL_INT* info );
double fpk_lapack_ssse3_dlange(const char* norm, const DAAL_INT* m,
    const DAAL_INT* n, const double* a, const DAAL_INT* lda, double* work );
void fpk_lapack_ssse3_dlarnv(const DAAL_INT* idist, DAAL_INT* iseed,
    const DAAL_INT* n, double* x );
void fpk_lapack_ssse3_dorgqr(const DAAL_INT* m, const DAAL_INT* n,
    const DAAL_INT* k, double* a, const DAAL_INT* lda, const double* tau,
    double* work, const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_ssse3_dorgrq(const DAAL_INT* m, const DAAL_INT* n,
    const DAAL_INT* k, double* a, const DAAL_INT* lda, const double* tau,
    double* work, const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_ssse3_dormqr(const char* side, const char* trans,
    const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k, const double* a,
    const DAAL_INT* lda, const double* tau, double* c, const DAAL_INT* ldc,
    double* work, const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_ssse3_dormrq(const char* side, const char* trans,
    const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k, const double* a,
    const DAAL_INT* lda, const double* tau, double* c, const DAAL_INT* ldc,
    double* work, const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_ssse3_dpftrf(const char* transr, const char* uplo,
    const DAAL_INT* n, double* a, DAAL_INT* info );
void fpk_lapack_ssse3_dpotrf(const char* uplo, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_ssse3_dpotri(const char* uplo, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_ssse3_dpotrs(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* nrhs, const double* a, const DAAL_INT* lda, double* b,
    const DAAL_INT* ldb, DAAL_INT* info );
void fpk_lapack_ssse3_dpptrf(const char* uplo, const DAAL_INT* n, double* ap,
    DAAL_INT* info );
void fpk_lapack_ssse3_dpstrf(const char* uplo, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* piv, DAAL_INT* rank, const double* tol,
    double* work, DAAL_INT* info );
void fpk_lapack_ssse3_dspevd(const char* jobz, const char* uplo,
    const DAAL_INT* n, double* ap, double* w, double* z, const DAAL_INT* ldz,
    double* work, const DAAL_INT* lwork, DAAL_INT* iwork, const DAAL_INT* liwork,
    DAAL_INT* info );
void fpk_lapack_ssse3_dsyev(const char* jobz, const char* uplo, const DAAL_INT* n,
    double* a, const DAAL_INT* lda, double* w, double* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_ssse3_dsyevd(const char* jobz, const char* uplo,
    const DAAL_INT* n, double* a, const DAAL_INT* lda, double* w, double* work,
    const DAAL_INT* lwork, DAAL_INT* iwork, const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_ssse3_dsyevr(const char* jobz, const char* range,
    const char* uplo, const DAAL_INT* n, double* a, const DAAL_INT* lda,
    const double* vl, const double* vu, const DAAL_INT* il, const DAAL_INT* iu,
    const double* abstol, DAAL_INT* m, double* w, double* z, const DAAL_INT* ldz,
    DAAL_INT* isuppz, double* work, const DAAL_INT* lwork, DAAL_INT* iwork,
    const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_ssse3_dtrtrs(const char* uplo, const char* trans,
    const char* diag, const DAAL_INT* n, const DAAL_INT* nrhs, const double* a,
    const DAAL_INT* lda, double* b, const DAAL_INT* ldb, DAAL_INT* info );
void fpk_lapack_ssse3_sgels(const char* trans, const DAAL_INT* m,
    const DAAL_INT* n, const DAAL_INT* nrhs, float* a, const DAAL_INT* lda,
    float* b, const DAAL_INT* ldb, float* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_ssse3_sgeqp3(const DAAL_INT* m, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* jpvt, float* tau, float* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_ssse3_sgeqrf(const DAAL_INT* m, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, float* tau, float* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_ssse3_sgerqf(const DAAL_INT* m, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, float* tau, float* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_ssse3_sgesvd(const char* jobu, const char* jobvt,
    const DAAL_INT* m, const DAAL_INT* n, float* a, const DAAL_INT* lda, float* s,
    float* u, const DAAL_INT* ldu, float* vt, const DAAL_INT* ldvt, float* work,
    const DAAL_INT* lwork, DAAL_INT* info );
float fpk_lapack_ssse3_slange(const char* norm, const DAAL_INT* m,
    const DAAL_INT* n, const float* a, const DAAL_INT* lda, float* work );
void fpk_lapack_ssse3_slarnv(const DAAL_INT* idist, DAAL_INT* iseed,
    const DAAL_INT* n, float* x );
void fpk_lapack_ssse3_sorgqr(const DAAL_INT* m, const DAAL_INT* n,
    const DAAL_INT* k, float* a, const DAAL_INT* lda, const float* tau,
    float* work, const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_ssse3_sorgrq(const DAAL_INT* m, const DAAL_INT* n,
    const DAAL_INT* k, float* a, const DAAL_INT* lda, const float* tau,
    float* work, const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_ssse3_sormqr(const char* side, const char* trans,
    const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k, const float* a,
    const DAAL_INT* lda, const float* tau, float* c, const DAAL_INT* ldc,
    float* work, const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_ssse3_sormrq(const char* side, const char* trans,
    const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k, const float* a,
    const DAAL_INT* lda, const float* tau, float* c, const DAAL_INT* ldc,
    float* work, const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_ssse3_spftrf(const char* transr, const char* uplo,
    const DAAL_INT* n, float* a, DAAL_INT* info );
void fpk_lapack_ssse3_spotrf(const char* uplo, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_ssse3_spotri(const char* uplo, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_ssse3_spotrs(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* nrhs, const float* a, const DAAL_INT* lda, float* b,
    const DAAL_INT* ldb, DAAL_INT* info );
void fpk_lapack_ssse3_spptrf(const char* uplo, const DAAL_INT* n, float* ap,
    DAAL_INT* info );
void fpk_lapack_ssse3_spstrf(const char* uplo, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* piv, DAAL_INT* rank, const float* tol,
    float* work, DAAL_INT* info );
void fpk_lapack_ssse3_sspevd(const char* jobz, const char* uplo,
    const DAAL_INT* n, float* ap, float* w, float* z, const DAAL_INT* ldz,
    float* work, const DAAL_INT* lwork, DAAL_INT* iwork, const DAAL_INT* liwork,
    DAAL_INT* info );
void fpk_lapack_ssse3_ssyev(const char* jobz, const char* uplo, const DAAL_INT* n,
    float* a, const DAAL_INT* lda, float* w, float* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_ssse3_ssyevd(const char* jobz, const char* uplo,
    const DAAL_INT* n, float* a, const DAAL_INT* lda, float* w, float* work,
    const DAAL_INT* lwork, DAAL_INT* iwork, const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_ssse3_ssyevr(const char* jobz, const char* range,
    const char* uplo, const DAAL_INT* n, float* a, const DAAL_INT* lda,
    const float* vl, const float* vu, const DAAL_INT* il, const DAAL_INT* iu,
    const float* abstol, DAAL_INT* m, float* w, float* z, const DAAL_INT* ldz,
    DAAL_INT* isuppz, float* work, const DAAL_INT* lwork, DAAL_INT* iwork,
    const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_ssse3_strtrs(const char* uplo, const char* trans,
    const char* diag, const DAAL_INT* n, const DAAL_INT* nrhs, const float* a,
    const DAAL_INT* lda, float* b, const DAAL_INT* ldb, DAAL_INT* info );

void fpk_lapack_sse42_dgels(const char* trans, const DAAL_INT* m,
    const DAAL_INT* n, const DAAL_INT* nrhs, double* a, const DAAL_INT* lda,
    double* b, const DAAL_INT* ldb, double* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_sse42_dgeqp3(const DAAL_INT* m, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* jpvt, double* tau, double* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_sse42_dgeqrf(const DAAL_INT* m, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, double* tau, double* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_sse42_dgerqf(const DAAL_INT* m, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, double* tau, double* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_sse42_dgesvd(const char* jobu, const char* jobvt,
    const DAAL_INT* m, const DAAL_INT* n, double* a, const DAAL_INT* lda, double* s,
    double* u, const DAAL_INT* ldu, double* vt, const DAAL_INT* ldvt, double* work,
    const DAAL_INT* lwork, DAAL_INT* info );
double fpk_lapack_sse42_dlange(const char* norm, const DAAL_INT* m,
    const DAAL_INT* n, const double* a, const DAAL_INT* lda, double* work );
void fpk_lapack_sse42_dlarnv(const DAAL_INT* idist, DAAL_INT* iseed,
    const DAAL_INT* n, double* x );
void fpk_lapack_sse42_dorgqr(const DAAL_INT* m, const DAAL_INT* n,
    const DAAL_INT* k, double* a, const DAAL_INT* lda, const double* tau,
    double* work, const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_sse42_dorgrq(const DAAL_INT* m, const DAAL_INT* n,
    const DAAL_INT* k, double* a, const DAAL_INT* lda, const double* tau,
    double* work, const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_sse42_dormqr(const char* side, const char* trans,
    const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k, const double* a,
    const DAAL_INT* lda, const double* tau, double* c, const DAAL_INT* ldc,
    double* work, const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_sse42_dormrq(const char* side, const char* trans,
    const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k, const double* a,
    const DAAL_INT* lda, const double* tau, double* c, const DAAL_INT* ldc,
    double* work, const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_sse42_dpftrf(const char* transr, const char* uplo,
    const DAAL_INT* n, double* a, DAAL_INT* info );
void fpk_lapack_sse42_dpotrf(const char* uplo, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_sse42_dpotri(const char* uplo, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_sse42_dpotrs(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* nrhs, const double* a, const DAAL_INT* lda, double* b,
    const DAAL_INT* ldb, DAAL_INT* info );
void fpk_lapack_sse42_dpptrf(const char* uplo, const DAAL_INT* n, double* ap,
    DAAL_INT* info );
void fpk_lapack_sse42_dpstrf(const char* uplo, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* piv, DAAL_INT* rank, const double* tol,
    double* work, DAAL_INT* info );
void fpk_lapack_sse42_dspevd(const char* jobz, const char* uplo,
    const DAAL_INT* n, double* ap, double* w, double* z, const DAAL_INT* ldz,
    double* work, const DAAL_INT* lwork, DAAL_INT* iwork, const DAAL_INT* liwork,
    DAAL_INT* info );
void fpk_lapack_sse42_dsyev(const char* jobz, const char* uplo, const DAAL_INT* n,
    double* a, const DAAL_INT* lda, double* w, double* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_sse42_dsyevd(const char* jobz, const char* uplo,
    const DAAL_INT* n, double* a, const DAAL_INT* lda, double* w, double* work,
    const DAAL_INT* lwork, DAAL_INT* iwork, const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_sse42_dsyevr(const char* jobz, const char* range,
    const char* uplo, const DAAL_INT* n, double* a, const DAAL_INT* lda,
    const double* vl, const double* vu, const DAAL_INT* il, const DAAL_INT* iu,
    const double* abstol, DAAL_INT* m, double* w, double* z, const DAAL_INT* ldz,
    DAAL_INT* isuppz, double* work, const DAAL_INT* lwork, DAAL_INT* iwork,
    const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_sse42_dtrtrs(const char* uplo, const char* trans,
    const char* diag, const DAAL_INT* n, const DAAL_INT* nrhs, const double* a,
    const DAAL_INT* lda, double* b, const DAAL_INT* ldb, DAAL_INT* info );
void fpk_lapack_sse42_sgels(const char* trans, const DAAL_INT* m,
    const DAAL_INT* n, const DAAL_INT* nrhs, float* a, const DAAL_INT* lda,
    float* b, const DAAL_INT* ldb, float* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_sse42_sgeqp3(const DAAL_INT* m, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* jpvt, float* tau, float* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_sse42_sgeqrf(const DAAL_INT* m, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, float* tau, float* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_sse42_sgerqf(const DAAL_INT* m, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, float* tau, float* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_sse42_sgesvd(const char* jobu, const char* jobvt,
    const DAAL_INT* m, const DAAL_INT* n, float* a, const DAAL_INT* lda, float* s,
    float* u, const DAAL_INT* ldu, float* vt, const DAAL_INT* ldvt, float* work,
    const DAAL_INT* lwork, DAAL_INT* info );
float fpk_lapack_sse42_slange(const char* norm, const DAAL_INT* m,
    const DAAL_INT* n, const float* a, const DAAL_INT* lda, float* work );
void fpk_lapack_sse42_slarnv(const DAAL_INT* idist, DAAL_INT* iseed,
    const DAAL_INT* n, float* x );
void fpk_lapack_sse42_sorgqr(const DAAL_INT* m, const DAAL_INT* n,
    const DAAL_INT* k, float* a, const DAAL_INT* lda, const float* tau,
    float* work, const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_sse42_sorgrq(const DAAL_INT* m, const DAAL_INT* n,
    const DAAL_INT* k, float* a, const DAAL_INT* lda, const float* tau,
    float* work, const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_sse42_sormqr(const char* side, const char* trans,
    const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k, const float* a,
    const DAAL_INT* lda, const float* tau, float* c, const DAAL_INT* ldc,
    float* work, const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_sse42_sormrq(const char* side, const char* trans,
    const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k, const float* a,
    const DAAL_INT* lda, const float* tau, float* c, const DAAL_INT* ldc,
    float* work, const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_sse42_spftrf(const char* transr, const char* uplo,
    const DAAL_INT* n, float* a, DAAL_INT* info );
void fpk_lapack_sse42_spotrf(const char* uplo, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_sse42_spotri(const char* uplo, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_sse42_spotrs(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* nrhs, const float* a, const DAAL_INT* lda, float* b,
    const DAAL_INT* ldb, DAAL_INT* info );
void fpk_lapack_sse42_spptrf(const char* uplo, const DAAL_INT* n, float* ap,
    DAAL_INT* info );
void fpk_lapack_sse42_spstrf(const char* uplo, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* piv, DAAL_INT* rank, const float* tol,
    float* work, DAAL_INT* info );
void fpk_lapack_sse42_sspevd(const char* jobz, const char* uplo,
    const DAAL_INT* n, float* ap, float* w, float* z, const DAAL_INT* ldz,
    float* work, const DAAL_INT* lwork, DAAL_INT* iwork, const DAAL_INT* liwork,
    DAAL_INT* info );
void fpk_lapack_sse42_ssyev(const char* jobz, const char* uplo, const DAAL_INT* n,
    float* a, const DAAL_INT* lda, float* w, float* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_sse42_ssyevd(const char* jobz, const char* uplo,
    const DAAL_INT* n, float* a, const DAAL_INT* lda, float* w, float* work,
    const DAAL_INT* lwork, DAAL_INT* iwork, const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_sse42_ssyevr(const char* jobz, const char* range,
    const char* uplo, const DAAL_INT* n, float* a, const DAAL_INT* lda,
    const float* vl, const float* vu, const DAAL_INT* il, const DAAL_INT* iu,
    const float* abstol, DAAL_INT* m, float* w, float* z, const DAAL_INT* ldz,
    DAAL_INT* isuppz, float* work, const DAAL_INT* lwork, DAAL_INT* iwork,
    const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_sse42_strtrs(const char* uplo, const char* trans,
    const char* diag, const DAAL_INT* n, const DAAL_INT* nrhs, const float* a,
    const DAAL_INT* lda, float* b, const DAAL_INT* ldb, DAAL_INT* info );

void fpk_lapack_avx_dgels(const char* trans, const DAAL_INT* m, const DAAL_INT* n,
    const DAAL_INT* nrhs, double* a, const DAAL_INT* lda, double* b,
    const DAAL_INT* ldb, double* work, const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx_dgeqp3(const DAAL_INT* m, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* jpvt, double* tau, double* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx_dgeqrf(const DAAL_INT* m, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, double* tau, double* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_avx_dgerqf(const DAAL_INT* m, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, double* tau, double* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_avx_dgesvd(const char* jobu, const char* jobvt, const DAAL_INT* m,
    const DAAL_INT* n, double* a, const DAAL_INT* lda, double* s, double* u,
    const DAAL_INT* ldu, double* vt, const DAAL_INT* ldvt, double* work,
    const DAAL_INT* lwork, DAAL_INT* info );
double fpk_lapack_avx_dlange(const char* norm, const DAAL_INT* m,
    const DAAL_INT* n, const double* a, const DAAL_INT* lda, double* work );
void fpk_lapack_avx_dlarnv(const DAAL_INT* idist, DAAL_INT* iseed,
    const DAAL_INT* n, double* x );
void fpk_lapack_avx_dorgqr(const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k,
    double* a, const DAAL_INT* lda, const double* tau, double* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx_dorgrq(const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k,
    double* a, const DAAL_INT* lda, const double* tau, double* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx_dormqr(const char* side, const char* trans, const DAAL_INT* m,
    const DAAL_INT* n, const DAAL_INT* k, const double* a, const DAAL_INT* lda,
    const double* tau, double* c, const DAAL_INT* ldc, double* work,
    const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_avx_dormrq(const char* side, const char* trans, const DAAL_INT* m,
    const DAAL_INT* n, const DAAL_INT* k, const double* a, const DAAL_INT* lda,
    const double* tau, double* c, const DAAL_INT* ldc, double* work,
    const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_avx_dpftrf(const char* transr, const char* uplo,
    const DAAL_INT* n, double* a, DAAL_INT* info );
void fpk_lapack_avx_dpotrf(const char* uplo, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_avx_dpotri(const char* uplo, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_avx_dpotrs(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* nrhs, const double* a, const DAAL_INT* lda, double* b,
    const DAAL_INT* ldb, DAAL_INT* info );
void fpk_lapack_avx_dpptrf(const char* uplo, const DAAL_INT* n, double* ap,
    DAAL_INT* info );
void fpk_lapack_avx_dpstrf(const char* uplo, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* piv, DAAL_INT* rank, const double* tol,
    double* work, DAAL_INT* info );
void fpk_lapack_avx_dspevd(const char* jobz, const char* uplo, const DAAL_INT* n,
    double* ap, double* w, double* z, const DAAL_INT* ldz, double* work,
    const DAAL_INT* lwork, DAAL_INT* iwork, const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_avx_dsyev(const char* jobz, const char* uplo, const DAAL_INT* n,
    double* a, const DAAL_INT* lda, double* w, double* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_avx_dsyevd(const char* jobz, const char* uplo, const DAAL_INT* n,
    double* a, const DAAL_INT* lda, double* w, double* work, const DAAL_INT* lwork,
    DAAL_INT* iwork, const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_avx_dsyevr(const char* jobz, const char* range, const char* uplo,
    const DAAL_INT* n, double* a, const DAAL_INT* lda, const double* vl,
    const double* vu, const DAAL_INT* il, const DAAL_INT* iu, const double* abstol,
    DAAL_INT* m, double* w, double* z, const DAAL_INT* ldz, DAAL_INT* isuppz,
    double* work, const DAAL_INT* lwork, DAAL_INT* iwork, const DAAL_INT* liwork,
    DAAL_INT* info );
void fpk_lapack_avx_dtrtrs(const char* uplo, const char* trans, const char* diag,
    const DAAL_INT* n, const DAAL_INT* nrhs, const double* a, const DAAL_INT* lda,
    double* b, const DAAL_INT* ldb, DAAL_INT* info );
void fpk_lapack_avx_sgels(const char* trans, const DAAL_INT* m, const DAAL_INT* n,
    const DAAL_INT* nrhs, float* a, const DAAL_INT* lda, float* b,
    const DAAL_INT* ldb, float* work, const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx_sgeqp3(const DAAL_INT* m, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* jpvt, float* tau, float* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx_sgeqrf(const DAAL_INT* m, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, float* tau, float* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_avx_sgerqf(const DAAL_INT* m, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, float* tau, float* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_avx_sgesvd(const char* jobu, const char* jobvt, const DAAL_INT* m,
    const DAAL_INT* n, float* a, const DAAL_INT* lda, float* s, float* u,
    const DAAL_INT* ldu, float* vt, const DAAL_INT* ldvt, float* work,
    const DAAL_INT* lwork, DAAL_INT* info );
float fpk_lapack_avx_slange(const char* norm, const DAAL_INT* m, const DAAL_INT* n,
    const float* a, const DAAL_INT* lda, float* work );
void fpk_lapack_avx_slarnv(const DAAL_INT* idist, DAAL_INT* iseed,
    const DAAL_INT* n, float* x );
void fpk_lapack_avx_sorgqr(const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k,
    float* a, const DAAL_INT* lda, const float* tau, float* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx_sorgrq(const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k,
    float* a, const DAAL_INT* lda, const float* tau, float* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx_sormqr(const char* side, const char* trans, const DAAL_INT* m,
    const DAAL_INT* n, const DAAL_INT* k, const float* a, const DAAL_INT* lda,
    const float* tau, float* c, const DAAL_INT* ldc, float* work,
    const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_avx_sormrq(const char* side, const char* trans, const DAAL_INT* m,
    const DAAL_INT* n, const DAAL_INT* k, const float* a, const DAAL_INT* lda,
    const float* tau, float* c, const DAAL_INT* ldc, float* work,
    const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_avx_spftrf(const char* transr, const char* uplo,
    const DAAL_INT* n, float* a, DAAL_INT* info );
void fpk_lapack_avx_spotrf(const char* uplo, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_avx_spotri(const char* uplo, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_avx_spotrs(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* nrhs, const float* a, const DAAL_INT* lda, float* b,
    const DAAL_INT* ldb, DAAL_INT* info );
void fpk_lapack_avx_spptrf(const char* uplo, const DAAL_INT* n, float* ap,
    DAAL_INT* info );
void fpk_lapack_avx_spstrf(const char* uplo, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* piv, DAAL_INT* rank, const float* tol,
    float* work, DAAL_INT* info );
void fpk_lapack_avx_sspevd(const char* jobz, const char* uplo, const DAAL_INT* n,
    float* ap, float* w, float* z, const DAAL_INT* ldz, float* work,
    const DAAL_INT* lwork, DAAL_INT* iwork, const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_avx_ssyev(const char* jobz, const char* uplo, const DAAL_INT* n,
    float* a, const DAAL_INT* lda, float* w, float* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_avx_ssyevd(const char* jobz, const char* uplo, const DAAL_INT* n,
    float* a, const DAAL_INT* lda, float* w, float* work, const DAAL_INT* lwork,
    DAAL_INT* iwork, const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_avx_ssyevr(const char* jobz, const char* range, const char* uplo,
    const DAAL_INT* n, float* a, const DAAL_INT* lda, const float* vl,
    const float* vu, const DAAL_INT* il, const DAAL_INT* iu, const float* abstol,
    DAAL_INT* m, float* w, float* z, const DAAL_INT* ldz, DAAL_INT* isuppz,
    float* work, const DAAL_INT* lwork, DAAL_INT* iwork, const DAAL_INT* liwork,
    DAAL_INT* info );
void fpk_lapack_avx_strtrs(const char* uplo, const char* trans, const char* diag,
    const DAAL_INT* n, const DAAL_INT* nrhs, const float* a, const DAAL_INT* lda,
    float* b, const DAAL_INT* ldb, DAAL_INT* info );

void fpk_lapack_avx2_dgels(const char* trans, const DAAL_INT* m, const DAAL_INT* n,
    const DAAL_INT* nrhs, double* a, const DAAL_INT* lda, double* b,
    const DAAL_INT* ldb, double* work, const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx2_dgeqp3(const DAAL_INT* m, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* jpvt, double* tau, double* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx2_dgeqrf(const DAAL_INT* m, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, double* tau, double* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_avx2_dgerqf(const DAAL_INT* m, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, double* tau, double* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_avx2_dgesvd(const char* jobu, const char* jobvt,
    const DAAL_INT* m, const DAAL_INT* n, double* a, const DAAL_INT* lda, double* s,
    double* u, const DAAL_INT* ldu, double* vt, const DAAL_INT* ldvt, double* work,
    const DAAL_INT* lwork, DAAL_INT* info );
double fpk_lapack_avx2_dlange(const char* norm, const DAAL_INT* m,
    const DAAL_INT* n, const double* a, const DAAL_INT* lda, double* work );
void fpk_lapack_avx2_dlarnv(const DAAL_INT* idist, DAAL_INT* iseed,
    const DAAL_INT* n, double* x );
void fpk_lapack_avx2_dorgqr(const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k,
    double* a, const DAAL_INT* lda, const double* tau, double* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx2_dorgrq(const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k,
    double* a, const DAAL_INT* lda, const double* tau, double* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx2_dormqr(const char* side, const char* trans,
    const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k, const double* a,
    const DAAL_INT* lda, const double* tau, double* c, const DAAL_INT* ldc,
    double* work, const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_avx2_dormrq(const char* side, const char* trans,
    const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k, const double* a,
    const DAAL_INT* lda, const double* tau, double* c, const DAAL_INT* ldc,
    double* work, const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_avx2_dpftrf(const char* transr, const char* uplo,
    const DAAL_INT* n, double* a, DAAL_INT* info );
void fpk_lapack_avx2_dpotrf(const char* uplo, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_avx2_dpotri(const char* uplo, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_avx2_dpotrs(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* nrhs, const double* a, const DAAL_INT* lda, double* b,
    const DAAL_INT* ldb, DAAL_INT* info );
void fpk_lapack_avx2_dpptrf(const char* uplo, const DAAL_INT* n, double* ap,
    DAAL_INT* info );
void fpk_lapack_avx2_dpstrf(const char* uplo, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* piv, DAAL_INT* rank, const double* tol,
    double* work, DAAL_INT* info );
void fpk_lapack_avx2_dspevd(const char* jobz, const char* uplo, const DAAL_INT* n,
    double* ap, double* w, double* z, const DAAL_INT* ldz, double* work,
    const DAAL_INT* lwork, DAAL_INT* iwork, const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_avx2_dsyev(const char* jobz, const char* uplo, const DAAL_INT* n,
    double* a, const DAAL_INT* lda, double* w, double* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_avx2_dsyevd(const char* jobz, const char* uplo, const DAAL_INT* n,
    double* a, const DAAL_INT* lda, double* w, double* work, const DAAL_INT* lwork,
    DAAL_INT* iwork, const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_avx2_dsyevr(const char* jobz, const char* range,
    const char* uplo, const DAAL_INT* n, double* a, const DAAL_INT* lda,
    const double* vl, const double* vu, const DAAL_INT* il, const DAAL_INT* iu,
    const double* abstol, DAAL_INT* m, double* w, double* z, const DAAL_INT* ldz,
    DAAL_INT* isuppz, double* work, const DAAL_INT* lwork, DAAL_INT* iwork,
    const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_avx2_dtrtrs(const char* uplo, const char* trans,
    const char* diag, const DAAL_INT* n, const DAAL_INT* nrhs, const double* a,
    const DAAL_INT* lda, double* b, const DAAL_INT* ldb, DAAL_INT* info );
void fpk_lapack_avx2_sgels(const char* trans, const DAAL_INT* m, const DAAL_INT* n,
    const DAAL_INT* nrhs, float* a, const DAAL_INT* lda, float* b,
    const DAAL_INT* ldb, float* work, const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx2_sgeqp3(const DAAL_INT* m, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* jpvt, float* tau, float* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx2_sgeqrf(const DAAL_INT* m, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, float* tau, float* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_avx2_sgerqf(const DAAL_INT* m, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, float* tau, float* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_avx2_sgesvd(const char* jobu, const char* jobvt,
    const DAAL_INT* m, const DAAL_INT* n, float* a, const DAAL_INT* lda, float* s,
    float* u, const DAAL_INT* ldu, float* vt, const DAAL_INT* ldvt, float* work,
    const DAAL_INT* lwork, DAAL_INT* info );
float fpk_lapack_avx2_slange(const char* norm, const DAAL_INT* m,
    const DAAL_INT* n, const float* a, const DAAL_INT* lda, float* work );
void fpk_lapack_avx2_slarnv(const DAAL_INT* idist, DAAL_INT* iseed,
    const DAAL_INT* n, float* x );
void fpk_lapack_avx2_sorgqr(const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k,
    float* a, const DAAL_INT* lda, const float* tau, float* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx2_sorgrq(const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k,
    float* a, const DAAL_INT* lda, const float* tau, float* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx2_sormqr(const char* side, const char* trans,
    const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k, const float* a,
    const DAAL_INT* lda, const float* tau, float* c, const DAAL_INT* ldc,
    float* work, const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_avx2_sormrq(const char* side, const char* trans,
    const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k, const float* a,
    const DAAL_INT* lda, const float* tau, float* c, const DAAL_INT* ldc,
    float* work, const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_avx2_spftrf(const char* transr, const char* uplo,
    const DAAL_INT* n, float* a, DAAL_INT* info );
void fpk_lapack_avx2_spotrf(const char* uplo, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_avx2_spotri(const char* uplo, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_avx2_spotrs(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* nrhs, const float* a, const DAAL_INT* lda, float* b,
    const DAAL_INT* ldb, DAAL_INT* info );
void fpk_lapack_avx2_spptrf(const char* uplo, const DAAL_INT* n, float* ap,
    DAAL_INT* info );
void fpk_lapack_avx2_spstrf(const char* uplo, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* piv, DAAL_INT* rank, const float* tol,
    float* work, DAAL_INT* info );
void fpk_lapack_avx2_sspevd(const char* jobz, const char* uplo, const DAAL_INT* n,
    float* ap, float* w, float* z, const DAAL_INT* ldz, float* work,
    const DAAL_INT* lwork, DAAL_INT* iwork, const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_avx2_ssyev(const char* jobz, const char* uplo, const DAAL_INT* n,
    float* a, const DAAL_INT* lda, float* w, float* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_avx2_ssyevd(const char* jobz, const char* uplo, const DAAL_INT* n,
    float* a, const DAAL_INT* lda, float* w, float* work, const DAAL_INT* lwork,
    DAAL_INT* iwork, const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_avx2_ssyevr(const char* jobz, const char* range,
    const char* uplo, const DAAL_INT* n, float* a, const DAAL_INT* lda,
    const float* vl, const float* vu, const DAAL_INT* il, const DAAL_INT* iu,
    const float* abstol, DAAL_INT* m, float* w, float* z, const DAAL_INT* ldz,
    DAAL_INT* isuppz, float* work, const DAAL_INT* lwork, DAAL_INT* iwork,
    const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_avx2_strtrs(const char* uplo, const char* trans,
    const char* diag, const DAAL_INT* n, const DAAL_INT* nrhs, const float* a,
    const DAAL_INT* lda, float* b, const DAAL_INT* ldb, DAAL_INT* info );

void fpk_lapack_avx512_dgels(const char* trans, const DAAL_INT* m,
    const DAAL_INT* n, const DAAL_INT* nrhs, double* a, const DAAL_INT* lda,
    double* b, const DAAL_INT* ldb, double* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_avx512_dgeqp3(const DAAL_INT* m, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* jpvt, double* tau, double* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx512_dgeqrf(const DAAL_INT* m, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, double* tau, double* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_avx512_dgerqf(const DAAL_INT* m, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, double* tau, double* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_avx512_dgesvd(const char* jobu, const char* jobvt,
    const DAAL_INT* m, const DAAL_INT* n, double* a, const DAAL_INT* lda, double* s,
    double* u, const DAAL_INT* ldu, double* vt, const DAAL_INT* ldvt, double* work,
    const DAAL_INT* lwork, DAAL_INT* info );
double fpk_lapack_avx512_dlange(const char* norm, const DAAL_INT* m,
    const DAAL_INT* n, const double* a, const DAAL_INT* lda, double* work );
void fpk_lapack_avx512_dlarnv(const DAAL_INT* idist, DAAL_INT* iseed,
    const DAAL_INT* n, double* x );
void fpk_lapack_avx512_dorgqr(const DAAL_INT* m, const DAAL_INT* n,
    const DAAL_INT* k, double* a, const DAAL_INT* lda, const double* tau,
    double* work, const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx512_dorgrq(const DAAL_INT* m, const DAAL_INT* n,
    const DAAL_INT* k, double* a, const DAAL_INT* lda, const double* tau,
    double* work, const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx512_dormqr(const char* side, const char* trans,
    const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k, const double* a,
    const DAAL_INT* lda, const double* tau, double* c, const DAAL_INT* ldc,
    double* work, const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_avx512_dormrq(const char* side, const char* trans,
    const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k, const double* a,
    const DAAL_INT* lda, const double* tau, double* c, const DAAL_INT* ldc,
    double* work, const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_avx512_dpftrf(const char* transr, const char* uplo,
    const DAAL_INT* n, double* a, DAAL_INT* info );
void fpk_lapack_avx512_dpotrf(const char* uplo, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_avx512_dpotri(const char* uplo, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_avx512_dpotrs(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* nrhs, const double* a, const DAAL_INT* lda, double* b,
    const DAAL_INT* ldb, DAAL_INT* info );
void fpk_lapack_avx512_dpptrf(const char* uplo, const DAAL_INT* n, double* ap,
    DAAL_INT* info );
void fpk_lapack_avx512_dpstrf(const char* uplo, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* piv, DAAL_INT* rank, const double* tol,
    double* work, DAAL_INT* info );
void fpk_lapack_avx512_dspevd(const char* jobz, const char* uplo,
    const DAAL_INT* n, double* ap, double* w, double* z, const DAAL_INT* ldz,
    double* work, const DAAL_INT* lwork, DAAL_INT* iwork, const DAAL_INT* liwork,
    DAAL_INT* info );
void fpk_lapack_avx512_dsyev(const char* jobz, const char* uplo,
    const DAAL_INT* n, double* a, const DAAL_INT* lda, double* w, double* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx512_dsyevd(const char* jobz, const char* uplo,
    const DAAL_INT* n, double* a, const DAAL_INT* lda, double* w, double* work,
    const DAAL_INT* lwork, DAAL_INT* iwork, const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_avx512_dsyevr(const char* jobz, const char* range,
    const char* uplo, const DAAL_INT* n, double* a, const DAAL_INT* lda,
    const double* vl, const double* vu, const DAAL_INT* il, const DAAL_INT* iu,
    const double* abstol, DAAL_INT* m, double* w, double* z, const DAAL_INT* ldz,
    DAAL_INT* isuppz, double* work, const DAAL_INT* lwork, DAAL_INT* iwork,
    const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_avx512_dtrtrs(const char* uplo, const char* trans,
    const char* diag, const DAAL_INT* n, const DAAL_INT* nrhs, const double* a,
    const DAAL_INT* lda, double* b, const DAAL_INT* ldb, DAAL_INT* info );
void fpk_lapack_avx512_sgels(const char* trans, const DAAL_INT* m,
    const DAAL_INT* n, const DAAL_INT* nrhs, float* a, const DAAL_INT* lda,
    float* b, const DAAL_INT* ldb, float* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_avx512_sgeqp3(const DAAL_INT* m, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* jpvt, float* tau, float* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx512_sgeqrf(const DAAL_INT* m, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, float* tau, float* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_avx512_sgerqf(const DAAL_INT* m, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, float* tau, float* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_avx512_sgesvd(const char* jobu, const char* jobvt,
    const DAAL_INT* m, const DAAL_INT* n, float* a, const DAAL_INT* lda, float* s,
    float* u, const DAAL_INT* ldu, float* vt, const DAAL_INT* ldvt, float* work,
    const DAAL_INT* lwork, DAAL_INT* info );
float fpk_lapack_avx512_slange(const char* norm, const DAAL_INT* m,
    const DAAL_INT* n, const float* a, const DAAL_INT* lda, float* work );
void fpk_lapack_avx512_slarnv(const DAAL_INT* idist, DAAL_INT* iseed,
    const DAAL_INT* n, float* x );
void fpk_lapack_avx512_sorgqr(const DAAL_INT* m, const DAAL_INT* n,
    const DAAL_INT* k, float* a, const DAAL_INT* lda, const float* tau,
    float* work, const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx512_sorgrq(const DAAL_INT* m, const DAAL_INT* n,
    const DAAL_INT* k, float* a, const DAAL_INT* lda, const float* tau,
    float* work, const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx512_sormqr(const char* side, const char* trans,
    const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k, const float* a,
    const DAAL_INT* lda, const float* tau, float* c, const DAAL_INT* ldc,
    float* work, const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_avx512_sormrq(const char* side, const char* trans,
    const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k, const float* a,
    const DAAL_INT* lda, const float* tau, float* c, const DAAL_INT* ldc,
    float* work, const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_avx512_spftrf(const char* transr, const char* uplo,
    const DAAL_INT* n, float* a, DAAL_INT* info );
void fpk_lapack_avx512_spotrf(const char* uplo, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_avx512_spotri(const char* uplo, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_avx512_spotrs(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* nrhs, const float* a, const DAAL_INT* lda, float* b,
    const DAAL_INT* ldb, DAAL_INT* info );
void fpk_lapack_avx512_spptrf(const char* uplo, const DAAL_INT* n, float* ap,
    DAAL_INT* info );
void fpk_lapack_avx512_spstrf(const char* uplo, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* piv, DAAL_INT* rank, const float* tol,
    float* work, DAAL_INT* info );
void fpk_lapack_avx512_sspevd(const char* jobz, const char* uplo,
    const DAAL_INT* n, float* ap, float* w, float* z, const DAAL_INT* ldz,
    float* work, const DAAL_INT* lwork, DAAL_INT* iwork, const DAAL_INT* liwork,
    DAAL_INT* info );
void fpk_lapack_avx512_ssyev(const char* jobz, const char* uplo,
    const DAAL_INT* n, float* a, const DAAL_INT* lda, float* w, float* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx512_ssyevd(const char* jobz, const char* uplo,
    const DAAL_INT* n, float* a, const DAAL_INT* lda, float* w, float* work,
    const DAAL_INT* lwork, DAAL_INT* iwork, const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_avx512_ssyevr(const char* jobz, const char* range,
    const char* uplo, const DAAL_INT* n, float* a, const DAAL_INT* lda,
    const float* vl, const float* vu, const DAAL_INT* il, const DAAL_INT* iu,
    const float* abstol, DAAL_INT* m, float* w, float* z, const DAAL_INT* ldz,
    DAAL_INT* isuppz, float* work, const DAAL_INT* lwork, DAAL_INT* iwork,
    const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_avx512_strtrs(const char* uplo, const char* trans,
    const char* diag, const DAAL_INT* n, const DAAL_INT* nrhs, const float* a,
    const DAAL_INT* lda, float* b, const DAAL_INT* ldb, DAAL_INT* info );

void fpk_lapack_avx512_mic_dgels(const char* trans, const DAAL_INT* m,
    const DAAL_INT* n, const DAAL_INT* nrhs, double* a, const DAAL_INT* lda,
    double* b, const DAAL_INT* ldb, double* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_avx512_mic_dgeqp3(const DAAL_INT* m, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* jpvt, double* tau, double* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx512_mic_dgeqrf(const DAAL_INT* m, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, double* tau, double* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_avx512_mic_dgerqf(const DAAL_INT* m, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, double* tau, double* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_avx512_mic_dgesvd(const char* jobu, const char* jobvt,
    const DAAL_INT* m, const DAAL_INT* n, double* a, const DAAL_INT* lda, double* s,
    double* u, const DAAL_INT* ldu, double* vt, const DAAL_INT* ldvt, double* work,
    const DAAL_INT* lwork, DAAL_INT* info );
double fpk_lapack_avx512_mic_dlange(const char* norm, const DAAL_INT* m,
    const DAAL_INT* n, const double* a, const DAAL_INT* lda, double* work );
void fpk_lapack_avx512_mic_dlarnv(const DAAL_INT* idist, DAAL_INT* iseed,
    const DAAL_INT* n, double* x );
void fpk_lapack_avx512_mic_dorgqr(const DAAL_INT* m, const DAAL_INT* n,
    const DAAL_INT* k, double* a, const DAAL_INT* lda, const double* tau,
    double* work, const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx512_mic_dorgrq(const DAAL_INT* m, const DAAL_INT* n,
    const DAAL_INT* k, double* a, const DAAL_INT* lda, const double* tau,
    double* work, const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx512_mic_dormqr(const char* side, const char* trans,
    const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k, const double* a,
    const DAAL_INT* lda, const double* tau, double* c, const DAAL_INT* ldc,
    double* work, const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_avx512_mic_dormrq(const char* side, const char* trans,
    const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k, const double* a,
    const DAAL_INT* lda, const double* tau, double* c, const DAAL_INT* ldc,
    double* work, const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_avx512_mic_dpftrf(const char* transr, const char* uplo,
    const DAAL_INT* n, double* a, DAAL_INT* info );
void fpk_lapack_avx512_mic_dpotrf(const char* uplo, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_avx512_mic_dpotri(const char* uplo, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_avx512_mic_dpotrs(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* nrhs, const double* a, const DAAL_INT* lda, double* b,
    const DAAL_INT* ldb, DAAL_INT* info );
void fpk_lapack_avx512_mic_dpptrf(const char* uplo, const DAAL_INT* n, double* ap,
    DAAL_INT* info );
void fpk_lapack_avx512_mic_dpstrf(const char* uplo, const DAAL_INT* n, double* a,
    const DAAL_INT* lda, DAAL_INT* piv, DAAL_INT* rank, const double* tol,
    double* work, DAAL_INT* info );
void fpk_lapack_avx512_mic_dspevd(const char* jobz, const char* uplo,
    const DAAL_INT* n, double* ap, double* w, double* z, const DAAL_INT* ldz,
    double* work, const DAAL_INT* lwork, DAAL_INT* iwork, const DAAL_INT* liwork,
    DAAL_INT* info );
void fpk_lapack_avx512_mic_dsyev(const char* jobz, const char* uplo,
    const DAAL_INT* n, double* a, const DAAL_INT* lda, double* w, double* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx512_mic_dsyevd(const char* jobz, const char* uplo,
    const DAAL_INT* n, double* a, const DAAL_INT* lda, double* w, double* work,
    const DAAL_INT* lwork, DAAL_INT* iwork, const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_avx512_mic_dsyevr(const char* jobz, const char* range,
    const char* uplo, const DAAL_INT* n, double* a, const DAAL_INT* lda,
    const double* vl, const double* vu, const DAAL_INT* il, const DAAL_INT* iu,
    const double* abstol, DAAL_INT* m, double* w, double* z, const DAAL_INT* ldz,
    DAAL_INT* isuppz, double* work, const DAAL_INT* lwork, DAAL_INT* iwork,
    const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_avx512_mic_dtrtrs(const char* uplo, const char* trans,
    const char* diag, const DAAL_INT* n, const DAAL_INT* nrhs, const double* a,
    const DAAL_INT* lda, double* b, const DAAL_INT* ldb, DAAL_INT* info );
void fpk_lapack_avx512_mic_sgels(const char* trans, const DAAL_INT* m,
    const DAAL_INT* n, const DAAL_INT* nrhs, float* a, const DAAL_INT* lda,
    float* b, const DAAL_INT* ldb, float* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_avx512_mic_sgeqp3(const DAAL_INT* m, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* jpvt, float* tau, float* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx512_mic_sgeqrf(const DAAL_INT* m, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, float* tau, float* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_avx512_mic_sgerqf(const DAAL_INT* m, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, float* tau, float* work, const DAAL_INT* lwork,
    DAAL_INT* info );
void fpk_lapack_avx512_mic_sgesvd(const char* jobu, const char* jobvt,
    const DAAL_INT* m, const DAAL_INT* n, float* a, const DAAL_INT* lda, float* s,
    float* u, const DAAL_INT* ldu, float* vt, const DAAL_INT* ldvt, float* work,
    const DAAL_INT* lwork, DAAL_INT* info );
float fpk_lapack_avx512_mic_slange(const char* norm, const DAAL_INT* m,
    const DAAL_INT* n, const float* a, const DAAL_INT* lda, float* work );
void fpk_lapack_avx512_mic_slarnv(const DAAL_INT* idist, DAAL_INT* iseed,
    const DAAL_INT* n, float* x );
void fpk_lapack_avx512_mic_sorgqr(const DAAL_INT* m, const DAAL_INT* n,
    const DAAL_INT* k, float* a, const DAAL_INT* lda, const float* tau,
    float* work, const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx512_mic_sorgrq(const DAAL_INT* m, const DAAL_INT* n,
    const DAAL_INT* k, float* a, const DAAL_INT* lda, const float* tau,
    float* work, const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx512_mic_sormqr(const char* side, const char* trans,
    const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k, const float* a,
    const DAAL_INT* lda, const float* tau, float* c, const DAAL_INT* ldc,
    float* work, const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_avx512_mic_sormrq(const char* side, const char* trans,
    const DAAL_INT* m, const DAAL_INT* n, const DAAL_INT* k, const float* a,
    const DAAL_INT* lda, const float* tau, float* c, const DAAL_INT* ldc,
    float* work, const DAAL_INT* lwork, DAAL_INT* info, int, int );
void fpk_lapack_avx512_mic_spftrf(const char* transr, const char* uplo,
    const DAAL_INT* n, float* a, DAAL_INT* info );
void fpk_lapack_avx512_mic_spotrf(const char* uplo, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_avx512_mic_spotri(const char* uplo, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* info );
void fpk_lapack_avx512_mic_spotrs(const char* uplo, const DAAL_INT* n,
    const DAAL_INT* nrhs, const float* a, const DAAL_INT* lda, float* b,
    const DAAL_INT* ldb, DAAL_INT* info );
void fpk_lapack_avx512_mic_spptrf(const char* uplo, const DAAL_INT* n, float* ap,
    DAAL_INT* info );
void fpk_lapack_avx512_mic_spstrf(const char* uplo, const DAAL_INT* n, float* a,
    const DAAL_INT* lda, DAAL_INT* piv, DAAL_INT* rank, const float* tol,
    float* work, DAAL_INT* info );
void fpk_lapack_avx512_mic_sspevd(const char* jobz, const char* uplo,
    const DAAL_INT* n, float* ap, float* w, float* z, const DAAL_INT* ldz,
    float* work, const DAAL_INT* lwork, DAAL_INT* iwork, const DAAL_INT* liwork,
    DAAL_INT* info );
void fpk_lapack_avx512_mic_ssyev(const char* jobz, const char* uplo,
    const DAAL_INT* n, float* a, const DAAL_INT* lda, float* w, float* work,
    const DAAL_INT* lwork, DAAL_INT* info );
void fpk_lapack_avx512_mic_ssyevd(const char* jobz, const char* uplo,
    const DAAL_INT* n, float* a, const DAAL_INT* lda, float* w, float* work,
    const DAAL_INT* lwork, DAAL_INT* iwork, const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_avx512_mic_ssyevr(const char* jobz, const char* range,
    const char* uplo, const DAAL_INT* n, float* a, const DAAL_INT* lda,
    const float* vl, const float* vu, const DAAL_INT* il, const DAAL_INT* iu,
    const float* abstol, DAAL_INT* m, float* w, float* z, const DAAL_INT* ldz,
    DAAL_INT* isuppz, float* work, const DAAL_INT* lwork, DAAL_INT* iwork,
    const DAAL_INT* liwork, DAAL_INT* info );
void fpk_lapack_avx512_mic_strtrs(const char* uplo, const char* trans,
    const char* diag, const DAAL_INT* n, const DAAL_INT* nrhs, const float* a,
    const DAAL_INT* lda, float* b, const DAAL_INT* ldb, DAAL_INT* info );


void fpk_pds_sse2_pardiso(_MKL_DSS_HANDLE_t, const DAAL_INT *, const DAAL_INT *,
    const DAAL_INT *, const DAAL_INT *, const DAAL_INT *, const void *,
    const DAAL_INT *, const DAAL_INT *, DAAL_INT *, const DAAL_INT *, DAAL_INT *,
    const DAAL_INT *, void *, void *, DAAL_INT * );
DAAL_INT fpk_pds_sse2_pardiso_getenv(const _MKL_DSS_HANDLE_t,
    const enum PARDISO_ENV_PARAM*, char* );
DAAL_INT fpk_pds_sse2_pardiso_setenv(_MKL_DSS_HANDLE_t,
    const enum PARDISO_ENV_PARAM*, const char* );

void fpk_pds_ssse3_pardiso(_MKL_DSS_HANDLE_t, const DAAL_INT *, const DAAL_INT *,
    const DAAL_INT *, const DAAL_INT *, const DAAL_INT *, const void *,
    const DAAL_INT *, const DAAL_INT *, DAAL_INT *, const DAAL_INT *, DAAL_INT *,
    const DAAL_INT *, void *, void *, DAAL_INT * );
DAAL_INT fpk_pds_ssse3_pardiso_getenv(const _MKL_DSS_HANDLE_t,
    const enum PARDISO_ENV_PARAM*, char* );
DAAL_INT fpk_pds_ssse3_pardiso_setenv(_MKL_DSS_HANDLE_t,
    const enum PARDISO_ENV_PARAM*, const char* );

void fpk_pds_sse42_pardiso(_MKL_DSS_HANDLE_t, const DAAL_INT *, const DAAL_INT *,
    const DAAL_INT *, const DAAL_INT *, const DAAL_INT *, const void *,
    const DAAL_INT *, const DAAL_INT *, DAAL_INT *, const DAAL_INT *, DAAL_INT *,
    const DAAL_INT *, void *, void *, DAAL_INT * );
DAAL_INT fpk_pds_sse42_pardiso_getenv(const _MKL_DSS_HANDLE_t,
    const enum PARDISO_ENV_PARAM*, char* );
DAAL_INT fpk_pds_sse42_pardiso_setenv(_MKL_DSS_HANDLE_t,
    const enum PARDISO_ENV_PARAM*, const char* );

void fpk_pds_avx_pardiso(_MKL_DSS_HANDLE_t, const DAAL_INT *, const DAAL_INT *,
    const DAAL_INT *, const DAAL_INT *, const DAAL_INT *, const void *,
    const DAAL_INT *, const DAAL_INT *, DAAL_INT *, const DAAL_INT *, DAAL_INT *,
    const DAAL_INT *, void *, void *, DAAL_INT * );
DAAL_INT fpk_pds_avx_pardiso_getenv(const _MKL_DSS_HANDLE_t,
    const enum PARDISO_ENV_PARAM*, char* );
DAAL_INT fpk_pds_avx_pardiso_setenv(_MKL_DSS_HANDLE_t,
    const enum PARDISO_ENV_PARAM*, const char* );

void fpk_pds_avx2_pardiso(_MKL_DSS_HANDLE_t, const DAAL_INT *, const DAAL_INT *,
    const DAAL_INT *, const DAAL_INT *, const DAAL_INT *, const void *,
    const DAAL_INT *, const DAAL_INT *, DAAL_INT *, const DAAL_INT *, DAAL_INT *,
    const DAAL_INT *, void *, void *, DAAL_INT * );
DAAL_INT fpk_pds_avx2_pardiso_getenv(const _MKL_DSS_HANDLE_t,
    const enum PARDISO_ENV_PARAM*, char* );
DAAL_INT fpk_pds_avx2_pardiso_setenv(_MKL_DSS_HANDLE_t,
    const enum PARDISO_ENV_PARAM*, const char* );

void fpk_pds_avx512_pardiso(_MKL_DSS_HANDLE_t, const DAAL_INT *, const DAAL_INT *,
    const DAAL_INT *, const DAAL_INT *, const DAAL_INT *, const void *,
    const DAAL_INT *, const DAAL_INT *, DAAL_INT *, const DAAL_INT *, DAAL_INT *,
    const DAAL_INT *, void *, void *, DAAL_INT * );
DAAL_INT fpk_pds_avx512_pardiso_getenv(const _MKL_DSS_HANDLE_t,
    const enum PARDISO_ENV_PARAM*, char* );
DAAL_INT fpk_pds_avx512_pardiso_setenv(_MKL_DSS_HANDLE_t,
    const enum PARDISO_ENV_PARAM*, const char* );

void fpk_pds_avx512_mic_pardiso(_MKL_DSS_HANDLE_t, const DAAL_INT *,
    const DAAL_INT *, const DAAL_INT *, const DAAL_INT *, const DAAL_INT *,
    const void *, const DAAL_INT *, const DAAL_INT *, DAAL_INT *, const DAAL_INT *,
    DAAL_INT *, const DAAL_INT *, void *, void *, DAAL_INT * );
DAAL_INT fpk_pds_avx512_mic_pardiso_getenv(const _MKL_DSS_HANDLE_t,
    const enum PARDISO_ENV_PARAM*, char* );
DAAL_INT fpk_pds_avx512_mic_pardiso_setenv(_MKL_DSS_HANDLE_t,
    const enum PARDISO_ENV_PARAM*, const char* );


void* fpk_serv_allocate(size_t size, int alignment);
int fpk_serv_cpuisknm(void);
void fpk_serv_deallocate(void *ptr);

int fpk_serv_enable_instructions(int);
void fpk_serv_free(void *ptr);

int fpk_serv_get_ht(void);

int fpk_serv_get_max_threads(void);

int fpk_serv_get_ncorespercpu(void);
int fpk_serv_get_ncpus(void);
int fpk_serv_get_nlogicalcores(void);

void* fpk_serv_malloc(size_t size, int align);

int fpk_serv_memcpy_s(void *dest, size_t dmax, const void *src, size_t slen);

int fpk_serv_register_jit_function(void *addr, size_t size, const char *name);

int fpk_serv_set_memory_limit(int mem_type, size_t limit);
void fpk_serv_set_num_threads(int nth);
int fpk_serv_set_num_threads_local(int nth);

int fpk_serv_strncat_s(char *dest, size_t dmax, const char *src, size_t slen);
int fpk_serv_strncpy_s(char *dest, size_t dmax, const char *src, size_t slen);
size_t fpk_serv_strnlen_s(const char *s, size_t smax);


void fpk_spblas_sse2_mkl_dcsrmm(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const char *matdescra, const double *val, const DAAL_INT *indx,
    const DAAL_INT *pntrb, const DAAL_INT *pntre, const double *b,
    const DAAL_INT *ldb, const double *beta, double *c, const DAAL_INT *ldc);
void fpk_spblas_sse2_mkl_dcsrmultd(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, double *a, DAAL_INT *ja, DAAL_INT *ia,
    double *b, DAAL_INT *jb, DAAL_INT *ib, double *c, DAAL_INT *ldc);
void fpk_spblas_sse2_mkl_dcsrmv(const char *transa, const DAAL_INT *m,
    const DAAL_INT *k, const double *alpha, const char *matdescra,
    const double *val, const DAAL_INT *indx, const DAAL_INT *pntrb,
    const DAAL_INT *pntre, const double *x, const double *beta, double *y);
void fpk_spblas_sse2_mkl_ddnscsr(const DAAL_INT *job, const DAAL_INT *m,
    const DAAL_INT *n, double *Adns, const DAAL_INT *lda, double *Acsr,
    DAAL_INT *AJ, DAAL_INT *AI, DAAL_INT *info);
void fpk_spblas_sse2_mkl_scsrmm(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const char *matdescra, const float *val, const DAAL_INT *indx,
    const DAAL_INT *pntrb, const DAAL_INT *pntre, const float *b,
    const DAAL_INT *ldb, const float *beta, float *c, const DAAL_INT *ldc);
void fpk_spblas_sse2_mkl_scsrmultd(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, float *a, DAAL_INT *ja, DAAL_INT *ia,
    float *b, DAAL_INT *jb, DAAL_INT *ib, float *c, DAAL_INT *ldc);
void fpk_spblas_sse2_mkl_scsrmv(const char *transa, const DAAL_INT *m,
    const DAAL_INT *k, const float *alpha, const char *matdescra,
    const float *val, const DAAL_INT *indx, const DAAL_INT *pntrb,
    const DAAL_INT *pntre, const float *x, const float *beta, float *y);
void fpk_spblas_sse2_mkl_sdnscsr(const DAAL_INT *job, const DAAL_INT *m,
    const DAAL_INT *n, float *Adns, const DAAL_INT *lda, float *Acsr, DAAL_INT *AJ,
    DAAL_INT *AI, DAAL_INT *info);

void fpk_spblas_ssse3_mkl_dcsrmm(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const char *matdescra, const double *val, const DAAL_INT *indx,
    const DAAL_INT *pntrb, const DAAL_INT *pntre, const double *b,
    const DAAL_INT *ldb, const double *beta, double *c, const DAAL_INT *ldc);
void fpk_spblas_ssse3_mkl_dcsrmultd(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, double *a, DAAL_INT *ja, DAAL_INT *ia,
    double *b, DAAL_INT *jb, DAAL_INT *ib, double *c, DAAL_INT *ldc);
void fpk_spblas_ssse3_mkl_dcsrmv(const char *transa, const DAAL_INT *m,
    const DAAL_INT *k, const double *alpha, const char *matdescra,
    const double *val, const DAAL_INT *indx, const DAAL_INT *pntrb,
    const DAAL_INT *pntre, const double *x, const double *beta, double *y);
void fpk_spblas_ssse3_mkl_ddnscsr(const DAAL_INT *job, const DAAL_INT *m,
    const DAAL_INT *n, double *Adns, const DAAL_INT *lda, double *Acsr,
    DAAL_INT *AJ, DAAL_INT *AI, DAAL_INT *info);
void fpk_spblas_ssse3_mkl_scsrmm(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const char *matdescra, const float *val, const DAAL_INT *indx,
    const DAAL_INT *pntrb, const DAAL_INT *pntre, const float *b,
    const DAAL_INT *ldb, const float *beta, float *c, const DAAL_INT *ldc);
void fpk_spblas_ssse3_mkl_scsrmultd(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, float *a, DAAL_INT *ja, DAAL_INT *ia,
    float *b, DAAL_INT *jb, DAAL_INT *ib, float *c, DAAL_INT *ldc);
void fpk_spblas_ssse3_mkl_scsrmv(const char *transa, const DAAL_INT *m,
    const DAAL_INT *k, const float *alpha, const char *matdescra,
    const float *val, const DAAL_INT *indx, const DAAL_INT *pntrb,
    const DAAL_INT *pntre, const float *x, const float *beta, float *y);
void fpk_spblas_ssse3_mkl_sdnscsr(const DAAL_INT *job, const DAAL_INT *m,
    const DAAL_INT *n, float *Adns, const DAAL_INT *lda, float *Acsr, DAAL_INT *AJ,
    DAAL_INT *AI, DAAL_INT *info);

void fpk_spblas_sse42_mkl_dcsrmm(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const char *matdescra, const double *val, const DAAL_INT *indx,
    const DAAL_INT *pntrb, const DAAL_INT *pntre, const double *b,
    const DAAL_INT *ldb, const double *beta, double *c, const DAAL_INT *ldc);
void fpk_spblas_sse42_mkl_dcsrmultd(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, double *a, DAAL_INT *ja, DAAL_INT *ia,
    double *b, DAAL_INT *jb, DAAL_INT *ib, double *c, DAAL_INT *ldc);
void fpk_spblas_sse42_mkl_dcsrmv(const char *transa, const DAAL_INT *m,
    const DAAL_INT *k, const double *alpha, const char *matdescra,
    const double *val, const DAAL_INT *indx, const DAAL_INT *pntrb,
    const DAAL_INT *pntre, const double *x, const double *beta, double *y);
void fpk_spblas_sse42_mkl_ddnscsr(const DAAL_INT *job, const DAAL_INT *m,
    const DAAL_INT *n, double *Adns, const DAAL_INT *lda, double *Acsr,
    DAAL_INT *AJ, DAAL_INT *AI, DAAL_INT *info);
void fpk_spblas_sse42_mkl_scsrmm(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const char *matdescra, const float *val, const DAAL_INT *indx,
    const DAAL_INT *pntrb, const DAAL_INT *pntre, const float *b,
    const DAAL_INT *ldb, const float *beta, float *c, const DAAL_INT *ldc);
void fpk_spblas_sse42_mkl_scsrmultd(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, float *a, DAAL_INT *ja, DAAL_INT *ia,
    float *b, DAAL_INT *jb, DAAL_INT *ib, float *c, DAAL_INT *ldc);
void fpk_spblas_sse42_mkl_scsrmv(const char *transa, const DAAL_INT *m,
    const DAAL_INT *k, const float *alpha, const char *matdescra,
    const float *val, const DAAL_INT *indx, const DAAL_INT *pntrb,
    const DAAL_INT *pntre, const float *x, const float *beta, float *y);
void fpk_spblas_sse42_mkl_sdnscsr(const DAAL_INT *job, const DAAL_INT *m,
    const DAAL_INT *n, float *Adns, const DAAL_INT *lda, float *Acsr, DAAL_INT *AJ,
    DAAL_INT *AI, DAAL_INT *info);

void fpk_spblas_avx_mkl_dcsrmm(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const char *matdescra, const double *val, const DAAL_INT *indx,
    const DAAL_INT *pntrb, const DAAL_INT *pntre, const double *b,
    const DAAL_INT *ldb, const double *beta, double *c, const DAAL_INT *ldc);
void fpk_spblas_avx_mkl_dcsrmultd(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, double *a, DAAL_INT *ja, DAAL_INT *ia,
    double *b, DAAL_INT *jb, DAAL_INT *ib, double *c, DAAL_INT *ldc);
void fpk_spblas_avx_mkl_dcsrmv(const char *transa, const DAAL_INT *m,
    const DAAL_INT *k, const double *alpha, const char *matdescra,
    const double *val, const DAAL_INT *indx, const DAAL_INT *pntrb,
    const DAAL_INT *pntre, const double *x, const double *beta, double *y);
void fpk_spblas_avx_mkl_ddnscsr(const DAAL_INT *job, const DAAL_INT *m,
    const DAAL_INT *n, double *Adns, const DAAL_INT *lda, double *Acsr,
    DAAL_INT *AJ, DAAL_INT *AI, DAAL_INT *info);
void fpk_spblas_avx_mkl_scsrmm(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const char *matdescra, const float *val, const DAAL_INT *indx,
    const DAAL_INT *pntrb, const DAAL_INT *pntre, const float *b,
    const DAAL_INT *ldb, const float *beta, float *c, const DAAL_INT *ldc);
void fpk_spblas_avx_mkl_scsrmultd(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, float *a, DAAL_INT *ja, DAAL_INT *ia,
    float *b, DAAL_INT *jb, DAAL_INT *ib, float *c, DAAL_INT *ldc);
void fpk_spblas_avx_mkl_scsrmv(const char *transa, const DAAL_INT *m,
    const DAAL_INT *k, const float *alpha, const char *matdescra,
    const float *val, const DAAL_INT *indx, const DAAL_INT *pntrb,
    const DAAL_INT *pntre, const float *x, const float *beta, float *y);
void fpk_spblas_avx_mkl_sdnscsr(const DAAL_INT *job, const DAAL_INT *m,
    const DAAL_INT *n, float *Adns, const DAAL_INT *lda, float *Acsr, DAAL_INT *AJ,
    DAAL_INT *AI, DAAL_INT *info);

void fpk_spblas_avx2_mkl_dcsrmm(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const char *matdescra, const double *val, const DAAL_INT *indx,
    const DAAL_INT *pntrb, const DAAL_INT *pntre, const double *b,
    const DAAL_INT *ldb, const double *beta, double *c, const DAAL_INT *ldc);
void fpk_spblas_avx2_mkl_dcsrmultd(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, double *a, DAAL_INT *ja, DAAL_INT *ia,
    double *b, DAAL_INT *jb, DAAL_INT *ib, double *c, DAAL_INT *ldc);
void fpk_spblas_avx2_mkl_dcsrmv(const char *transa, const DAAL_INT *m,
    const DAAL_INT *k, const double *alpha, const char *matdescra,
    const double *val, const DAAL_INT *indx, const DAAL_INT *pntrb,
    const DAAL_INT *pntre, const double *x, const double *beta, double *y);
void fpk_spblas_avx2_mkl_ddnscsr(const DAAL_INT *job, const DAAL_INT *m,
    const DAAL_INT *n, double *Adns, const DAAL_INT *lda, double *Acsr,
    DAAL_INT *AJ, DAAL_INT *AI, DAAL_INT *info);
void fpk_spblas_avx2_mkl_scsrmm(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const char *matdescra, const float *val, const DAAL_INT *indx,
    const DAAL_INT *pntrb, const DAAL_INT *pntre, const float *b,
    const DAAL_INT *ldb, const float *beta, float *c, const DAAL_INT *ldc);
void fpk_spblas_avx2_mkl_scsrmultd(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, float *a, DAAL_INT *ja, DAAL_INT *ia,
    float *b, DAAL_INT *jb, DAAL_INT *ib, float *c, DAAL_INT *ldc);
void fpk_spblas_avx2_mkl_scsrmv(const char *transa, const DAAL_INT *m,
    const DAAL_INT *k, const float *alpha, const char *matdescra,
    const float *val, const DAAL_INT *indx, const DAAL_INT *pntrb,
    const DAAL_INT *pntre, const float *x, const float *beta, float *y);
void fpk_spblas_avx2_mkl_sdnscsr(const DAAL_INT *job, const DAAL_INT *m,
    const DAAL_INT *n, float *Adns, const DAAL_INT *lda, float *Acsr, DAAL_INT *AJ,
    DAAL_INT *AI, DAAL_INT *info);

void fpk_spblas_avx512_mkl_dcsrmm(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const char *matdescra, const double *val, const DAAL_INT *indx,
    const DAAL_INT *pntrb, const DAAL_INT *pntre, const double *b,
    const DAAL_INT *ldb, const double *beta, double *c, const DAAL_INT *ldc);
void fpk_spblas_avx512_mkl_dcsrmultd(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, double *a, DAAL_INT *ja, DAAL_INT *ia,
    double *b, DAAL_INT *jb, DAAL_INT *ib, double *c, DAAL_INT *ldc);
void fpk_spblas_avx512_mkl_dcsrmv(const char *transa, const DAAL_INT *m,
    const DAAL_INT *k, const double *alpha, const char *matdescra,
    const double *val, const DAAL_INT *indx, const DAAL_INT *pntrb,
    const DAAL_INT *pntre, const double *x, const double *beta, double *y);
void fpk_spblas_avx512_mkl_ddnscsr(const DAAL_INT *job, const DAAL_INT *m,
    const DAAL_INT *n, double *Adns, const DAAL_INT *lda, double *Acsr,
    DAAL_INT *AJ, DAAL_INT *AI, DAAL_INT *info);
void fpk_spblas_avx512_mkl_scsrmm(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const char *matdescra, const float *val, const DAAL_INT *indx,
    const DAAL_INT *pntrb, const DAAL_INT *pntre, const float *b,
    const DAAL_INT *ldb, const float *beta, float *c, const DAAL_INT *ldc);
void fpk_spblas_avx512_mkl_scsrmultd(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, float *a, DAAL_INT *ja, DAAL_INT *ia,
    float *b, DAAL_INT *jb, DAAL_INT *ib, float *c, DAAL_INT *ldc);
void fpk_spblas_avx512_mkl_scsrmv(const char *transa, const DAAL_INT *m,
    const DAAL_INT *k, const float *alpha, const char *matdescra,
    const float *val, const DAAL_INT *indx, const DAAL_INT *pntrb,
    const DAAL_INT *pntre, const float *x, const float *beta, float *y);
void fpk_spblas_avx512_mkl_sdnscsr(const DAAL_INT *job, const DAAL_INT *m,
    const DAAL_INT *n, float *Adns, const DAAL_INT *lda, float *Acsr, DAAL_INT *AJ,
    DAAL_INT *AI, DAAL_INT *info);

void fpk_spblas_avx512_mic_mkl_dcsrmm(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, const double *alpha,
    const char *matdescra, const double *val, const DAAL_INT *indx,
    const DAAL_INT *pntrb, const DAAL_INT *pntre, const double *b,
    const DAAL_INT *ldb, const double *beta, double *c, const DAAL_INT *ldc);
void fpk_spblas_avx512_mic_mkl_dcsrmultd(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, double *a, DAAL_INT *ja, DAAL_INT *ia,
    double *b, DAAL_INT *jb, DAAL_INT *ib, double *c, DAAL_INT *ldc);
void fpk_spblas_avx512_mic_mkl_dcsrmv(const char *transa, const DAAL_INT *m,
    const DAAL_INT *k, const double *alpha, const char *matdescra,
    const double *val, const DAAL_INT *indx, const DAAL_INT *pntrb,
    const DAAL_INT *pntre, const double *x, const double *beta, double *y);
void fpk_spblas_avx512_mic_mkl_ddnscsr(const DAAL_INT *job, const DAAL_INT *m,
    const DAAL_INT *n, double *Adns, const DAAL_INT *lda, double *Acsr,
    DAAL_INT *AJ, DAAL_INT *AI, DAAL_INT *info);
void fpk_spblas_avx512_mic_mkl_scsrmm(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, const float *alpha,
    const char *matdescra, const float *val, const DAAL_INT *indx,
    const DAAL_INT *pntrb, const DAAL_INT *pntre, const float *b,
    const DAAL_INT *ldb, const float *beta, float *c, const DAAL_INT *ldc);
void fpk_spblas_avx512_mic_mkl_scsrmultd(const char *transa, const DAAL_INT *m,
    const DAAL_INT *n, const DAAL_INT *k, float *a, DAAL_INT *ja, DAAL_INT *ia,
    float *b, DAAL_INT *jb, DAAL_INT *ib, float *c, DAAL_INT *ldc);
void fpk_spblas_avx512_mic_mkl_scsrmv(const char *transa, const DAAL_INT *m,
    const DAAL_INT *k, const float *alpha, const char *matdescra,
    const float *val, const DAAL_INT *indx, const DAAL_INT *pntrb,
    const DAAL_INT *pntre, const float *x, const float *beta, float *y);
void fpk_spblas_avx512_mic_mkl_sdnscsr(const DAAL_INT *job, const DAAL_INT *m,
    const DAAL_INT *n, float *Adns, const DAAL_INT *lda, float *Acsr, DAAL_INT *AJ,
    DAAL_INT *AI, DAAL_INT *info);


void fpk_trans_sse2_mkl_domatcopy(char ordering, char trans, size_t rows,
    size_t cols, const double alpha, const double * A, size_t lda, double * B,
    size_t ldb);
void fpk_trans_sse2_mkl_somatcopy(char ordering, char trans, size_t rows,
    size_t cols, const float alpha, const float * A, size_t lda, float * B,
    size_t ldb);

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

void fpk_trans_avx512_mic_mkl_domatcopy(char ordering, char trans, size_t rows,
    size_t cols, const double alpha, const double * A, size_t lda, double * B,
    size_t ldb);
void fpk_trans_avx512_mic_mkl_somatcopy(char ordering, char trans, size_t rows,
    size_t cols, const float alpha, const float * A, size_t lda, float * B,
    size_t ldb);


#if defined(__cplusplus)
}
#endif

#endif /*MKL_DAL_H*/
