/* file: core_threading_win_dll.cpp */
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
//  Implementation of "stubs" for threading layer functions for win dll case.
//--
*/

#include <windows.h>
#include <stdio.h>
#include "threading.h"
#include "env_detect.h"
#include "mkl_daal.h"

static HMODULE daal_thr_dll_handle = NULL;
daal::services::Environment::LibraryThreadingType __daal_serv_get_thr_set();

static void load_daal_thr_dll(void)
{
    if(daal_thr_dll_handle != NULL) { return; }

    switch(__daal_serv_get_thr_set())
    {
        case daal::services::Environment::MultiThreaded: {
            daal_thr_dll_handle = LoadLibrary( "daal_thread.dll" );
            if(daal_thr_dll_handle != NULL) { return; }

            printf("Intel DAAL FATAL ERROR: Cannot load libdaal_thread.dll.\n");
            exit(1);

            break;
        }
        case daal::services::Environment::SingleThreaded: {
            daal_thr_dll_handle = LoadLibrary( "daal_sequential.dll" );
            if(daal_thr_dll_handle != NULL) { return; }

            printf("Intel DAAL FATAL ERROR: Cannot load libdaal_sequential.dll.\n");
            exit(1);

            break;
        }
        default: {
            daal_thr_dll_handle = LoadLibrary( "daal_thread.dll" );
            if(daal_thr_dll_handle != NULL) { return; }

            daal_thr_dll_handle = LoadLibrary( "daal_sequential.dll" );
            if(daal_thr_dll_handle != NULL) { return; }

            printf("Intel DAAL FATAL ERROR: Cannot load neither libdaal_thread.dll nor libdaal_sequential.dll.\n");
            exit(1);
        }
    }
}

FARPROC load_daal_thr_func(char *ordinal)
{
    FARPROC FuncAddress;

    if(daal_thr_dll_handle == NULL)
    {
        printf("Intel DAAL FATAL ERROR: Cannot load \"%s\" function because threaded layer DLL isn`t loaded.\n", ordinal);
        exit(1);
    }

    FuncAddress = GetProcAddress(daal_thr_dll_handle, ordinal);
    if(FuncAddress == NULL)
    {
        printf("Intel DAAL FATAL ERROR: Cannot load \"%s\" function.\n", ordinal);
        exit(1);
    }

    return FuncAddress;
}

typedef void (* _daal_threader_for_t)(int , int , const void *, daal::functype );
typedef void (* _daal_threader_for_blocked_t)(int , int , const void *, daal::functype2 );
typedef int (* _daal_threader_get_max_threads_t)(void);
typedef void *(* _daal_get_tls_ptr_t)(void *, daal::tls_functype );
typedef void (* _daal_del_tls_ptr_t)(void *);
typedef void *(* _daal_get_tls_local_t)(void *);
typedef void (* _daal_reduce_tls_t)(void *, void *, daal::tls_reduce_functype );
typedef bool (* _daal_is_in_parallel_t)();
typedef size_t (* _setNumberOfThreads_t)(const size_t, void**);

static _daal_threader_for_t _daal_threader_for_ptr = NULL;
static _daal_threader_for_blocked_t _daal_threader_for_blocked_ptr = NULL;
static _daal_threader_get_max_threads_t _daal_threader_get_max_threads_ptr = NULL;
static _daal_get_tls_ptr_t _daal_get_tls_ptr_ptr = NULL;
static _daal_del_tls_ptr_t _daal_del_tls_ptr_ptr = NULL;
static _daal_get_tls_local_t _daal_get_tls_local_ptr = NULL;
static _daal_reduce_tls_t _daal_reduce_tls_ptr = NULL;
static _daal_is_in_parallel_t _daal_is_in_parallel_ptr = NULL;
static _setNumberOfThreads_t _setNumberOfThreads_ptr = NULL;

DAAL_EXPORT void _daal_threader_for(int n, int threads_request, const void *a, daal::functype func)
{
    load_daal_thr_dll();
    if(_daal_threader_for_ptr == NULL) { _daal_threader_for_ptr = (_daal_threader_for_t)load_daal_thr_func("_daal_threader_for"); }
    _daal_threader_for_ptr(n, threads_request, a, func);
}

DAAL_EXPORT void _daal_threader_for_blocked(int n, int threads_request, const void *a, daal::functype2 func)
{
    load_daal_thr_dll();
    if(_daal_threader_for_blocked_ptr == NULL)
    {
        _daal_threader_for_blocked_ptr
            = (_daal_threader_for_blocked_t)load_daal_thr_func("_daal_threader_for_blocked");
    }
    _daal_threader_for_blocked_ptr(n, threads_request, a, func);
}

DAAL_EXPORT int _daal_threader_get_max_threads()
{
    load_daal_thr_dll();
    if(_daal_threader_get_max_threads_ptr == NULL)
    {
        _daal_threader_get_max_threads_ptr
            = (_daal_threader_get_max_threads_t)load_daal_thr_func("_daal_threader_get_max_threads");
    }
    return _daal_threader_get_max_threads_ptr();
}

DAAL_EXPORT void *_daal_get_tls_ptr(void *a, daal::tls_functype func)
{
    load_daal_thr_dll();
    if(_daal_get_tls_ptr_ptr == NULL) { _daal_get_tls_ptr_ptr = (_daal_get_tls_ptr_t)load_daal_thr_func("_daal_get_tls_ptr"); }
    return _daal_get_tls_ptr_ptr(a, func);
}

DAAL_EXPORT void _daal_del_tls_ptr(void *tlsPtr)
{
    load_daal_thr_dll();
    if(_daal_del_tls_ptr_ptr == NULL) { _daal_del_tls_ptr_ptr = (_daal_del_tls_ptr_t)load_daal_thr_func("_daal_del_tls_ptr"); }
    _daal_del_tls_ptr_ptr(tlsPtr);
}

DAAL_EXPORT void *_daal_get_tls_local(void *tlsPtr)
{
    load_daal_thr_dll();
    if(_daal_get_tls_local_ptr == NULL) { _daal_get_tls_local_ptr = (_daal_get_tls_local_t)load_daal_thr_func("_daal_get_tls_local"); }
    return _daal_get_tls_local_ptr(tlsPtr);
}

DAAL_EXPORT void _daal_reduce_tls(void *tlsPtr, void *a, daal::tls_reduce_functype func)
{
    load_daal_thr_dll();
    if(_daal_reduce_tls_ptr == NULL) { _daal_reduce_tls_ptr = (_daal_reduce_tls_t)load_daal_thr_func("_daal_reduce_tls"); }
    _daal_reduce_tls_ptr(tlsPtr, a, func);
}

DAAL_EXPORT bool _daal_is_in_parallel()
{
    load_daal_thr_dll();
    if(_daal_is_in_parallel_ptr == NULL) { _daal_is_in_parallel_ptr = (_daal_is_in_parallel_t)load_daal_thr_func("_daal_is_in_parallel"); }
    return _daal_is_in_parallel_ptr();
}

DAAL_EXPORT size_t _setNumberOfThreads(const size_t numThreads, void** init)
{
    load_daal_thr_dll();
    if(_setNumberOfThreads_ptr == NULL) { _setNumberOfThreads_ptr = (_setNumberOfThreads_t)load_daal_thr_func("_setNumberOfThreads"); }
    return _setNumberOfThreads_ptr(numThreads, init);
}

#define CALL_VOID_FUNC_FROM_DLL(fn_dpref,fn_name,argdecl,argcall)                 \
    typedef void (* ##fn_dpref##fn_name##_t)##argdecl;                            \
    static fn_dpref##fn_name##_t fn_dpref##fn_name##_ptr=NULL;                    \
    CALL_VOID_FUNC_FROM_DLL_CPU(fn_dpref,avx512_,fn_name,argdecl,argcall)         \
    CALL_VOID_FUNC_FROM_DLL_CPU_MIC(fn_dpref,avx512_mic_,fn_name,argdecl,argcall) \
    CALL_VOID_FUNC_FROM_DLL_CPU(fn_dpref,avx2_,fn_name,argdecl,argcall)           \
    CALL_VOID_FUNC_FROM_DLL_CPU(fn_dpref,avx_,fn_name,argdecl,argcall)            \
    CALL_VOID_FUNC_FROM_DLL_CPU(fn_dpref,sse42_,fn_name,argdecl,argcall)          \
    CALL_VOID_FUNC_FROM_DLL_CPU(fn_dpref,ssse3_,fn_name,argdecl,argcall)          \
    CALL_VOID_FUNC_FROM_DLL_CPU(fn_dpref,sse2_,fn_name,argdecl,argcall)

#define CALL_VOID_FUNC_FROM_DLL_CPU(fn_dpref,fn_cpu,fn_name,argdecl,argcall)                               \
void  fn_dpref##fn_cpu##fn_name##argdecl                                                                   \
{                                                                                                          \
    load_daal_thr_dll();                                                                                   \
    if(##fn_dpref##fn_name##_ptr == NULL) {                                                                \
        ##fn_dpref##fn_name##_ptr = (##fn_dpref##fn_name##_t)load_daal_thr_func(#fn_dpref#fn_cpu#fn_name); \
    }                                                                                                      \
    ##fn_dpref##fn_name##_ptr##argcall;                                                                    \
}

#if defined(_WIN64)
    #define CALL_VOID_FUNC_FROM_DLL_CPU_MIC(fn_dpref,fn_cpu,fn_name,argdecl,argcall)                           \
    void  fn_dpref##fn_cpu##fn_name##argdecl                                                                   \
    {                                                                                                          \
        load_daal_thr_dll();                                                                                   \
        if(##fn_dpref##fn_name##_ptr == NULL) {                                                                \
            ##fn_dpref##fn_name##_ptr = (##fn_dpref##fn_name##_t)load_daal_thr_func(#fn_dpref#fn_cpu#fn_name); \
        }                                                                                                      \
        ##fn_dpref##fn_name##_ptr##argcall;                                                                    \
    }
#else
    #define CALL_VOID_FUNC_FROM_DLL_CPU_MIC(fn_dpref,fn_cpu,fn_name,argdecl,argcall)
#endif

#define CALL_RET_FUNC_FROM_DLL(ret_type,fn_dpref,fn_name,argdecl,argcall)                  \
    typedef ret_type (* ##fn_dpref##fn_name##_t)##argdecl;                                 \
    static fn_dpref##fn_name##_t fn_dpref##fn_name##_ptr=NULL;                             \
    CALL_RET_FUNC_FROM_DLL_CPU(ret_type,fn_dpref,avx512_,fn_name,argdecl,argcall)          \
    CALL_RET_FUNC_FROM_DLL_CPU_MIC(ret_type,fn_dpref,avx512_mic_,fn_name,argdecl,argcall)  \
    CALL_RET_FUNC_FROM_DLL_CPU(ret_type,fn_dpref,avx2_,fn_name,argdecl,argcall)            \
    CALL_RET_FUNC_FROM_DLL_CPU(ret_type,fn_dpref,avx_,fn_name,argdecl,argcall)             \
    CALL_RET_FUNC_FROM_DLL_CPU(ret_type,fn_dpref,sse42_,fn_name,argdecl,argcall)           \
    CALL_RET_FUNC_FROM_DLL_CPU(ret_type,fn_dpref,ssse3_,fn_name,argdecl,argcall)           \
    CALL_RET_FUNC_FROM_DLL_CPU(ret_type,fn_dpref,sse2_,fn_name,argdecl,argcall)

#define CALL_RET_FUNC_FROM_DLL_CPU(ret_type,fn_dpref,fn_cpu,fn_name,argdecl,argcall)                       \
ret_type fn_dpref##fn_cpu##fn_name##argdecl                                                                \
{                                                                                                          \
    load_daal_thr_dll();                                                                                   \
    if(##fn_dpref##fn_name##_ptr == NULL) {                                                                \
        ##fn_dpref##fn_name##_ptr = (##fn_dpref##fn_name##_t)load_daal_thr_func(#fn_dpref#fn_cpu#fn_name); \
    }                                                                                                      \
    return fn_dpref##fn_name##_ptr##argcall;                                                               \
}

#if defined(_WIN64)
    #define CALL_RET_FUNC_FROM_DLL_CPU_MIC(ret_type,fn_dpref,fn_cpu,fn_name,argdecl,argcall)                   \
    ret_type fn_dpref##fn_cpu##fn_name##argdecl                                                                \
    {                                                                                                          \
        load_daal_thr_dll();                                                                                   \
        if(##fn_dpref##fn_name##_ptr == NULL) {                                                                \
            ##fn_dpref##fn_name##_ptr = (##fn_dpref##fn_name##_t)load_daal_thr_func(#fn_dpref#fn_cpu#fn_name); \
        }                                                                                                      \
        return fn_dpref##fn_name##_ptr##argcall;                                                               \
    }
#else
    #define CALL_RET_FUNC_FROM_DLL_CPU_MIC(ret_type,fn_dpref,fn_cpu,fn_name,argdecl,argcall)
#endif

/* Used directly in Intel DAAL */
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, dsyrk,           (const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *beta, double *c, const MKL_INT *ldc),
    (uplo, trans, n, k, alpha, a, lda, beta, c, ldc));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, dsyr,            (const char *uplo, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx,
    double *a, const MKL_INT *lda),
    (uplo, n, alpha, x, incx, a, lda));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, dgemm,           (const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
    const double *alpha, const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb, const double *beta, double *c,const MKL_INT *ldc),
    (transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, xdgemm,          (const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
    const double *alpha, const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb, const double *beta, double *c,const MKL_INT *ldc),
    (transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, dsymm,           (const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const double *alpha,
    const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc),
    (side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, dgemv,           (const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a,
    const MKL_INT *lda, const double *x, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy),
    (trans, m, n, alpha, a, lda, x, incx, beta, y, incy));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dpotrf,        (const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, MKL_INT* info ),
    (uplo, n, a, lda, info ));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dpotrs,        (const char* uplo, const MKL_INT* n, const MKL_INT* nrhs, const double* a, const MKL_INT* lda,
    double* b, const MKL_INT* ldb, MKL_INT* info ), (uplo, n, nrhs, a, lda, b, ldb, info ));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dpotri,        (const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, MKL_INT* info ),
    (uplo, n, a, lda, info ));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, daxpy,           (const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx, double *y,
    const MKL_INT *incy), (n, alpha, x, incx, y, incy));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dgerqf,        (const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, double* tau, double* work,
    const MKL_INT* lwork, MKL_INT* info ), (m, n, a, lda, tau, work, lwork, info ));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dormrq,        (const char* side, const char* trans, const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
    const double* a, const MKL_INT* lda, const double* tau, double* c, const MKL_INT* ldc, double* work, const MKL_INT* lwork,MKL_INT* info,int,int),
    (side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info, 1, 1 ));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dtrtrs,        (const char* uplo, const char* trans, const char* diag, const MKL_INT* n, const MKL_INT* nrhs,
    const double* a, const MKL_INT* lda, double* b, const MKL_INT* ldb, MKL_INT* info ),
    (uplo, trans, diag, n, nrhs, a, lda, b, ldb, info ));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dpptrf,        (const char* uplo, const MKL_INT* n, double* ap, MKL_INT* info ),
    (uplo, n, ap, info ));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dgeqrf,        (const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, double* tau, double* work,
    const MKL_INT* lwork, MKL_INT* info ),(m, n, a, lda, tau, work, lwork, info ));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dgeqp3,        (const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, MKL_INT* jpvt, double* tau,
    double* work, const MKL_INT* lwork, MKL_INT* info ),(m, n, a, lda, jpvt, tau, work, lwork, info ));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dorgqr,        (const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, double* a, const MKL_INT* lda,
    const double* tau, double* work, const MKL_INT* lwork, MKL_INT* info ),(m, n, k, a, lda, tau, work, lwork, info ));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dgesvd,        (const char* jobu, const char* jobvt, const MKL_INT* m, const MKL_INT* n, double* a,
    const MKL_INT* lda, double* s, double* u, const MKL_INT* ldu, double* vt, const MKL_INT* ldvt, double* work, const MKL_INT* lwork,MKL_INT* info),
    (jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info ));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, dsyevd,        (const char* jobz, const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, double* w,
    double* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info ), (jobz, uplo, n, a, lda, w, work, lwork, iwork,liwork,
    info ));
CALL_VOID_FUNC_FROM_DLL(fpk_spblas_, mkl_dcsrmultd, (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, double *a,MKL_INT *ja,
    MKL_INT *ia, double *b, MKL_INT *jb, MKL_INT *ib, double *c, MKL_INT *ldc),(transa, m, n, k, a, ja, ia, b, jb, ib, c, ldc));
CALL_VOID_FUNC_FROM_DLL(fpk_spblas_, mkl_dcsrmv,    (const char *transa, const MKL_INT *m, const MKL_INT *k, const double *alpha,
    const char *matdescra, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, const double *beta,
    double *y), (transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y) );
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, ssyrk,           (const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *beta, float *c, const MKL_INT *ldc), (uplo, trans, n, k, alpha, a, lda, beta, c, ldc));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, ssyr,            (const char *uplo, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx,
    float *a, const MKL_INT *lda), (uplo, n, alpha, x, incx, a, lda));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, sgemm,           (const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
    const float *alpha, const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc),
    (transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, xsgemm,          (const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
    const float *alpha, const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc),
    (transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, ssymm,           (const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const float *alpha,
    const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc), (side, uplo, m, n,
    alpha, a, lda, b, ldb, beta, c, ldc));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, sgemv,           (const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a,
    const MKL_INT *lda, const float *x, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy), (trans, m, n, alpha, a, lda, x, incx,
    beta, y, incy));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, spotrf,        (const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda, MKL_INT* info ),
    (uplo, n, a, lda, info ));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, spotrs,        (const char* uplo, const MKL_INT* n, const MKL_INT* nrhs, const float* a, const MKL_INT* lda,
    float* b, const MKL_INT* ldb, MKL_INT* info ),(uplo, n, nrhs, a, lda, b, ldb, info ));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, spotri,        (const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda, MKL_INT* info ),
    (uplo, n, a, lda, info ));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, saxpy,           (const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx, float *y,
    const MKL_INT *incy),(n, alpha, x, incx, y, incy));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, sgerqf,        (const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda, float* tau, float* work,
    const MKL_INT* lwork, MKL_INT* info ),(m, n, a, lda, tau, work, lwork, info ));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, sormrq,        (const char* side, const char* trans, const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
    const float* a, const MKL_INT* lda, const float* tau, float* c, const MKL_INT* ldc, float* work, const MKL_INT* lwork, MKL_INT* info, int , int),
    (side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info , 1, 1 ));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, strtrs,        (const char* uplo, const char* trans, const char* diag, const MKL_INT* n, const MKL_INT* nrhs,
    const float* a, const MKL_INT* lda, float* b, const MKL_INT* ldb, MKL_INT* info ),(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info ));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, spptrf,        (const char* uplo, const MKL_INT* n, float* ap, MKL_INT* info ),(uplo, n, ap, info ));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, sgeqrf,        (const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda, float* tau, float* work,
    const MKL_INT* lwork, MKL_INT* info ),(m, n, a, lda, tau, work, lwork, info ));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, sgeqp3,        (const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda, MKL_INT* jpvt, float* tau,
    float* work, const MKL_INT* lwork, MKL_INT* info ),(m, n, a, lda, jpvt, tau, work, lwork, info ));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, sorgqr,        (const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, float* a, const MKL_INT* lda,
    const float* tau, float* work, const MKL_INT* lwork, MKL_INT* info ),(m, n, k, a, lda, tau, work, lwork, info ));
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, sgesvd,        (const char* jobu, const char* jobvt, const MKL_INT* m, const MKL_INT* n, float* a,
    const MKL_INT* lda, float* s, float* u, const MKL_INT* ldu, float* vt, const MKL_INT* ldvt, float* work, const MKL_INT* lwork, MKL_INT* info ),
    (jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info )  );
CALL_VOID_FUNC_FROM_DLL(fpk_lapack_, ssyevd,        (const char* jobz, const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda, float* w,
    float* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info ), (jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork,
    info ));
CALL_VOID_FUNC_FROM_DLL(fpk_spblas_, mkl_scsrmultd, (const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, float *a, MKL_INT *ja,
    MKL_INT *ia, float *b, MKL_INT *jb, MKL_INT *ib, float *c, MKL_INT *ldc), (transa, m, n, k, a, ja, ia, b, jb, ib, c, ldc));
CALL_VOID_FUNC_FROM_DLL(fpk_spblas_, mkl_scsrmv,    (const char *transa, const MKL_INT *m, const MKL_INT *k, const float *alpha,
    const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, const float *beta,
    float *y), (transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y) );


CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, GroupsConvolutionCreateForwardBias_F32, (                                                  \
        dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t algorithm, size_t nGroups,size_t dimension,   \
        const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],                                                      \
        const size_t convolutionStrides[], const int inputOffset[],                                                                     \
        const dnnBorder_t border_type), (                                                                                               \
        pConvolution, NULL, algorithm, nGroups, dimension,                                                                              \
        srcSize, dstSize, filterSize,                                                                                                   \
        convolutionStrides, inputOffset,                                                                                                \
        border_type) );
CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, GroupsConvolutionCreateBackwardData_F32, (                                                 \
        dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t algorithm, size_t nGroups, size_t dimension,  \
        const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],                                                      \
        const size_t convolutionStrides[], const int inputOffset[],                                                                     \
        const dnnBorder_t border_type), (                                                                                               \
        pConvolution, NULL, algorithm, nGroups, dimension,                                                                              \
        srcSize, dstSize, filterSize,                                                                                                   \
        convolutionStrides, inputOffset,                                                                                                \
        border_type) );
CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, GroupsConvolutionCreateBackwardFilter_F32, (                                               \
        dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t algorithm, size_t nGroups, size_t dimension,  \
        const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],                                                      \
        const size_t convolutionStrides[], const int inputOffset[],                                                                     \
        const dnnBorder_t border_type), (                                                                                               \
        pConvolution, NULL, algorithm, nGroups, dimension,                                                                              \
        srcSize, dstSize, filterSize,                                                                                                   \
        convolutionStrides, inputOffset,                                                                                                \
        border_type) );
CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, GroupsConvolutionCreateBackwardBias_F32, (                                                 \
        dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t algorithm, size_t nGroups, size_t dimension,  \
        const size_t dstSize[]), (                                                                                                      \
        pConvolution, NULL, algorithm, nGroups, dimension,                                                                              \
        dstSize) );

CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, Execute_F32, (dnnPrimitive_t primitive, void *resources[]), \
    (primitive, resources) );
CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, ConversionExecute_F32, (dnnPrimitive_t conversion, void *from, void *to), \
    (conversion, from, to) );
CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, LayoutCreate_F32,                              \
    (dnnLayout_t *pLayout, size_t dimension, const size_t size[], const size_t strides[]),  \
    (pLayout, dimension, size, strides) );
CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, LayoutCreateFromPrimitive_F32,                 \
    (dnnLayout_t *pLayout, const dnnPrimitive_t primitive, dnnResourceType_t type),         \
    (pLayout, primitive, type) );
CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, AllocateBuffer_F32, (void **pPtr, dnnLayout_t layout), \
    (pPtr, layout) );
CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, ReleaseBuffer_F32, (void *ptr), \
    (ptr) );
CALL_RET_FUNC_FROM_DLL(int, fpk_dnn_, LayoutCompare_F32, (const dnnLayout_t l1, const dnnLayout_t l2), \
    (l1, l2) );
CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, ConversionCreate_F32, (dnnPrimitive_t* pConversion, const dnnLayout_t from, const dnnLayout_t to), \
    (pConversion, from, to) );
CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, Delete_F32, (dnnPrimitive_t primitive), \
    (primitive) );
CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, LayoutDelete_F32, (dnnLayout_t layout), \
    (layout) );

CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, GroupsConvolutionCreateForwardBias_F64, (                                                  \
        dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t algorithm, size_t nGroups,size_t dimension,   \
        const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],                                                      \
        const size_t convolutionStrides[], const int inputOffset[],                                                                     \
        const dnnBorder_t border_type), (                                                                                               \
        pConvolution, NULL, algorithm, nGroups, dimension,                                                                              \
        srcSize, dstSize, filterSize,                                                                                                   \
        convolutionStrides, inputOffset,                                                                                                \
        border_type) );
CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, GroupsConvolutionCreateBackwardData_F64, (                                                 \
        dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t algorithm, size_t nGroups, size_t dimension,  \
        const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],                                                      \
        const size_t convolutionStrides[], const int inputOffset[],                                                                     \
        const dnnBorder_t border_type), (                                                                                               \
        pConvolution, NULL, algorithm, nGroups, dimension,                                                                              \
        srcSize, dstSize, filterSize,                                                                                                   \
        convolutionStrides, inputOffset,                                                                                                \
        border_type) );
CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, GroupsConvolutionCreateBackwardFilter_F64, (                                               \
        dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t algorithm, size_t nGroups, size_t dimension,  \
        const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],                                                      \
        const size_t convolutionStrides[], const int inputOffset[],                                                                     \
        const dnnBorder_t border_type), (                                                                                               \
        pConvolution, NULL, algorithm, nGroups, dimension,                                                                              \
        srcSize, dstSize, filterSize,                                                                                                   \
        convolutionStrides, inputOffset,                                                                                                \
        border_type) );
CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, GroupsConvolutionCreateBackwardBias_F64, (                                                 \
        dnnPrimitive_t* pConvolution, dnnPrimitiveAttributes_t attributes, dnnAlgorithm_t algorithm, size_t nGroups, size_t dimension,  \
        const size_t dstSize[]), (                                                                                                      \
        pConvolution, NULL, algorithm, nGroups, dimension,                                                                              \
        dstSize) );

CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, Execute_F64, (dnnPrimitive_t primitive, void *resources[]), \
    (primitive, resources) );
CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, ConversionExecute_F64, (dnnPrimitive_t conversion, void *from, void *to), \
    (conversion, from, to) );
CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, LayoutCreate_F64,                              \
    (dnnLayout_t *pLayout, size_t dimension, const size_t size[], const size_t strides[]),  \
    (pLayout, dimension, size, strides) );
CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, LayoutCreateFromPrimitive_F64,                 \
    (dnnLayout_t *pLayout, const dnnPrimitive_t primitive, dnnResourceType_t type),         \
    (pLayout, primitive, type) );
CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, AllocateBuffer_F64, (void **pPtr, dnnLayout_t layout), \
    (pPtr, layout) );
CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, ReleaseBuffer_F64, (void *ptr), \
    (ptr) );
CALL_RET_FUNC_FROM_DLL(int, fpk_dnn_, LayoutCompare_F64, (const dnnLayout_t l1, const dnnLayout_t l2), \
    (l1, l2) );
CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, ConversionCreate_F64, (dnnPrimitive_t* pConversion, const dnnLayout_t from, const dnnLayout_t to), \
    (pConversion, from, to) );
CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, Delete_F64, (dnnPrimitive_t primitive), \
    (primitive) );
CALL_RET_FUNC_FROM_DLL(dnnError_t, fpk_dnn_, LayoutDelete_F64, (dnnLayout_t layout), \
    (layout) );


#define CSRMM_ARGS(FPTYPE)                                                                  \
    const char *transa, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,               \
    const FPTYPE *alpha, const char *matdescra, const FPTYPE *val, const MKL_INT *indx,     \
    const MKL_INT *pntrb, const MKL_INT *pntre,                                             \
    const FPTYPE *b, const MKL_INT *ldb, const FPTYPE *beta, FPTYPE *c, const MKL_INT *ldc

CALL_VOID_FUNC_FROM_DLL(fpk_spblas_, mkl_scsrmm, (CSRMM_ARGS(float) ),
    (transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, beta, c, ldc) );
CALL_VOID_FUNC_FROM_DLL(fpk_spblas_, mkl_dcsrmm, (CSRMM_ARGS(double)),
    (transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, beta, c, ldc) );

CALL_VOID_FUNC_FROM_DLL(fpk_blas_, xssyrk,           (const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const float *alpha,
    const float *a, const MKL_INT *lda, const float *beta, float *c, const MKL_INT *ldc), (uplo, trans, n, k, alpha, a, lda, beta, c, ldc));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, xssyr,            (const char *uplo, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx,
    float *a, const MKL_INT *lda), (uplo, n, alpha, x, incx, a, lda));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, xdsyrk,           (const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const double *alpha,
    const double *a, const MKL_INT *lda, const double *beta, double *c, const MKL_INT *ldc),
    (uplo, trans, n, k, alpha, a, lda, beta, c, ldc));
CALL_VOID_FUNC_FROM_DLL(fpk_blas_, xdsyr,            (const char *uplo, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx,
    double *a, const MKL_INT *lda),
    (uplo, n, alpha, x, incx, a, lda));

/* Used in Intel DAAL via SS */
CALL_RET_FUNC_FROM_DLL(IppStatus, fpk_dft_, ippsSortRadixAscend_64f_I, (Ipp64f *pSrcDst, Ipp64f *pTmp, Ipp32s len), (pSrcDst, pTmp, len));
CALL_RET_FUNC_FROM_DLL(IppStatus, fpk_dft_, ippsSortRadixAscend_32f_I, (Ipp32f *pSrcDst, Ipp32f *pTmp, Ipp32s len), (pSrcDst, pTmp, len));
CALL_VOID_FUNC_FROM_DLL(          fpk_blas_, xdsymm,                   (const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n,
    const double *alpha, const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb, const double *beta, double *c,const MKL_INT *ldc),
    (side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc) );
CALL_VOID_FUNC_FROM_DLL(          fpk_blas_, xssymm,                   (const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n,
    const float *alpha, const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc),
    (side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc) );
CALL_VOID_FUNC_FROM_DLL(          fpk_lapack_, dsyev,                  (const char* jobz, const char* uplo, const MKL_INT* n, double* a,
    const MKL_INT* lda, double* w, double* work, const MKL_INT* lwork, MKL_INT* info ), (jobz, uplo, n, a, lda, w, work, lwork, info ));
CALL_VOID_FUNC_FROM_DLL(          fpk_lapack_, ssyev,                  (const char* jobz, const char* uplo, const MKL_INT* n, float* a,
    const MKL_INT* lda, float* w, float* work, const MKL_INT* lwork, MKL_INT* info ),(jobz, uplo, n, a, lda, w, work, lwork, info ));


#define CALL_VOID_FUNC_FROM_DLL_ALONE(fn_dpref,fn_name,argdecl,argcall)   \
    typedef void (* ##fn_dpref##fn_name##_t)##argdecl;                    \
    static fn_dpref##fn_name##_t fn_dpref##fn_name##_ptr=NULL;            \
    void  fn_dpref##fn_name##argdecl                                                                    \
    {                                                                                                   \
        load_daal_thr_dll();                                                                            \
        if(##fn_dpref##fn_name##_ptr == NULL) {                                                         \
            ##fn_dpref##fn_name##_ptr = (##fn_dpref##fn_name##_t)load_daal_thr_func(#fn_dpref#fn_name); \
        }                                                                                               \
        ##fn_dpref##fn_name##_ptr##argcall;                                                             \
    }

#define CALL_RET_FUNC_FROM_DLL_ALONE(ret_type,fn_dpref,fn_name,argdecl,argcall) \
    typedef ret_type (* ##fn_dpref##fn_name##_t)##argdecl;                \
    static fn_dpref##fn_name##_t fn_dpref##fn_name##_ptr=NULL;            \
    ret_type fn_dpref##fn_name##argdecl                                                                 \
    {                                                                                                   \
        load_daal_thr_dll();                                                                            \
        if(##fn_dpref##fn_name##_ptr == NULL) {                                                         \
            ##fn_dpref##fn_name##_ptr = (##fn_dpref##fn_name##_t)load_daal_thr_func(#fn_dpref#fn_name); \
        }                                                                                               \
        return fn_dpref##fn_name##_ptr##argcall;                                                        \
    }

CALL_VOID_FUNC_FROM_DLL_ALONE(   fpk_serv_,set_num_threads,(int nth),(nth));
CALL_RET_FUNC_FROM_DLL_ALONE(int,fpk_serv_,get_max_threads,(void),());
CALL_RET_FUNC_FROM_DLL_ALONE(int,fpk_serv_,set_num_threads_local,(int nth),(nth));
CALL_RET_FUNC_FROM_DLL_ALONE(int,fpk_serv_,get_ncpus,(void),());
CALL_RET_FUNC_FROM_DLL_ALONE(int,fpk_serv_,get_ncorespercpu,(void),());
CALL_RET_FUNC_FROM_DLL_ALONE(int,fpk_serv_,get_ht,(void),());
CALL_RET_FUNC_FROM_DLL_ALONE(int,fpk_serv_,get_nlogicalcores,(void),());
