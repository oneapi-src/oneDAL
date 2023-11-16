/* file: service_spblas_mkl.h */
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

#ifndef __SERVICE_SPBLAS_MKL_H__
#define __SERVICE_SPBLAS_MKL_H__

#include "services/daal_defines.h"
//#include "mkl_daal.h"
#include <mkl.h>

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

// #define __DAAL_MKLFN(f_cpu, f_pref, f_name)              __DAAL_CONCAT4(fpk_, f_pref, f_cpu, f_name)
#define __DAAL_MKLFN(f_cpu, f_pref, f_name) f_name
#define __DAAL_MKLFN_CALL(f_pref, f_name, f_args)        __DAAL_MKLFN_CALL1(f_pref, f_name, f_args)
#define __DAAL_MKLFN_CALL_RETURN(f_pref, f_name, f_args) __DAAL_MKLFN_CALL2(f_pref, f_name, f_args)

#define __DAAL_MKLFN_CALL1(f_pref, f_name, f_args)             \
    if (avx512 == cpu)                                         \
    {                                                          \
        __DAAL_MKLFN(avx512_, f_pref, f_name) f_args;          \
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
        __DAAL_MKLFN(__DAAL_MKL_SSE2, f_pref, f_name) f_args;  \
    }

#define __DAAL_MKLFN_CALL2(f_pref, f_name, f_args)                    \
    if (avx512 == cpu)                                                \
    {                                                                 \
        return __DAAL_MKLFN(avx512_, f_pref, f_name) f_args;          \
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
        return __DAAL_MKLFN(__DAAL_MKL_SSE2, f_pref, f_name) f_args;  \
    }

namespace daal
{
namespace internal
{
namespace mkl
{
template <typename fpType, CpuType cpu>
struct MklSpBlas
{};

/*
// Double precision functions definition
*/

template <CpuType cpu>
struct MklSpBlas<double, cpu>
{
    typedef DAAL_INT SizeType;

    static void xcsrmultd(const char * transa, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, double * a, DAAL_INT * ja, DAAL_INT * ia,
                          double * b, DAAL_INT * jb, DAAL_INT * ib, double * c, DAAL_INT * ldc)
    {
        __DAAL_MKLFN_CALL(spblas_, mkl_dcsrmultd, (transa, (const MKL_INT*)m, (const MKL_INT*)n, (const MKL_INT*)k, a, (MKL_INT*)ja, (MKL_INT*)ia, b, (MKL_INT*)jb, (MKL_INT*)ib, c, (MKL_INT*)ldc));
    }

    static void xcsrmv(const char * transa, const DAAL_INT * m, const DAAL_INT * k, const double * alpha, const char * matdescra, const double * val,
                       const DAAL_INT * indx, const DAAL_INT * pntrb, const DAAL_INT * pntre, const double * x, const double * beta, double * y)
    {
        __DAAL_MKLFN_CALL(spblas_, mkl_dcsrmv, (transa, (const MKL_INT*)m, (const MKL_INT*)k, alpha, matdescra, val, (const MKL_INT*)indx, (const MKL_INT*)pntrb, (const MKL_INT*)pntre, x, beta, y));
    }

    static void xcsrmm(const char * transa, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, const double * alpha, const char * matdescra,
                       const double * val, const DAAL_INT * indx, const DAAL_INT * pntrb, const double * b, const DAAL_INT * ldb, const double * beta,
                       double * c, const DAAL_INT * ldc)
    {
        __DAAL_MKLFN_CALL(spblas_, mkl_dcsrmm, (transa, (const MKL_INT*)m, (const MKL_INT*)n, (const MKL_INT*)k, alpha, matdescra, val, (const MKL_INT*)indx, (const MKL_INT*)pntrb, (const MKL_INT*)(pntrb + 1), b, (const MKL_INT*)ldb, beta, c, (const MKL_INT*)ldc));
    }

    static void xxcsrmm(const char * transa, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, const double * alpha, const char * matdescra,
                        const double * val, const DAAL_INT * indx, const DAAL_INT * pntrb, const double * b, const DAAL_INT * ldb,
                        const double * beta, double * c, const DAAL_INT * ldc)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(spblas_, mkl_dcsrmm, (transa, (const MKL_INT*)m, (const MKL_INT*)n, (const MKL_INT*)k, alpha, matdescra, val, (const MKL_INT*)indx, (const MKL_INT*)pntrb, (const MKL_INT*)(pntrb + 1), b, (const MKL_INT*)ldb, beta, c, (const MKL_INT*)ldc));
        fpk_serv_set_num_threads_local(old_threads);
    }
};

/*
// Single precision functions definition
*/

template <CpuType cpu>
struct MklSpBlas<float, cpu>
{
    typedef DAAL_INT SizeType;

    static void xcsrmultd(const char * transa, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, float * a, DAAL_INT * ja, DAAL_INT * ia,
                          float * b, DAAL_INT * jb, DAAL_INT * ib, float * c, DAAL_INT * ldc)
    {
        __DAAL_MKLFN_CALL(spblas_, mkl_scsrmultd, (transa, (const MKL_INT*)m, (const MKL_INT*)n, (const MKL_INT*)k, a, (MKL_INT*)ja, (MKL_INT*)ia, b, (MKL_INT*)jb, (MKL_INT*)ib, c, (MKL_INT*)ldc));
    }

    static void xcsrmv(const char * transa, const DAAL_INT * m, const DAAL_INT * k, const float * alpha, const char * matdescra, const float * val,
                       const DAAL_INT * indx, const DAAL_INT * pntrb, const DAAL_INT * pntre, const float * x, const float * beta, float * y)
    {
        __DAAL_MKLFN_CALL(spblas_, mkl_scsrmv, (transa, (const MKL_INT*)m, (const MKL_INT*)k, alpha, matdescra, val, (const MKL_INT*)indx, (const MKL_INT*)pntrb, (const MKL_INT*)pntre, x, beta, y));
    }

    static void xcsrmm(const char * transa, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, const float * alpha, const char * matdescra,
                       const float * val, const DAAL_INT * indx, const DAAL_INT * pntrb, const float * b, const DAAL_INT * ldb, const float * beta,
                       float * c, const DAAL_INT * ldc)
    {
        __DAAL_MKLFN_CALL(spblas_, mkl_scsrmm, (transa, (const MKL_INT*)m, (const MKL_INT*)n, (const MKL_INT*)k, alpha, matdescra, val, (const MKL_INT*)indx, (const MKL_INT*)pntrb, (const MKL_INT*)(pntrb + 1), b, (const MKL_INT*)ldb, beta, c, (const MKL_INT*)ldc));
    }

    static void xxcsrmm(const char * transa, const DAAL_INT * m, const DAAL_INT * n, const DAAL_INT * k, const float * alpha, const char * matdescra,
                        const float * val, const DAAL_INT * indx, const DAAL_INT * pntrb, const float * b, const DAAL_INT * ldb, const float * beta,
                        float * c, const DAAL_INT * ldc)
    {
        int old_threads = fpk_serv_set_num_threads_local(1);
        __DAAL_MKLFN_CALL(spblas_, mkl_scsrmm, (transa, (const MKL_INT*)m, (const MKL_INT*)n, (const MKL_INT*)k, alpha, matdescra, val, (const MKL_INT*)indx, (const MKL_INT*)pntrb, (const MKL_INT*)(pntrb + 1), b, (const MKL_INT*)ldb, beta, c, (const MKL_INT*)ldc));
        fpk_serv_set_num_threads_local(old_threads);
    }
};

} // namespace mkl
} // namespace internal
} // namespace daal

#endif
