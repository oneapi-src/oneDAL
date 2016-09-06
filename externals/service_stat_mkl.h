/* file: service_stat_mkl.h */
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


#ifndef __SERVICE_STAT_MKL_H__
#define __SERVICE_STAT_MKL_H__

#include "daal_defines.h"
#include "mkl_daal.h"
#include "vmlvsl.h"
#include "threading.h"

#if !defined(__DAAL_CONCAT4)
    #define __DAAL_CONCAT4(a,b,c,d) __DAAL_CONCAT41(a,b,c,d)
    #define __DAAL_CONCAT41(a,b,c,d) a##b##c##d
#endif

#if !defined(__DAAL_CONCAT5)
    #define __DAAL_CONCAT5(a,b,c,d,e) __DAAL_CONCAT51(a,b,c,d,e)
    #define __DAAL_CONCAT51(a,b,c,d,e) a##b##c##d##e
#endif

#define __DAAL_VSLFN(f_cpu,f_pref,f_name)        __DAAL_CONCAT5(f_pref,_,f_cpu,_,f_name)
#define __DAAL_VSLFN_CALL(f_pref,f_name,f_args,errcode)  __DAAL_VSLFN_CALL1(f_pref,f_name,f_args,errcode)
#define __DAAL_VSLFN_CALL_NR(f_pref,f_name,f_args,errcode)  __DAAL_VSLFN_CALL2(f_pref,f_name,f_args,errcode)

#if defined(_WIN64) || defined(__x86_64__)

#if defined(__APPLE__)
    #define __DAAL_MKLVSL_SSE2   u8
#else
    #define __DAAL_MKLVSL_SSE2   ex
#endif

#define __DAAL_VSLFN_CALL1(f_pref,f_name,f_args,errcode)                  \
    if(avx512 == cpu)                                                     \
    {                                                                     \
        errcode = __DAAL_VSLFN(z0,f_pref,f_name) f_args;                  \
    }                                                                     \
    if(avx512_mic == cpu)                                                 \
    {                                                                     \
        errcode = __DAAL_VSLFN(b3,f_pref,f_name) f_args;                  \
    }                                                                     \
    if(avx2 == cpu)                                                       \
    {                                                                     \
        errcode = __DAAL_VSLFN(l9,f_pref,f_name) f_args;                  \
    }                                                                     \
    if(avx == cpu)                                                        \
    {                                                                     \
        errcode = __DAAL_VSLFN(e9,f_pref,f_name) f_args;                  \
    }                                                                     \
    if(sse42 == cpu)                                                      \
    {                                                                     \
        errcode = __DAAL_VSLFN(h8,f_pref,f_name) f_args;                  \
    }                                                                     \
    if(ssse3 == cpu)                                                      \
    {                                                                     \
        errcode = __DAAL_VSLFN(u8,f_pref,f_name) f_args;                  \
    }                                                                     \
    if(sse2 == cpu)                                                       \
    {                                                                     \
        errcode = __DAAL_VSLFN(__DAAL_MKLVSL_SSE2,f_pref,f_name) f_args;  \
    }                                                                     \
    if (errcode != 0) { return errcode; }
#define __DAAL_VSLFN_CALL2(f_pref,f_name,f_args,retcode)                  \
    if(avx512 == cpu)                                                     \
    {                                                                     \
        retcode = __DAAL_VSLFN(z0,f_pref,f_name) f_args;                  \
    }                                                                     \
    if(avx512_mic == cpu)                                                 \
    {                                                                     \
        retcode = __DAAL_VSLFN(b3,f_pref,f_name) f_args;                  \
    }                                                                     \
    if(avx2 == cpu)                                                       \
    {                                                                     \
        retcode = __DAAL_VSLFN(l9,f_pref,f_name) f_args;                  \
    }                                                                     \
    if(avx == cpu)                                                        \
    {                                                                     \
        retcode = __DAAL_VSLFN(e9,f_pref,f_name) f_args;                  \
    }                                                                     \
    if(sse42 == cpu)                                                      \
    {                                                                     \
        retcode = __DAAL_VSLFN(h8,f_pref,f_name) f_args;                  \
    }                                                                     \
    if(ssse3 == cpu)                                                      \
    {                                                                     \
        retcode = __DAAL_VSLFN(u8,f_pref,f_name) f_args;                  \
    }                                                                     \
    if(sse2 == cpu)                                                       \
    {                                                                     \
        retcode = __DAAL_VSLFN(__DAAL_MKLVSL_SSE2,f_pref,f_name) f_args;  \
    }

#else

#if defined(__APPLE__)
    #define __DAAL_MKLVSL_SSE2   v8
#else
    #define __DAAL_MKLVSL_SSE2   w7
#endif

#define __DAAL_VSLFN_CALL1(f_pref,f_name,f_args,errcode)                  \
    if(avx512 == cpu)                                                     \
    {                                                                     \
        errcode = __DAAL_VSLFN(x0,f_pref,f_name) f_args;                  \
    }                                                                     \
    if(avx512_mic == cpu)                                                 \
    {                                                                     \
        errcode = __DAAL_VSLFN(a3,f_pref,f_name) f_args;                  \
    }                                                                     \
    if(avx2 == cpu)                                                       \
    {                                                                     \
        errcode = __DAAL_VSLFN(s9,f_pref,f_name) f_args;                  \
    }                                                                     \
    if(avx == cpu)                                                        \
    {                                                                     \
        errcode = __DAAL_VSLFN(g9,f_pref,f_name) f_args;                  \
    }                                                                     \
    if(sse42 == cpu)                                                      \
    {                                                                     \
        errcode = __DAAL_VSLFN(n8,f_pref,f_name) f_args;                  \
    }                                                                     \
    if(ssse3 == cpu)                                                      \
    {                                                                     \
        errcode = __DAAL_VSLFN(v8,f_pref,f_name) f_args;                  \
    }                                                                     \
    if(sse2 == cpu)                                                       \
    {                                                                     \
        errcode = __DAAL_VSLFN(__DAAL_MKLVSL_SSE2,f_pref,f_name) f_args;  \
    }                                                                     \
    if (errcode != 0) { return errcode; }
#define __DAAL_VSLFN_CALL2(f_pref,f_name,f_args,retcode)                  \
    if(avx512 == cpu)                                                     \
    {                                                                     \
        retcode = __DAAL_VSLFN(x0,f_pref,f_name) f_args;                  \
    }                                                                     \
    if(avx512_mic == cpu)                                                 \
    {                                                                     \
        retcode = __DAAL_VSLFN(a3,f_pref,f_name) f_args;                  \
    }                                                                     \
    if(avx2 == cpu)                                                       \
    {                                                                     \
        retcode = __DAAL_VSLFN(s9,f_pref,f_name) f_args;                  \
    }                                                                     \
    if(avx == cpu)                                                        \
    {                                                                     \
        retcode = __DAAL_VSLFN(g9,f_pref,f_name) f_args;                  \
    }                                                                     \
    if(sse42 == cpu)                                                      \
    {                                                                     \
        retcode = __DAAL_VSLFN(n8,f_pref,f_name) f_args;                  \
    }                                                                     \
    if(ssse3 == cpu)                                                      \
    {                                                                     \
        retcode = __DAAL_VSLFN(v8,f_pref,f_name) f_args;                  \
    }                                                                     \
    if(sse2 == cpu)                                                       \
    {                                                                     \
        retcode = __DAAL_VSLFN(__DAAL_MKLVSL_SSE2,f_pref,f_name) f_args;  \
    }

#endif


#if defined(_WIN64) || defined(__x86_64__)
    #define __SS_ILP_FLAG__ 1
#else
    #define __SS_ILP_FLAG__ 0
#endif

#ifndef __DAAL__MKL_VSL_SS_
#define __DAAL__MKL_VSL_SS_

extern "C"
{
#define __DAAL_VSL_SS_MATRIX_STORAGE_COLS              0x00020000
#define __DAAL_VSL_SS_MATRIX_STORAGE_FULL              0x00000000
#define __DAAL_VSL_SS_ED_WEIGHTS                       6
#define __DAAL_VSL_SS_ED_MIN                           16
#define __DAAL_VSL_SS_ED_MAX                           17
#define __DAAL_VSL_SS_ED_SUM                           67
#define __DAAL_VSL_SS_ED_MEAN                          7
#define __DAAL_VSL_SS_ED_2R_MOM                        8
#define __DAAL_VSL_SS_ED_2C_MOM                        11
#define __DAAL_VSL_SS_ED_VARIATION                     18
#define __DAAL_VSL_SS_ED_CP                            74
#define __DAAL_VSL_SS_ED_CP_STORAGE                    75
#define __DAAL_VSL_SS_CP                               0x0000000100000000
#define __DAAL_VSL_SS_METHOD_FAST                      0x00000001
#define __DAAL_VSL_SS_METHOD_1PASS                     0x00000002
#define __DAAL_VSL_SS_METHOD_FAST_USER_MEAN            0x00000100
#define __DAAL_VSL_SS_MIN                              0x0000000000000400
#define __DAAL_VSL_SS_MAX                              0x0000000000000800
#define __DAAL_VSL_SS_SUM                              0x0000000002000000
#define __DAAL_VSL_SS_MEAN                             0x0000000000000001
#define __DAAL_VSL_SS_2R_MOM                           0x0000000000000002
#define __DAAL_VSL_SS_2C_MOM                           0x0000000000000010
#define __DAAL_VSL_SS_VARIATION                        0x0000000000000200
#define __DAAL_VSL_SS_ED_ACCUM_WEIGHT                  23
#define __DAAL_VSL_SS_METHOD_FAST_USER_MEAN            0x00000100
#define __DAAL_VSL_SS_METHOD_BACON_MEDIAN_INIT         0x00000002
#define __DAAL_VSL_SS_METHOD_BACON_MAHALANOBIS_INIT    0x00000001
#define __DAAL_VSL_SS_OUTLIERS                         0x0000000000080000
#define __DAAL_VSL_SS_METHOD_BACON                     0x00000020
#define __DAAL_VSL_SS_QUANTS                           0x0000000000010000
#define __DAAL_VSL_SS_ED_QUANT_ORDER_N                 24
#define __DAAL_VSL_SS_ED_QUANT_ORDER                   25
#define __DAAL_VSL_SS_ED_QUANT_QUANTILES               26
#define __DAAL_VSL_SS_SORTED_OBSERV                    0x0000008000000000
#define __DAAL_VSL_SS_ED_SORTED_OBSERV                 78
#define __DAAL_VSL_SS_ED_SORTED_OBSERV_STORAGE         79
#define __DAAL_VSL_SS_METHOD_RADIX                     0x00100000

    // MKL errors
#define __DAAL_VSL_SS_ERROR_BAD_QUANT_ORDER                   -4022
#define __DAAL_VSL_SS_ERROR_INDICES_NOT_SUPPORTED             -4085



    typedef void *tDAAL_VSLSSTaskPtr;

    typedef void          (*func_type)(MKL_INT , MKL_INT , MKL_INT , void *);
    typedef void          (*threadfuncfor)(MKL_INT , MKL_INT , void *, func_type );
    typedef void          (*threadfuncforordered)(MKL_INT , MKL_INT , void *, func_type );
    typedef void          (*threadfuncsection)(MKL_INT , void *, func_type );
    typedef void          (*threadfuncordered)(MKL_INT , MKL_INT , MKL_INT, void *, func_type );
    typedef MKL_INT       (*threadgetlimit)(void);

    struct ThreadingFuncs
    {
        threadfuncfor     funcfor;
        threadfuncfor     funcforordered;
        threadfuncsection funcsection;
        threadfuncordered funcordered;
        threadgetlimit    getlimit;
    };

    static void _daal_mkl_threader_for_sequential(MKL_INT n, MKL_INT threads_request, void *a, func_type func)
    {
        MKL_INT i;

        for (i = 0; i < n; i++)
        {
            func(i, 0, 1, a);
        }
    }

    static void _daal_mkl_threader_for_ordered_sequential(MKL_INT n, MKL_INT threads_request, void *a, func_type func)
    {
        MKL_INT i;

        for (i = 0; i < n; i++)
        {
            func(i, 0, 1, a);
        }
    }

    static void _daal_mkl_threader_sections_sequential(MKL_INT threads_request, void *a, func_type func)
    {
        func(0, 0, 1, a);
    }

    static void _daal_mkl_threader_ordered_sequential(MKL_INT i, MKL_INT th_idx, MKL_INT th_num, void *a, func_type func)
    {
        func(i, th_idx, th_num, a);
    }

    static MKL_INT _daal_mkl_threader_get_max_threads_sequential()
    {
        return 1;
    }

    static void _daal_mkl_threader_for(MKL_INT n, MKL_INT threads_request, void *a, func_type func)
    {
        MKL_INT quo, mod;
        MKL_INT threadsNumber;

        threadsNumber = (n < threads_request) ? n : threads_request;

        quo = n / threads_request;
        mod = n % threads_request;

        if(mod == 0)
        {
            daal::threader_for(threadsNumber, threadsNumber, [=](int j)
            {
                for (int i = 0; i < quo; i++)
                {
                    func(j * quo + i, j, threads_request, a);
                }
            });
        }
        else
        {
            daal::threader_for(threadsNumber, threadsNumber, [=](int j)
            {
                for (int i = 0; i < quo; i++)
                {
                    func(j * quo + i, j, threadsNumber, a);
                }
            });
            for (int i = 0; i < mod; i++)
            {
                func(threadsNumber * quo + i, threadsNumber - 1, threadsNumber, a);
            }
        }
    }

    static void _daal_mkl_threader_for_ordered(MKL_INT n, MKL_INT threads_request, void *a, func_type func)
    {
        //not used. To be implemented if needed.
    }

    static void _daal_mkl_threader_sections(MKL_INT threads_request, void *a, func_type func)
    {
        daal::threader_for(threads_request, threads_request, [=](int i)
        {
            func(i, i, threads_request, a);
        });
    }

    static void _daal_mkl_threader_ordered(MKL_INT i, MKL_INT th_idx, MKL_INT th_num, void *a, func_type func)
    {
        //not used. To be implemented if needed.
    }

    static MKL_INT _daal_mkl_threader_get_max_threads()
    {
        return daal::threader_get_threads_number();
    }
}

#endif

namespace daal
{
namespace internal
{
namespace mkl
{

template<typename fpType, CpuType cpu>
struct MklStatistics {};

/*
// Double precision functions definition
*/

template<CpuType cpu>
struct MklStatistics<double, cpu>
{
    typedef __int64 SizeType;
    typedef __int64 MethodType;
    typedef int     ErrorType;

    static int xcp(double *data, __int64 nFeatures, __int64 nVectors, double *nPreviousObservations, double *sum,
            double *crossProduct, __int64 method)
    {
        tDAAL_VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;
        __int64 cpStorage   = __DAAL_VSL_SS_MATRIX_STORAGE_FULL;

        double *mean = (double *)daal::services::daal_malloc(nFeatures * sizeof(double));

        if (method == __DAAL_VSL_SS_METHOD_FAST_USER_MEAN)
        {
            double invNVectors = 1.0 / (double)nVectors;
            for (size_t i = 0; i < nFeatures; i++)
            {
                mean[i] = sum[i] * invNVectors;
            }
        }

        double weight[2] = { *nPreviousObservations, *nPreviousObservations };

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSNewTask, (&task, &nFeatures, &nVectors, &dataStorage, data,
                                                              0, 0, __SS_ILP_FLAG__), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_SUM, sum), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_MEAN, mean), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_CP, crossProduct), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsliSSEditTask, (task, __DAAL_VSL_SS_ED_CP_STORAGE, &cpStorage), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_ACCUM_WEIGHT, weight), errcode);

        ThreadingFuncs threading =
        {
            _daal_mkl_threader_for,
            _daal_mkl_threader_for_ordered,
            _daal_mkl_threader_sections,
            _daal_mkl_threader_ordered,
            _daal_mkl_threader_get_max_threads
        };

        __DAAL_VSLFN_CALL(fpk_vsl_kernel, dSSBasic, (task, __DAAL_VSL_SS_CP | __DAAL_VSL_SS_SUM,
                                                     method, &threading), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode);

        daal::services::daal_free(mean);
        return errcode;
    }


    static int xxcp_weight(double *data, __int64 nFeatures, __int64 nVectors, double *weight, double *accumWeight, double *mean,
                    double *crossProduct, __int64 method)
    {
        tDAAL_VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;
        __int64 cpStorage   = __DAAL_VSL_SS_MATRIX_STORAGE_FULL;

        double *sum = (double *)daal::services::daal_malloc(nFeatures * sizeof(double));

        double accumWeightsAll[2] = {0, 0};

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSNewTask, (&task, &nFeatures, &nVectors, &dataStorage, data,
                          0, 0, __SS_ILP_FLAG__), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_SUM, sum), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_MEAN, mean), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_WEIGHTS, weight), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_CP, crossProduct), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsliSSEditTask, (task, __DAAL_VSL_SS_ED_CP_STORAGE, &cpStorage), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_ACCUM_WEIGHT, accumWeightsAll), errcode);

        ThreadingFuncs threading =
        {
            _daal_mkl_threader_for_sequential,
            _daal_mkl_threader_for_ordered_sequential,
            _daal_mkl_threader_sections_sequential,
            _daal_mkl_threader_ordered_sequential,
            _daal_mkl_threader_get_max_threads_sequential
        };

        __DAAL_VSLFN_CALL(fpk_vsl_kernel, dSSBasic, (task, __DAAL_VSL_SS_CP | __DAAL_VSL_SS_MEAN , method, &threading), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode);

        *accumWeight = accumWeightsAll[0];

        daal::services::daal_free(sum);
        return errcode;
    }

    static int x2c_mom(double *data, __int64 nFeatures, __int64 nVectors, double *variance, __int64 method)
    {
        tDAAL_VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        double *mean                 = (double *)daal::services::daal_malloc(nFeatures * sizeof(double));
        double *secondOrderRawMoment = (double *)daal::services::daal_malloc(nFeatures * sizeof(double));

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSNewTask, (&task, &nFeatures, &nVectors, &dataStorage, data, 0, 0, __SS_ILP_FLAG__), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_MEAN, mean), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_2R_MOM,secondOrderRawMoment), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_2C_MOM, variance), errcode);

        ThreadingFuncs threading =
        {
            _daal_mkl_threader_for,
            _daal_mkl_threader_for_ordered,
            _daal_mkl_threader_sections,
            _daal_mkl_threader_ordered,
            _daal_mkl_threader_get_max_threads
        };

        __DAAL_VSLFN_CALL(fpk_vsl_kernel, dSSBasic, (task, __DAAL_VSL_SS_2C_MOM, method, &threading), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode);

        daal::services::daal_free(mean);
        daal::services::daal_free(secondOrderRawMoment);
        return errcode;
    }

    static int xoutlierdetection(double *data, __int64 nFeatures, __int64 nVectors,  __int64 nParams,
                          double *baconParams, double *baconWeights)
    {
        tDAAL_VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSNewTask, (&task, &nFeatures, &nVectors, &dataStorage, data, 0, 0,
                                                              __SS_ILP_FLAG__), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditOutDetect, (task, &nParams, baconParams, baconWeights),
                          errcode);

        ThreadingFuncs threading =
        {
            _daal_mkl_threader_for,
            _daal_mkl_threader_for_ordered,
            _daal_mkl_threader_sections,
            _daal_mkl_threader_ordered,
            _daal_mkl_threader_get_max_threads
        };

        __DAAL_VSLFN_CALL(fpk_vsl_kernel, dSSOutliersDetection, (task, __DAAL_VSL_SS_OUTLIERS,
                                                                 __DAAL_VSL_SS_METHOD_BACON, &threading), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode);
        return errcode;
    }

    static int xLowOrderMoments(double *data, __int64 nFeatures, __int64 nVectors, __int64 method,
                         double *sum, double *mean, double *secondOrderRawMoment,
                         double *variance, double *variation)
    {
        tDAAL_VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSNewTask, (&task, &nFeatures, &nVectors, &dataStorage, data,
                                                              0, 0, __SS_ILP_FLAG__), errcode);

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_SUM, sum), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_MEAN, mean), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_2R_MOM,
                                                               secondOrderRawMoment), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_2C_MOM, variance), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_VARIATION, variation), errcode);

        ThreadingFuncs threading =
        {
            _daal_mkl_threader_for,
            _daal_mkl_threader_for_ordered,
            _daal_mkl_threader_sections,
            _daal_mkl_threader_ordered,
            _daal_mkl_threader_get_max_threads
        };

        __DAAL_VSLFN_CALL(fpk_vsl_kernel, dSSBasic, (task,
                                                     __DAAL_VSL_SS_SUM | __DAAL_VSL_SS_MEAN | __DAAL_VSL_SS_2R_MOM |
                                                     __DAAL_VSL_SS_2C_MOM | __DAAL_VSL_SS_VARIATION,
                                                     method, &threading), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode);
        return errcode;
    }

    static int xSumAndVariance(double *data, __int64 nFeatures, __int64 nVectors, double *nPreviousObservations,
                        __int64 method, double *sum, double *mean, double *secondOrderRawMoment,
                        double *variance)
    {
        tDAAL_VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSNewTask, (&task, &nFeatures, &nVectors, &dataStorage, data,
                                                              0, 0, __SS_ILP_FLAG__), errcode);

        double weight[2] = { *nPreviousObservations, *nPreviousObservations };

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_SUM, sum), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_MEAN, mean), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_2R_MOM,
                                                               secondOrderRawMoment), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_2C_MOM, variance), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_ACCUM_WEIGHT, weight), errcode);

        ThreadingFuncs threading =
        {
            _daal_mkl_threader_for,
            _daal_mkl_threader_for_ordered,
            _daal_mkl_threader_sections,
            _daal_mkl_threader_ordered,
            _daal_mkl_threader_get_max_threads
        };

        __DAAL_VSLFN_CALL(fpk_vsl_kernel, dSSBasic, (task,
                                                     __DAAL_VSL_SS_SUM | __DAAL_VSL_SS_MEAN | __DAAL_VSL_SS_2R_MOM |
                                                     __DAAL_VSL_SS_2C_MOM, method, &threading), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode);
        return errcode;
    }

    static int xQuantiles(double *data, __int64 nFeatures, __int64 nVectors, __int64 quantOrderN, double* quantOrder, double *quants)
    {
        tDAAL_VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSNewTask, (&task, &nFeatures, &nVectors, &dataStorage, data, 0, 0, __SS_ILP_FLAG__), errcode);
        if(errcode) {return errcode;}

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsliSSEditTask, (task, __DAAL_VSL_SS_ED_QUANT_ORDER_N, &quantOrderN), errcode);
        if(errcode) { __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode); return errcode; }

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_QUANT_ORDER, quantOrder), errcode);
        if(errcode) { __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode); return errcode; }

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_QUANT_QUANTILES, quants), errcode);
        if(errcode) { __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode); return errcode; }

        ThreadingFuncs threading =
        {
            _daal_mkl_threader_for,
            _daal_mkl_threader_for_ordered,
            _daal_mkl_threader_sections,
            _daal_mkl_threader_ordered,
            _daal_mkl_threader_get_max_threads
        };

        __DAAL_VSLFN_CALL(fpk_vsl_kernel, dSSQuantiles, (task, __DAAL_VSL_SS_QUANTS, __DAAL_VSL_SS_METHOD_FAST, &threading), errcode);
        if(errcode) { __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode); return errcode; }

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode);
        return errcode;
    }

    static int xSort(double *data, __int64 nFeatures, __int64 nVectors, double *sortedData)
    {
        tDAAL_VSLSSTaskPtr task;
        int errcode = 0;

        __int64 inputStorage  = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;
        __int64 outputStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSNewTask, (&task, &nFeatures, &nVectors, &inputStorage, data, 0, 0, __SS_ILP_FLAG__), errcode);
        if(errcode) {return errcode;}

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsldSSEditTask, (task, __DAAL_VSL_SS_ED_SORTED_OBSERV, sortedData), errcode);
        if(errcode) { __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode); return errcode; }

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsliSSEditTask, (task, __DAAL_VSL_SS_ED_SORTED_OBSERV_STORAGE, &outputStorage), errcode);
        if(errcode) { __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode); return errcode; }

        ThreadingFuncs threading =
        {
            _daal_mkl_threader_for,
            _daal_mkl_threader_for_ordered,
            _daal_mkl_threader_sections,
            _daal_mkl_threader_ordered,
            _daal_mkl_threader_get_max_threads
        };

        __DAAL_VSLFN_CALL(fpk_vsl_kernel, dSSSort, (task, __DAAL_VSL_SS_SORTED_OBSERV, __DAAL_VSL_SS_METHOD_RADIX, &threading), errcode);
        if(errcode) { __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode); return errcode; }

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode);
        return errcode;
    }
};

/*
// Single precision functions definition
*/

template<CpuType cpu>
struct MklStatistics<float, cpu>
{
    typedef __int64 SizeType;
    typedef __int64 MethodType;
    typedef int     ErrorType;

    static int xcp(float *data, __int64 nFeatures, __int64 nVectors, float *nPreviousObservations, float *sum,
            float *crossProduct, __int64 method)
    {
        tDAAL_VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;
        __int64 cpStorage   = __DAAL_VSL_SS_MATRIX_STORAGE_FULL;

        float *mean = (float *)daal::services::daal_malloc(nFeatures * sizeof(float));

        if (method == __DAAL_VSL_SS_METHOD_FAST_USER_MEAN)
        {
            float invNVectors = 1.0 / (float)nVectors;
            for (size_t i = 0; i < nFeatures; i++)
            {
                mean[i] = sum[i] * invNVectors;
            }
        }

        float weight[2] = { *nPreviousObservations, *nPreviousObservations };

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSNewTask, (&task, &nFeatures, &nVectors, &dataStorage, data,
                                                              0, 0, __SS_ILP_FLAG__), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_SUM, sum), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_MEAN, mean), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_CP, crossProduct), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsliSSEditTask, (task, __DAAL_VSL_SS_ED_CP_STORAGE, &cpStorage), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_ACCUM_WEIGHT, weight), errcode);

        ThreadingFuncs threading =
        {
            _daal_mkl_threader_for,
            _daal_mkl_threader_for_ordered,
            _daal_mkl_threader_sections,
            _daal_mkl_threader_ordered,
            _daal_mkl_threader_get_max_threads
        };

        __DAAL_VSLFN_CALL(fpk_vsl_kernel, sSSBasic, (task, __DAAL_VSL_SS_CP | __DAAL_VSL_SS_SUM,
                                                     method, &threading), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode);

        daal::services::daal_free(mean);
        return errcode;
    }


    static int xxcp_weight(float *data, __int64 nFeatures, __int64 nVectors, float *weight, float *accumWeight, float *mean,
                    float *crossProduct, __int64 method)
    {
        tDAAL_VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;
        __int64 cpStorage   = __DAAL_VSL_SS_MATRIX_STORAGE_FULL;

        float *sum = (float *)daal::services::daal_malloc(nFeatures * sizeof(float));

        float accumWeightsAll[2] = {0, 0};

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSNewTask, (&task, &nFeatures, &nVectors, &dataStorage, data,
                          0, 0, __SS_ILP_FLAG__), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_SUM, sum), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_MEAN, mean), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_WEIGHTS, weight), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_CP, crossProduct), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsliSSEditTask, (task, __DAAL_VSL_SS_ED_CP_STORAGE, &cpStorage), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_ACCUM_WEIGHT, accumWeightsAll), errcode);

        ThreadingFuncs threading =
        {
            _daal_mkl_threader_for_sequential,
            _daal_mkl_threader_for_ordered_sequential,
            _daal_mkl_threader_sections_sequential,
            _daal_mkl_threader_ordered_sequential,
            _daal_mkl_threader_get_max_threads_sequential
        };

        __DAAL_VSLFN_CALL(fpk_vsl_kernel, sSSBasic, (task, __DAAL_VSL_SS_CP | __DAAL_VSL_SS_MEAN , method, &threading), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode);

        *accumWeight = accumWeightsAll[0];

        daal::services::daal_free(sum);
        return errcode;
    }

    static int x2c_mom(float *data, __int64 nFeatures, __int64 nVectors, float *variance, __int64 method)
    {
        tDAAL_VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        float *mean                 = (float *)daal::services::daal_malloc(nFeatures * sizeof(float));
        float *secondOrderRawMoment = (float *)daal::services::daal_malloc(nFeatures * sizeof(float));

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSNewTask, (&task, &nFeatures, &nVectors, &dataStorage, data, 0, 0, __SS_ILP_FLAG__), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_MEAN, mean), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_2R_MOM,secondOrderRawMoment), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_2C_MOM, variance), errcode);

        ThreadingFuncs threading =
        {
            _daal_mkl_threader_for,
            _daal_mkl_threader_for_ordered,
            _daal_mkl_threader_sections,
            _daal_mkl_threader_ordered,
            _daal_mkl_threader_get_max_threads
        };

        __DAAL_VSLFN_CALL(fpk_vsl_kernel, sSSBasic, (task, __DAAL_VSL_SS_2C_MOM, method, &threading), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode);

        daal::services::daal_free(mean);
        daal::services::daal_free(secondOrderRawMoment);
        return errcode;
    }

    static int xoutlierdetection(float *data, __int64 nFeatures, __int64 nVectors,  __int64 nParams,
                          float *baconParams, float *baconWeights)
    {
        tDAAL_VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSNewTask, (&task, &nFeatures, &nVectors, &dataStorage, data, 0, 0,
                                                              __SS_ILP_FLAG__), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditOutDetect, (task, &nParams, baconParams, baconWeights),
                          errcode);

        ThreadingFuncs threading =
        {
            _daal_mkl_threader_for,
            _daal_mkl_threader_for_ordered,
            _daal_mkl_threader_sections,
            _daal_mkl_threader_ordered,
            _daal_mkl_threader_get_max_threads
        };

        __DAAL_VSLFN_CALL(fpk_vsl_kernel, sSSOutliersDetection, (task, __DAAL_VSL_SS_OUTLIERS,
                                                                 __DAAL_VSL_SS_METHOD_BACON, &threading), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode);
        return errcode;
    }

    static int xLowOrderMoments(float *data, __int64 nFeatures, __int64 nVectors, __int64 method,
                         float *sum, float *mean, float *secondOrderRawMoment,
                         float *variance, float *variation)
    {
        tDAAL_VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSNewTask, (&task, &nFeatures, &nVectors, &dataStorage, data,
                                                              0, 0, __SS_ILP_FLAG__), errcode);

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_SUM, sum), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_MEAN, mean), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_2R_MOM,
                                                               secondOrderRawMoment), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_2C_MOM, variance), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_VARIATION, variation), errcode);

        ThreadingFuncs threading =
        {
            _daal_mkl_threader_for,
            _daal_mkl_threader_for_ordered,
            _daal_mkl_threader_sections,
            _daal_mkl_threader_ordered,
            _daal_mkl_threader_get_max_threads
        };

        __DAAL_VSLFN_CALL(fpk_vsl_kernel, sSSBasic, (task,
                                                     __DAAL_VSL_SS_SUM | __DAAL_VSL_SS_MEAN | __DAAL_VSL_SS_2R_MOM |
                                                     __DAAL_VSL_SS_2C_MOM | __DAAL_VSL_SS_VARIATION,
                                                     method, &threading), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode);
        return errcode;
    }

    static int xSumAndVariance(float *data, __int64 nFeatures, __int64 nVectors, float *nPreviousObservations,
                        __int64 method, float *sum, float *mean, float *secondOrderRawMoment,
                        float *variance)
    {
        tDAAL_VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSNewTask, (&task, &nFeatures, &nVectors, &dataStorage, data,
                                                              0, 0, __SS_ILP_FLAG__), errcode);

        float weight[2] = { *nPreviousObservations, *nPreviousObservations };

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_SUM, sum), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_MEAN, mean), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_2R_MOM,
                                                               secondOrderRawMoment), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_2C_MOM, variance), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_ACCUM_WEIGHT, weight), errcode);

        ThreadingFuncs threading =
        {
            _daal_mkl_threader_for,
            _daal_mkl_threader_for_ordered,
            _daal_mkl_threader_sections,
            _daal_mkl_threader_ordered,
            _daal_mkl_threader_get_max_threads
        };

        __DAAL_VSLFN_CALL(fpk_vsl_kernel, sSSBasic, (task,
                                                     __DAAL_VSL_SS_SUM | __DAAL_VSL_SS_MEAN | __DAAL_VSL_SS_2R_MOM |
                                                     __DAAL_VSL_SS_2C_MOM, method, &threading), errcode);
        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode);
        return errcode;
    }

    static int xQuantiles(float *data, __int64 nFeatures, __int64 nVectors, __int64 quantOrderN, float* quantOrder, float *quants)
    {
        tDAAL_VSLSSTaskPtr task;
        int errcode = 0;

        __int64 dataStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSNewTask, (&task, &nFeatures, &nVectors, &dataStorage, data, 0, 0, __SS_ILP_FLAG__), errcode);
        if(errcode) {return errcode;}

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsliSSEditTask, (task, __DAAL_VSL_SS_ED_QUANT_ORDER_N, &quantOrderN), errcode);
        if(errcode) { __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode); return errcode; }

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_QUANT_ORDER, quantOrder), errcode);
        if(errcode) { __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode); return errcode; }

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_QUANT_QUANTILES, quants), errcode);
        if(errcode) { __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode); return errcode; }

        ThreadingFuncs threading =
        {
            _daal_mkl_threader_for,
            _daal_mkl_threader_for_ordered,
            _daal_mkl_threader_sections,
            _daal_mkl_threader_ordered,
            _daal_mkl_threader_get_max_threads
        };

        __DAAL_VSLFN_CALL(fpk_vsl_kernel, sSSQuantiles, (task, __DAAL_VSL_SS_QUANTS, __DAAL_VSL_SS_METHOD_FAST, &threading), errcode);
        if(errcode) { __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode); return errcode; }

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode);
        return errcode;
    }

    static int xSort(float *data, __int64 nFeatures, __int64 nVectors, float *sortedData)
    {
        tDAAL_VSLSSTaskPtr task;
        int errcode = 0;

        __int64 inputStorage  = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;
        __int64 outputStorage = __DAAL_VSL_SS_MATRIX_STORAGE_COLS;

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSNewTask, (&task, &nFeatures, &nVectors, &inputStorage, data, 0, 0, __SS_ILP_FLAG__), errcode);
        if(errcode) {return errcode;}

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslsSSEditTask, (task, __DAAL_VSL_SS_ED_SORTED_OBSERV, sortedData), errcode);
        if(errcode) { __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode); return errcode; }

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vsliSSEditTask, (task, __DAAL_VSL_SS_ED_SORTED_OBSERV_STORAGE, &outputStorage), errcode);
        if(errcode) { __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode); return errcode; }

        ThreadingFuncs threading =
        {
            _daal_mkl_threader_for,
            _daal_mkl_threader_for_ordered,
            _daal_mkl_threader_sections,
            _daal_mkl_threader_ordered,
            _daal_mkl_threader_get_max_threads
        };

        __DAAL_VSLFN_CALL(fpk_vsl_kernel, sSSSort, (task, __DAAL_VSL_SS_SORTED_OBSERV, __DAAL_VSL_SS_METHOD_RADIX, &threading), errcode);
        if(errcode) { __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode); return errcode; }

        __DAAL_VSLFN_CALL(fpk_vsl_sub_kernel, vslSSDeleteTask, (&task), errcode);
        return errcode;
    }
};

} // namespace mkl
} // namespace internal
} // namespace daal

#endif
