/* file: service_stat_rng_mkl.h */
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

#ifndef __SERVICE_STAT_RNG_MKL_H__
#define __SERVICE_STAT_RNG_MKL_H__

#if !defined(__DAAL_CONCAT5)
    #define __DAAL_CONCAT5(a, b, c, d, e)  __DAAL_CONCAT51(a, b, c, d, e)
    #define __DAAL_CONCAT51(a, b, c, d, e) a##b##c##d##e
#endif

#define __DAAL_VSLFN(f_cpu, f_pref, f_name)                   __DAAL_CONCAT5(f_pref, _, f_cpu, _, f_name)
#define __DAAL_VSLFN_CALL(f_pref, f_name, f_args, errcode)    __DAAL_VSLFN_CALL1(f_pref, f_name, f_args, errcode)
#define __DAAL_VSLFN_CALL_NR(f_pref, f_name, f_args, errcode) __DAAL_VSLFN_CALL_NO_V(f_pref, f_name, f_args, errcode)
#define __DAAL_VSLFN_CALL_NR_WHILE(f_pref, f_name, f_args, errcode)   \
    {                                                                 \
        size_t nn_left = n;                                           \
        while (nn_left > 0)                                           \
        {                                                             \
            nn = (nn_left > 0xFFFFFFFL) ? 0xFFFFFFF : (int)(nn_left); \
                                                                      \
            __DAAL_VSLFN_CALL_V(f_pref, f_name, f_args, errcode);     \
            if (errcode < 0) return errcode;                          \
                                                                      \
            rr += nn;                                                 \
            nn_left -= nn;                                            \
        }                                                             \
    }

#if defined(__APPLE__)
    #define __DAAL_MKLVSL_SSE2  e9
    #define __DAAL_MKLVSL_SSE42 e9
#else
    #define __DAAL_MKLVSL_SSE2  ex
    #define __DAAL_MKLVSL_SSE42 h8
#endif

#define __DAAL_VSLFN_CALL1(f_pref, f_name, f_args, errcode)                 \
    if (avx512 == cpu)                                                      \
    {                                                                       \
        errcode = __DAAL_VSLFN(z0, f_pref, f_name) f_args;                  \
    }                                                                       \
    if (avx2 == cpu)                                                        \
    {                                                                       \
        errcode = __DAAL_VSLFN(l9, f_pref, f_name) f_args;                  \
    }                                                                       \
    if (sse42 == cpu)                                                       \
    {                                                                       \
        errcode = __DAAL_VSLFN(__DAAL_MKLVSL_SSE42, f_pref, f_name) f_args; \
    }                                                                       \
    if (sse2 == cpu)                                                        \
    {                                                                       \
        errcode = __DAAL_VSLFN(__DAAL_MKLVSL_SSE2, f_pref, f_name) f_args;  \
    }                                                                       \
    if (errcode != 0)                                                       \
    {                                                                       \
        return errcode;                                                     \
    }

#define __DAAL_VSLFN_CALL2(f_pref, f_name, f_args, retcode)                 \
    if (avx512 == cpu)                                                      \
    {                                                                       \
        retcode = __DAAL_VSLFN(z0, f_pref, f_name) f_args;                  \
    }                                                                       \
    if (avx2 == cpu)                                                        \
    {                                                                       \
        retcode = __DAAL_VSLFN(l9, f_pref, f_name) f_args;                  \
    }                                                                       \
    if (sse42 == cpu)                                                       \
    {                                                                       \
        retcode = __DAAL_VSLFN(__DAAL_MKLVSL_SSE42, f_pref, f_name) f_args; \
    }                                                                       \
    if (sse2 == cpu)                                                        \
    {                                                                       \
        retcode = __DAAL_VSLFN(__DAAL_MKLVSL_SSE2, f_pref, f_name) f_args;  \
    }

#define __DAAL_VSLFN_CALL_V(f_pref, f_name, f_args, retcode) v##f_name f_args;

#define __DAAL_VSLFN_CALL_NO_V(f_pref, f_name, f_args, retcode) f_name f_args;

#endif
