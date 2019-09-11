/* file: service_stat_rng_mkl.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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

/*
//++
//  Template wrappers for common Intel(R) MKL functions.
//--
*/

#ifndef __SERVICE_STAT_RNG_MKL_H__
#define __SERVICE_STAT_RNG_MKL_H__


#if !defined(__DAAL_CONCAT5)
    #define __DAAL_CONCAT5(a,b,c,d,e) __DAAL_CONCAT51(a,b,c,d,e)
    #define __DAAL_CONCAT51(a,b,c,d,e) a##b##c##d##e
#endif

#define __DAAL_VSLFN(f_cpu,f_pref,f_name)        __DAAL_CONCAT5(f_pref,_,f_cpu,_,f_name)
#define __DAAL_VSLFN_CALL(f_pref,f_name,f_args,errcode)  __DAAL_VSLFN_CALL1(f_pref,f_name,f_args,errcode)
#define __DAAL_VSLFN_CALL_NR(f_pref,f_name,f_args,errcode)  __DAAL_VSLFN_CALL2(f_pref,f_name,f_args,errcode)
#define __DAAL_VSLFN_CALL_NR_WHILE(f_pref,f_name,f_args,errcode)          \
{                                                                         \
    size_t nn_left = n;                                                   \
    while( nn_left > 0 )                                                  \
    {                                                                     \
        nn = ( nn_left > 0xFFFFFFFL )? 0xFFFFFFF : (int)( nn_left );      \
                                                                          \
        __DAAL_VSLFN_CALL2(f_pref,f_name,f_args,errcode);                 \
        if( errcode < 0 ) return errcode;                                 \
                                                                          \
        rr      += nn;                                                    \
        nn_left -= nn;                                                    \
    }                                                                     \
}

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

#endif
