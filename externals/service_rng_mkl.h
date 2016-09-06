/* file: service_rng_mkl.h */
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
//  Declaration of math service functions
//--
*/

#ifndef __SERVICE_RNG_MKL_H__
#define __SERVICE_RNG_MKL_H__


#include "vmlvsl.h"
#include "service_defines.h"
#include "vmlvsl.h"


#if !defined(__DAAL_CONCAT5)
  #define __DAAL_CONCAT5(a,b,c,d,e) __DAAL_CONCAT51(a,b,c,d,e)
  #define __DAAL_CONCAT51(a,b,c,d,e) a##b##c##d##e
#endif

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

#define __DAAL_VSLFN(f_cpu,f_pref,f_name)        __DAAL_CONCAT5(f_pref,_,f_cpu,_,f_name)
#define __DAAL_VSLFN_CALL(f_pref,f_name,f_args,errcode)  __DAAL_VSLFN_CALL1(f_pref,f_name,f_args,errcode)
#define __DAAL_VSLFN_CALL_NR(f_pref,f_name,f_args,errcode)  __DAAL_VSLFN_CALL2(f_pref,f_name,f_args,errcode)

namespace daal
{
namespace internal
{
namespace mkl
{

template<typename baseType, CpuType cpu>
struct MklIntRng {};

/*
// Double precision functions definition
*/

template<CpuType cpu>
struct MklIntRng<int, cpu>
{
    typedef size_t SizeType;

    MklIntRng(int seed):
        BRNG((1<<20)*8), /* MT19937 */
        SEED(seed),
        METHOD(0), /* VSL_RNG_METHOD_BERNOULLI_ICDF */
        stream(0),
        errcode(0)
    {
        __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslNewStreamEx, ( &stream, BRNG, 1, &SEED ),errcode);
    }

    ~MklIntRng()
    {
        __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslDeleteStream, ( &stream ),errcode);
    }

    void uniform(size_t n, int a, int b, int* r)
    {
        __DAAL_VSLFN_CALL_NR(fpk_vsl_kernel, iRngUniform, ( METHOD, stream, n, r, a, b ),errcode);
    }

    void bernoulli(size_t n, int* r, double p)
    {
        __DAAL_VSLFN_CALL_NR(fpk_vsl_kernel, iRngBernoulli, ( METHOD, stream, n, r, p ),errcode);
    }

    int getStreamSize() const
    {
        int res = 0;
        __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslGetStreamSize, (stream), res);
        return res;
    }

    void saveStream(void* dest) const
    {
        __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslSaveStreamM, (stream, (char*)dest), errcode);
    }

    void loadStream(const void* src)
    {
        __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslDeleteStream, (&stream), errcode);
        if(!errcode)
            __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslLoadStreamM, (&stream, (const char*)src), errcode);
    }

private:
   const int BRNG;
   const unsigned int SEED;
   const int METHOD;
   void* stream;
   mutable int errcode;
};

template<typename baseType, CpuType cpu>
struct MklUniformRng {};

/*
// Double precision functions definition
*/

template<CpuType cpu>
struct MklUniformRng<float, cpu>
{
    typedef size_t SizeType;

    MklUniformRng(int seed):
        BRNG((1<<20)*8), /* MT19937 */
        SEED(seed),
        METHOD(0), /* VSL_RNG_METHOD_BERNOULLI_ICDF */
        stream(0),
        errcode(0)
    {
        __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslNewStreamEx, ( &stream, BRNG, 1, &SEED ),errcode);
    }

    ~MklUniformRng()
    {
        __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslDeleteStream, ( &stream ),errcode);
    }

    void uniform(size_t n, float a, float b, float* r)
    {
        __DAAL_VSLFN_CALL_NR(fpk_vsl_kernel, sRngUniform, ( METHOD, stream, n, r, a, b ),errcode);
    }

    int getStreamSize() const
    {
        int res = 0;
        __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslGetStreamSize, (stream), res);
        return res;
    }

    void saveStream(void* dest) const
    {
        __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslSaveStreamM, (stream, (char*)dest), errcode);
    }

    void loadStream(const void* src)
    {
        __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslDeleteStream, (&stream), errcode);
        if(!errcode)
            __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslLoadStreamM, (&stream, (const char*)src), errcode);
    }

private:
   const int BRNG;
   const unsigned int SEED;
   const int METHOD;
   void* stream;
   mutable int errcode;
};

template<CpuType cpu>
struct MklUniformRng<double, cpu>
{
    typedef size_t SizeType;

    MklUniformRng(int seed):
        BRNG((1<<20)*8), /* MT19937 */
        SEED(seed),
        METHOD(0), /* VSL_RNG_METHOD_BERNOULLI_ICDF */
        stream(0),
        errcode(0)
    {
        __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslNewStreamEx, ( &stream, BRNG, 1, &SEED ),errcode);
    }

    ~MklUniformRng()
    {
        __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslDeleteStream, ( &stream ),errcode);
    }

    void uniform(size_t n, double a, double b, double* r)
    {
        __DAAL_VSLFN_CALL_NR(fpk_vsl_kernel, dRngUniform, ( METHOD, stream, n, r, a, b ),errcode);
    }

    int getStreamSize() const
    {
        int res = 0;
        __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslGetStreamSize, (stream), res);
        return res;
    }

    void saveStream(void* dest) const
    {
        __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslSaveStreamM, (stream, (char*)dest), errcode);
    }

    void loadStream(const void* src)
    {
        __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslDeleteStream, (&stream), errcode);
        if(!errcode)
            __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslLoadStreamM, (&stream, (const char*)src), errcode);
    }

private:
   const int BRNG;
   const unsigned int SEED;
   const int METHOD;
   void* stream;
   mutable int errcode;
};

}
}
}

#endif
