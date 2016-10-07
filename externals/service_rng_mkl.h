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
#include "service_rng_common.h"


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

// RNGs
#define __DAAL_BRNG_MT19937                     VSL_BRNG_MT19937
#define __DAAL_RNG_METHOD_UNIFORM_STD           VSL_RNG_METHOD_UNIFORM_STD
#define __DAAL_RNG_METHOD_BERNOULLI_ICDF        VSL_RNG_METHOD_BERNOULLI_ICDF
#define __DAAL_RNG_METHOD_GAUSSIAN_BOXMULLER    VSL_RNG_METHOD_GAUSSIAN_BOXMULLER
#define __DAAL_RNG_METHOD_GAUSSIAN_BOXMULLER2   VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2
#define __DAAL_RNG_METHOD_GAUSSIAN_ICDF         VSL_RNG_METHOD_GAUSSIAN_ICDF

namespace daal
{
namespace internal
{
namespace mkl
{
/* Uniform distribution generator functions */
template<typename T, CpuType cpu>
int uniformRNG(const int n, T* r, void* stream, const T a, const T b, const int method);

template<CpuType cpu>
int uniformRNG(const int n, int* r, void* stream, const int a, const int b, const int method)
{
    int errcode = 0;
    __DAAL_VSLFN_CALL_NR(fpk_vsl_kernel, iRngUniform, ( method, stream, n, r, a, b ), errcode);
    return errcode;
}

template<CpuType cpu>
int uniformRNG(const int n, float* r, void* stream, const float a, const float b, const int method)
{
    int errcode = 0;
    __DAAL_VSLFN_CALL_NR(fpk_vsl_kernel, sRngUniform, ( method, stream, n, r, a, b ), errcode);
    return errcode;
}

template<CpuType cpu>
int uniformRNG(const int n, double* r, void* stream, const double a, const double b, const int method)
{
    int errcode = 0;
    __DAAL_VSLFN_CALL_NR(fpk_vsl_kernel, dRngUniform, ( method, stream, n, r, a, b ), errcode);
    return errcode;
}

/* Gaussian distribution generator functions */
template<typename T, CpuType cpu>
int gaussianRNG(const int n, T* r, void* stream, const T a, const T sigma, const int method);

template<CpuType cpu>
int gaussianRNG(const int n, float* r, void* stream, const float a, const float sigma, const int method)
{
    int errcode = 0;
    __DAAL_VSLFN_CALL_NR(fpk_vsl_kernel, sRngGaussian, ( method, stream, n, r, a, sigma ), errcode);
    return errcode;
}

template<CpuType cpu>
int gaussianRNG(const int n, double* r, void* stream, const double a, const double sigma, const int method)
{
    int errcode = 0;
    __DAAL_VSLFN_CALL_NR(fpk_vsl_kernel, dRngGaussian, ( method, stream, n, r, a, sigma ), errcode);
    return errcode;
}

/* Bernoulli distribution generator functions */
template<typename T, CpuType cpu>
int bernoulliRNG(const int n, T* r, void *stream, const double p, const int method);

template<CpuType cpu>
int bernoulliRNG(const int n, int* r, void *stream, const double p, const int method)
{
    int errcode = 0;
    __DAAL_VSLFN_CALL_NR(fpk_vsl_kernel, iRngBernoulli, (method, stream, n, r, p), errcode);
    return errcode;
}

template<CpuType cpu>
class BaseRNG : public BaseRNGIface<cpu>
{
public:
    BaseRNG(const unsigned int _seed, const int _brngId) : stream(0)
    {
        int errcode = 0;
        __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslNewStreamEx, ( &stream, _brngId, 1, &_seed ), errcode);
    }

    BaseRNG(const int n, const unsigned int* _seed, const int _brngId = __DAAL_BRNG_MT19937) : stream(0)
    {
        int errcode = 0;
        __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslNewStreamEx, ( &stream, _brngId, n, _seed ), errcode);
    }

    ~BaseRNG()
    {
        int errcode = 0;
        __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslDeleteStream, ( &stream ), errcode);
    }

    int getStateSize()
    {
        int res = 0;
        __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslGetStreamSize, (stream), res);
        return res;
    }

    int saveState(void* dest)
    {
        int errcode = 0;
        __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslSaveStreamM, (stream, (char*)dest), errcode);
        return errcode;
    }

    int loadState(const void* src)
    {
        int errcode = 0;
        __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslDeleteStream, (&stream), errcode);
        if(!errcode)
            __DAAL_VSLFN_CALL_NR(fpk_vsl_sub_kernel, vslLoadStreamM, (&stream, (const char*)src), errcode);
        return errcode;
    }

    void* getState()
    {
        return stream;
    }

private:
   void* stream;
};

/*
// Generator functions definition
*/
template<typename Type, CpuType cpu>
class RNGs
{
public:
    typedef DAAL_INT SizeType;
    typedef BaseRNG<cpu> BaseType;

    RNGs() {}

    int uniform(const SizeType n, Type* r, BaseType &brng, const Type a, const Type b, const int method)
    {
        return uniformRNG<cpu>(n, r, brng.getState(), a, b, method);
    }

    int bernoulli(const SizeType n, Type* r, BaseType &brng, const double p, const int method)
    {
        return bernoulliRNG<cpu>(n, r, brng.getState(), p, method);
    }

    int gaussian(const SizeType n, Type* r, BaseType &brng, const Type a, const Type sigma, const int method)
    {
        return gaussianRNG<cpu>(n, r, brng.getState(), a, sigma, method);
    }
};

}
}
}

#endif
