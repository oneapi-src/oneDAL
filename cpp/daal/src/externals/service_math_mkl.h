/* file: service_math_mkl.h */
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
//  Declaration of math service functions
//--
*/

#ifndef __SERVICE_MATH_MKL_H__
#define __SERVICE_MATH_MKL_H__

#include <math.h>
#include <mkl_vml.h>
#include <mkl_vsl.h>
#include <mkl.h>
#include "src/services/service_defines.h"

#if !defined(__DAAL_CONCAT5)
    #define __DAAL_CONCAT5(a, b, c, d, e)  __DAAL_CONCAT51(a, b, c, d, e)
    #define __DAAL_CONCAT51(a, b, c, d, e) a##b##c##d##e
#endif

#define VMLFN(f_cpu, f_name, f_suff) __DAAL_CONCAT5(fpk_vml_, f_name, _, f_cpu, f_suff)
// #define VMLFN_CALL(f_name, f_suff, f_args) VMLFN_CALL1(f_name, f_suff, f_args)
#define VMLFN_CALL(f_name, f_suff, f_args) \
    v##f_name f_args;                      \
    return;

#if defined(__APPLE__)
    #define __DAAL_MKLVML_SSE2  E9
    #define __DAAL_MKLVML_SSE42 E9
#else
    #define __DAAL_MKLVML_SSE2  EX
    #define __DAAL_MKLVML_SSE42 H8
#endif

#define VMLFN_CALL1(f_name, f_suff, f_args)                \
    if (avx512 == cpu)                                     \
    {                                                      \
        VMLFN(Z0, f_name, f_suff) f_args;                  \
        return;                                            \
    }                                                      \
    if (avx2 == cpu)                                       \
    {                                                      \
        VMLFN(L9, f_name, f_suff) f_args;                  \
        return;                                            \
    }                                                      \
    if (sse42 == cpu)                                      \
    {                                                      \
        VMLFN(__DAAL_MKLVML_SSE42, f_name, f_suff) f_args; \
        return;                                            \
    }                                                      \
    if (sse2 == cpu)                                       \
    {                                                      \
        VMLFN(__DAAL_MKLVML_SSE2, f_name, f_suff) f_args;  \
        return;                                            \
    }

namespace daal
{
namespace internal
{
namespace mkl
{
template <typename fpType, CpuType cpu>
struct MklMath
{};

/*
// Double precision functions definition
*/

template <CpuType cpu>
struct MklMath<double, cpu>
{
    typedef size_t SizeType;

    static double sFabs(double in) { return (in >= 0.0f) ? in : -in; }

    static double sMin(double in1, double in2) { return (in1 > in2) ? in2 : in1; }

    static double sMax(double in1, double in2) { return (in1 < in2) ? in2 : in1; }

    static double sSqrt(double in) { return sqrt(in); }

    static double sPowx(double in, double in1)
    {
        double r;
        vPowx(1, &in, in1, &r);
        return r;
    }

    static double sCeil(double in)
    {
        double r;
        vCeil(1, &in, &r);
        return r;
    }

    static double sErfInv(double in)
    {
        double r;
        vErfInv(1, &in, &r);
        return r;
    }

    static double sErf(double in)
    {
        double r;
        vErf(1, &in, &r);
        return r;
    }

    static double sLog(double in)
    {
        double r;
        vLog(1, &in, &r);
        return r;
    }

    static double sCdfNormInv(double in)
    {
        double r;
        vCdfNormInv(1, &in, &r);
        return r;
    }

    static void vPowx(SizeType n, const double * in, double in1, double * out) { VMLFN_CALL(dPowx, HAynn, ((int)n, in, in1, out)); }

    static void vCeil(SizeType n, const double * in, double * out) { VMLFN_CALL(dCeil, HAynn, ((int)n, in, out)); }

    static void vErfInv(SizeType n, const double * in, double * out) { VMLFN_CALL(dErfInv, HAynn, ((int)n, in, out)); }

    static void vErf(SizeType n, const double * in, double * out) { VMLFN_CALL(dErf, HAynn, ((int)n, in, out)); }

    static void vExp(SizeType n, const double * in, double * out) { VMLFN_CALL(dExp, HAynn, ((int)n, in, out)); }

    static double vExpThreshold() { return -650.0; }

    static void vTanh(SizeType n, const double * in, double * out) { VMLFN_CALL(dTanh, HAynn, ((int)n, in, out)); }

    static void vSqrt(SizeType n, const double * in, double * out) { VMLFN_CALL(dSqrt, HAynn, ((int)n, in, out)); }

    static void vLog(SizeType n, const double * in, double * out) { VMLFN_CALL(dLn, HAynn, ((int)n, in, out)); }

    static void vLog1p(SizeType n, const double * in, double * out) { VMLFN_CALL(dLog1p, HAynn, ((int)n, in, out)); }

    static void vCdfNormInv(SizeType n, const double * in, double * out) { VMLFN_CALL(dCdfNormInv, HAynn, ((int)n, in, out)); }
};

/*
// Single precision functions definition
*/

template <CpuType cpu>
struct MklMath<float, cpu>
{
    typedef size_t SizeType;

    static float sFabs(float in) { return (in >= 0.0) ? in : -in; }

    static float sMin(float in1, float in2) { return (in1 > in2) ? in2 : in1; }

    static float sMax(float in1, float in2) { return (in1 < in2) ? in2 : in1; }

    static float sSqrt(float in) { return sqrt(in); }

    static float sPowx(float in, float in1)
    {
        float r;
        vPowx(1, &in, in1, &r);
        return r;
    }

    static float sCeil(float in)
    {
        float r;
        vCeil(1, &in, &r);
        return r;
    }

    static float sErfInv(float in)
    {
        float r;
        vErfInv(1, &in, &r);
        return r;
    }

    static float sErf(float in)
    {
        float r;
        vErf(1, &in, &r);
        return r;
    }

    static float sLog(float in)
    {
        float r;
        vLog(1, &in, &r);
        return r;
    }

    static float sCdfNormInv(float in)
    {
        float r;
        vCdfNormInv(1, &in, &r);
        return r;
    }

    static void vPowx(SizeType n, const float * in, float in1, float * out) { VMLFN_CALL(sPowx, HAynn, ((int)n, in, in1, out)); }

    static void vCeil(SizeType n, const float * in, float * out) { VMLFN_CALL(sCeil, HAynn, ((int)n, in, out)); }

    static void vErfInv(SizeType n, const float * in, float * out) { VMLFN_CALL(sErfInv, HAynn, ((int)n, in, out)); }

    static void vErf(SizeType n, const float * in, float * out) { VMLFN_CALL(sErf, HAynn, ((int)n, in, out)); }

    static void vExp(SizeType n, const float * in, float * out) { VMLFN_CALL(sExp, HAynn, ((int)n, in, out)); }

    static float vExpThreshold() { return -75.0f; }

    static void vTanh(SizeType n, const float * in, float * out) { VMLFN_CALL(sTanh, HAynn, ((int)n, in, out)); }

    static void vSqrt(SizeType n, const float * in, float * out) { VMLFN_CALL(sSqrt, HAynn, ((int)n, in, out)); }

    static void vLog(SizeType n, const float * in, float * out) { VMLFN_CALL(sLn, HAynn, ((int)n, in, out)); }

    static void vLog1p(SizeType n, const float * in, float * out) { VMLFN_CALL(sLog1p, HAynn, ((int)n, in, out)); }

    static void vCdfNormInv(SizeType n, const float * in, float * out) { VMLFN_CALL(sCdfNormInv, HAynn, ((int)n, in, out)); }
};

} // namespace mkl
} // namespace internal
} // namespace daal

#endif
