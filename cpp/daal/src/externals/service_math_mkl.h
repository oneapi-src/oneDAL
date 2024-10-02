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
#include <mkl.h>
#include "src/services/service_defines.h"

#define __DAAL_MKLFN_CALL_MATH(f_name, f_args) \
    f_name f_args;                             \
    return;

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

    static double xsPowx(double in, double in1)
    {
        double r;
        xvPowx(1, &in, in1, &r);
        return r;
    }

    static double sCeil(double in)
    {
        double r;
        vCeil(1, &in, &r);
        return r;
    }

    static double xsCeil(double in)
    {
        double r;
        xvCeil(1, &in, &r);
        return r;
    }

    static double sErfInv(double in)
    {
        double r;
        vErfInv(1, &in, &r);
        return r;
    }

    static double xsErfInv(double in)
    {
        double r;
        xvErfInv(1, &in, &r);
        return r;
    }

    static double sErf(double in)
    {
        double r;
        vErf(1, &in, &r);
        return r;
    }

    static double xsErf(double in)
    {
        double r;
        xvErf(1, &in, &r);
        return r;
    }

    static double sLog(double in)
    {
        double r;
        vLog(1, &in, &r);
        return r;
    }

    static double xsLog(double in)
    {
        double r;
        xvLog(1, &in, &r);
        return r;
    }

    static double sCdfNormInv(double in)
    {
        double r;
        vCdfNormInv(1, &in, &r);
        return r;
    }

    static double xsCdfNormInv(double in)
    {
        double r;
        xvCdfNormInv(1, &in, &r);
        return r;
    }

    static void vPowx(SizeType n, const double * in, double in1, double * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmdPowx, ((MKL_INT)n, in, in1, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void xvPowx(SizeType n, const double * in, double in1, double * out)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_MATH(vmdPowx, ((MKL_INT)n, in, in1, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
        mkl_set_num_threads_local(old_nthr);
    }

    static void vCeil(SizeType n, const double * in, double * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmdCeil, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void xvCeil(SizeType n, const double * in, double * out)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_MATH(vmdCeil, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
        mkl_set_num_threads_local(old_nthr);
    }

    static void vErfInv(SizeType n, const double * in, double * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmdErfInv, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void xvErfInv(SizeType n, const double * in, double * out)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_MATH(vmdErfInv, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
        mkl_set_num_threads_local(old_nthr);
    }

    static void vErf(SizeType n, const double * in, double * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmdErf, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void xvErf(SizeType n, const double * in, double * out)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_MATH(vmdErf, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
        mkl_set_num_threads_local(old_nthr);
    }

    static void vExp(SizeType n, const double * in, double * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmdExp, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void xvExp(SizeType n, const double * in, double * out)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_MATH(vmdExp, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
        mkl_set_num_threads_local(old_nthr);
    }

    static double vExpThreshold() { return -650.0; }

    static void vTanh(SizeType n, const double * in, double * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmdTanh, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void xvTanh(SizeType n, const double * in, double * out)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_MATH(vmdTanh, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
        mkl_set_num_threads_local(old_nthr);
    }

    static void vSqrt(SizeType n, const double * in, double * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmdSqrt, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void xvSqrt(SizeType n, const double * in, double * out)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_MATH(vmdSqrt, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
        mkl_set_num_threads_local(old_nthr);
    }

    static void vLog(SizeType n, const double * in, double * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmdLn, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void xvLog(SizeType n, const double * in, double * out)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_MATH(vmdLn, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
        mkl_set_num_threads_local(old_nthr);
    }

    static void vLog1p(SizeType n, const double * in, double * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmdLog1p, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void xvLog1p(SizeType n, const double * in, double * out)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_MATH(vmdLog1p, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
        mkl_set_num_threads_local(old_nthr);
    }

    static void vCdfNormInv(SizeType n, const double * in, double * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmdCdfNormInv, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void xvCdfNormInv(SizeType n, const double * in, double * out)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_MATH(vmdCdfNormInv, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
        mkl_set_num_threads_local(old_nthr);
    }
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

    static float xsPowx(float in, float in1)
    {
        float r;
        xvPowx(1, &in, in1, &r);
        return r;
    }

    static float sCeil(float in)
    {
        float r;
        vCeil(1, &in, &r);
        return r;
    }

    static float xsCeil(float in)
    {
        float r;
        xvCeil(1, &in, &r);
        return r;
    }

    static float sErfInv(float in)
    {
        float r;
        vErfInv(1, &in, &r);
        return r;
    }

    static float xsErfInv(float in)
    {
        float r;
        xvErfInv(1, &in, &r);
        return r;
    }

    static float sErf(float in)
    {
        float r;
        vErf(1, &in, &r);
        return r;
    }

    static float xsErf(float in)
    {
        float r;
        xvErf(1, &in, &r);
        return r;
    }

    static float sLog(float in)
    {
        float r;
        vLog(1, &in, &r);
        return r;
    }

    static float xsLog(float in)
    {
        float r;
        xvLog(1, &in, &r);
        return r;
    }

    static float sCdfNormInv(float in)
    {
        float r;
        vCdfNormInv(1, &in, &r);
        return r;
    }

    static float xsCdfNormInv(float in)
    {
        float r;
        xvCdfNormInv(1, &in, &r);
        return r;
    }

    static void vPowx(SizeType n, const float * in, float in1, float * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmsPowx, ((MKL_INT)n, in, in1, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void xvPowx(SizeType n, const float * in, float in1, float * out)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_MATH(vmsPowx, ((MKL_INT)n, in, in1, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
        mkl_set_num_threads_local(old_nthr);
    }

    static void vCeil(SizeType n, const float * in, float * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmsCeil, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void xvCeil(SizeType n, const float * in, float * out)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_MATH(vmsCeil, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
        mkl_set_num_threads_local(old_nthr);
    }

    static void vErfInv(SizeType n, const float * in, float * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmsErfInv, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void xvErfInv(SizeType n, const float * in, float * out)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_MATH(vmsErfInv, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
        mkl_set_num_threads_local(old_nthr);
    }

    static void vErf(SizeType n, const float * in, float * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmsErf, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void xvErf(SizeType n, const float * in, float * out)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_MATH(vmsErf, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
        mkl_set_num_threads_local(old_nthr);
    }

    static void vExp(SizeType n, const float * in, float * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmsExp, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void xvExp(SizeType n, const float * in, float * out)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_MATH(vmsExp, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
        mkl_set_num_threads_local(old_nthr);
    }

    static float vExpThreshold() { return -75.0f; }

    static void vTanh(SizeType n, const float * in, float * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmsTanh, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void xvTanh(SizeType n, const float * in, float * out)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_MATH(vmsTanh, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
        mkl_set_num_threads_local(old_nthr);
    }

    static void vSqrt(SizeType n, const float * in, float * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmsSqrt, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void xvSqrt(SizeType n, const float * in, float * out)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_MATH(vmsSqrt, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
        mkl_set_num_threads_local(old_nthr);
    }

    static void vLog(SizeType n, const float * in, float * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmsLn, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void xvLog(SizeType n, const float * in, float * out)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_MATH(vmsLn, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
        mkl_set_num_threads_local(old_nthr);
    }

    static void vLog1p(SizeType n, const float * in, float * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmsLog1p, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void xvLog1p(SizeType n, const float * in, float * out)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_MATH(vmsLog1p, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
        mkl_set_num_threads_local(old_nthr);
    }

    static void vCdfNormInv(SizeType n, const float * in, float * out)
    {
        __DAAL_MKLFN_CALL_MATH(vmsCdfNormInv, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
    }

    static void xvCdfNormInv(SizeType n, const float * in, float * out)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_MATH(vmsCdfNormInv, ((MKL_INT)n, in, out, (MKL_INT)(VML_HA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)));
        mkl_set_num_threads_local(old_nthr);
    }
};

} // namespace mkl
} // namespace internal
} // namespace daal

#endif
