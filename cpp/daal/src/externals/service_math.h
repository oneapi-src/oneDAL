/* file: service_math.h */
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
//  Template wrappers for math functions.
//--
*/

#ifndef __SERVICE_MATH_H__
#define __SERVICE_MATH_H__

#include "src/services/service_defines.h"

#include "src/externals/config.h"

namespace daal
{
namespace internal
{
/*
// Template functions definition
*/
template <typename fpType, CpuType cpu, template <typename, CpuType> class _impl>
struct Math
{
    typedef typename _impl<fpType, cpu>::SizeType SizeType;

    static fpType sFabs(fpType in) { return _impl<fpType, cpu>::sFabs(in); }

    static fpType sMin(fpType in1, fpType in2) { return _impl<fpType, cpu>::sMin(in1, in2); }

    static fpType sMax(fpType in1, fpType in2) { return _impl<fpType, cpu>::sMax(in1, in2); }

    static fpType sSqrt(fpType in) { return _impl<fpType, cpu>::sSqrt(in); }

    static fpType sPowx(fpType in, fpType in1) { return _impl<fpType, cpu>::sPowx(in, in1); }

    static fpType xsPowx(fpType in, fpType in1) { return _impl<fpType, cpu>::xsPowx(in, in1); }

    static fpType sCeil(fpType in) { return _impl<fpType, cpu>::sCeil(in); }

    static fpType xsCeil(fpType in) { return _impl<fpType, cpu>::xsCeil(in); }

    static fpType sErfInv(fpType in) { return _impl<fpType, cpu>::sErfInv(in); }

    static fpType xsErfInv(fpType in) { return _impl<fpType, cpu>::xsErfInv(in); }

    static fpType sErf(fpType in) { return _impl<fpType, cpu>::sErf(in); }

    static fpType xsErf(fpType in) { return _impl<fpType, cpu>::xsErf(in); }

    static fpType sLog(fpType in) { return _impl<fpType, cpu>::sLog(in); }

    static fpType xsLog(fpType in) { return _impl<fpType, cpu>::xsLog(in); }

    static fpType sCdfNormInv(fpType in) { return _impl<fpType, cpu>::sCdfNormInv(in); }

    static fpType xsCdfNormInv(fpType in) { return _impl<fpType, cpu>::xsCdfNormInv(in); }

    static void vPowx(SizeType n, const fpType * in, fpType in1, fpType * out) { _impl<fpType, cpu>::vPowx(n, in, in1, out); }

    static void xvPowx(SizeType n, const fpType * in, fpType in1, fpType * out) { _impl<fpType, cpu>::xvPowx(n, in, in1, out); }

    static void vPowxAsLnExp(SizeType n, const fpType * in, fpType in1, fpType * out)
    {
        _impl<fpType, cpu>::vLog(n, in, out);
        for (size_t i = 0; i < n; i++)
        {
            out[i] *= in1;
        }
        _impl<fpType, cpu>::vExp(n, out, out);
    }

    static void xvPowxAsLnExp(SizeType n, const fpType * in, fpType in1, fpType * out)
    {
        _impl<fpType, cpu>::xvLog(n, in, out);
        for (size_t i = 0; i < n; i++)
        {
            out[i] *= in1;
        }
        _impl<fpType, cpu>::xvExp(n, out, out);
    }

    static void vCeil(SizeType n, const fpType * in, fpType * out) { _impl<fpType, cpu>::vCeil(n, in, out); }

    static void xvCeil(SizeType n, const fpType * in, fpType * out) { _impl<fpType, cpu>::xvCeil(n, in, out); }

    static void vErfInv(SizeType n, const fpType * in, fpType * out) { _impl<fpType, cpu>::vErfInv(n, in, out); }

    static void xvErfInv(SizeType n, const fpType * in, fpType * out) { _impl<fpType, cpu>::xvErfInv(n, in, out); }

    static void vErf(SizeType n, const fpType * in, fpType * out) { _impl<fpType, cpu>::vErf(n, in, out); }

    static void xvErf(SizeType n, const fpType * in, fpType * out) { _impl<fpType, cpu>::xvErf(n, in, out); }

    static void vExp(SizeType n, const fpType * in, fpType * out) { _impl<fpType, cpu>::vExp(n, in, out); }

    static void xvExp(SizeType n, const fpType * in, fpType * out) { _impl<fpType, cpu>::xvExp(n, in, out); }

    static fpType vExpThreshold() { return _impl<fpType, cpu>::vExpThreshold(); }

    static void vTanh(SizeType n, const fpType * in, fpType * out) { _impl<fpType, cpu>::vTanh(n, in, out); }

    static void xvTanh(SizeType n, const fpType * in, fpType * out) { _impl<fpType, cpu>::xvTanh(n, in, out); }

    static void vSqrt(SizeType n, const fpType * in, fpType * out) { _impl<fpType, cpu>::vSqrt(n, in, out); }

    static void xvSqrt(SizeType n, const fpType * in, fpType * out) { _impl<fpType, cpu>::xvSqrt(n, in, out); }

    static void vLog(SizeType n, const fpType * in, fpType * out) { _impl<fpType, cpu>::vLog(n, in, out); }

    static void xvLog(SizeType n, const fpType * in, fpType * out) { _impl<fpType, cpu>::xvLog(n, in, out); }

    static void vLog1p(SizeType n, const fpType * in, fpType * out) { _impl<fpType, cpu>::vLog1p(n, in, out); }

    static void xvLog1p(SizeType n, const fpType * in, fpType * out) { _impl<fpType, cpu>::xvLog1p(n, in, out); }

    static void vCdfNormInv(SizeType n, const fpType * in, fpType * out) { _impl<fpType, cpu>::vCdfNormInv(n, in, out); }

    static void xvCdfNormInv(SizeType n, const fpType * in, fpType * out) { _impl<fpType, cpu>::xvCdfNormInv(n, in, out); }
};

} // namespace internal
} // namespace daal

namespace daal
{
namespace internal
{
template <typename fpType, CpuType cpu>
using MathInst = Math<fpType, cpu, MathBackend>;
} // namespace internal
} // namespace daal

#endif
