/* file: service_math.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
//  Template wrappers for math functions.
//--
*/

#ifndef __SERVICE_MATH_H__
#define __SERVICE_MATH_H__

#include "service_defines.h"
#include "service_math_mkl.h"

namespace daal
{
namespace internal
{

/*
// Template functions definition
*/
template<typename fpType, CpuType cpu, template<typename, CpuType> class _impl=mkl::MklMath>
struct Math
{
    typedef typename _impl<fpType,cpu>::SizeType SizeType;

    static fpType sFabs(fpType in)
    {
        return _impl<fpType,cpu>::sFabs(in);
    }

    static fpType sMin(fpType in1, fpType in2)
    {
        return _impl<fpType,cpu>::sMin(in1, in2);
    }

    static fpType sMax(fpType in1, fpType in2)
    {
        return _impl<fpType,cpu>::sMax(in1, in2);
    }

    static fpType sSqrt(fpType in)
    {
        return _impl<fpType,cpu>::sSqrt(in);
    }

    static fpType sPowx(fpType in, fpType in1)
    {
        return _impl<fpType,cpu>::sPowx(in, in1);
    }

    static fpType sCeil(fpType in)
    {
        return _impl<fpType,cpu>::sCeil(in);
    }

    static fpType sErfInv(fpType in)
    {
        return _impl<fpType,cpu>::sErfInv(in);
    }

    static fpType sErf(fpType in)
    {
        return _impl<fpType,cpu>::sErf(in);
    }

    static fpType sLog(fpType in)
    {
        return _impl<fpType,cpu>::sLog(in);
    }

    static fpType sCdfNormInv(fpType in)
    {
        return _impl<fpType,cpu>::sCdfNormInv(in);
    }

    static void vPowx(SizeType n, const fpType *in, fpType in1, fpType *out)
    {
        _impl<fpType,cpu>::vPowx(n, in, in1, out);
    }

    static void vPowxAsLnExp(SizeType n, const fpType *in, fpType in1, fpType *out)
    {
        _impl<fpType,cpu>::vLog(n, in, out);
        for(size_t i = 0; i < n ; i++) {out[i] *= in1;}
        _impl<fpType,cpu>::vExp(n, out, out);
    }

    static void vCeil(SizeType n, const fpType *in, fpType *out)
    {
        _impl<fpType,cpu>::vCeil(n, in, out);
    }

    static void vErfInv(SizeType n, const fpType *in, fpType *out)
    {
        _impl<fpType,cpu>::vErfInv(n, in, out);
    }

    static void vErf(SizeType n, const fpType *in, fpType *out)
    {
        _impl<fpType,cpu>::vErf(n, in, out);
    }

    static void vExp(SizeType n, const fpType *in, fpType* out)
    {
        _impl<fpType,cpu>::vExp(n, in, out);
    }

    static fpType vExpThreshold()
    {
        return _impl<fpType,cpu>::vExpThreshold();
    }

    static void vTanh(SizeType n, const fpType *in, fpType *out)
    {
        _impl<fpType,cpu>::vTanh(n, in, out);
    }

    static void vSqrt(SizeType n, const fpType *in, fpType *out)
    {
        _impl<fpType,cpu>::vSqrt(n, in, out);
    }

    static void vLog(SizeType n, const fpType *in, fpType *out)
    {
        _impl<fpType,cpu>::vLog(n, in, out);
    }

    static void vLog1p(SizeType n, const fpType *in, fpType *out)
    {
        _impl<fpType,cpu>::vLog1p(n, in, out);
    }

    static void vCdfNormInv(SizeType n, const fpType *in, fpType *out)
    {
        _impl<fpType,cpu>::vCdfNormInv(n, in, out);
    }
};

} // namespace internal
} // namespace daal

#endif
