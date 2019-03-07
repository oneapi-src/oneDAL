/* file: service_data_utils.h */
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
//  Declaration of service constants
//--
*/

#ifndef __SERVICE_DATA_UTILS_H__
#define __SERVICE_DATA_UTILS_H__

#include "service_defines.h"

namespace daal
{
namespace services
{
namespace internal
{

template<typename T>
struct MaxVal
{
    DAAL_FORCEINLINE static T get()
    {
        return 0;
    }
};

template<>
struct MaxVal<int>
{
    DAAL_FORCEINLINE static int get()
    {
        return INT_MAX;
    }
};

template<>
struct MaxVal<double>
{
    DAAL_FORCEINLINE static double get()
    {
        return DBL_MAX;
    }
};

template<>
struct MaxVal<float>
{
    DAAL_FORCEINLINE static float get()
    {
        return FLT_MAX;
    }
};

template<typename T>
struct MinVal
{
    DAAL_FORCEINLINE static T get()
    {
        return 0;
    }
};

template<>
struct MinVal<int>
{
    DAAL_FORCEINLINE static int get()
    {
        return INT_MIN;
    }
};

template<>
struct MinVal<double>
{
    DAAL_FORCEINLINE static double get()
    {
        return DBL_MIN;
    }
};

template<>
struct MinVal<float>
{
    DAAL_FORCEINLINE static float get()
    {
        return FLT_MIN;
    }
};

template<typename T>
struct EpsilonVal
{
    DAAL_FORCEINLINE static T get()
    {
        return 0;
    }
};

template<>
struct EpsilonVal<double>
{
    DAAL_FORCEINLINE static double get()
    {
        return DBL_EPSILON;
    }
};

template<>
struct EpsilonVal<float>
{
    DAAL_FORCEINLINE static float get()
    {
        return FLT_EPSILON;
    }
};

template<typename T, CpuType cpu>
struct SignBit;

template<CpuType cpu>
struct SignBit<float, cpu>
{
    static int get(float val)
    {
        return ((_daal_sp_union_t*)&val)->bits.sign;
    }
};

template<CpuType cpu>
struct SignBit<double, cpu>
{
    static int get(double val)
    {
        return ((_daal_dp_union_t*)&val)->bits.sign;
    }
};

template<typename T1, typename T2, CpuType cpu>
void vectorConvertFuncCpu(size_t n, void *src, void *dst);

template<typename T1, typename T2, CpuType cpu>
void vectorStrideConvertFuncCpu(size_t n, void *src, size_t srcByteStride, void *dst, size_t dstByteStride);

} // namespace internal
} // namespace services
} // namespace daal

#endif
