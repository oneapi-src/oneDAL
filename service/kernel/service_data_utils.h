/* file: service_data_utils.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Declaration of service constants
//--
*/

#ifndef __SERVICE_DATA_UTILS_H__
#define __SERVICE_DATA_UTILS_H__

#include "data_utils.h"
#include "service_defines.h"

namespace daal
{
namespace data_feature_utils
{
namespace internal
{

template<typename T, CpuType cpu>
struct MaxVal
{
    static T get()
    {
        return 0;
    }
};

template<CpuType cpu>
struct MaxVal<int, cpu>
{
    static int get()
    {
        return INT_MAX;
    }
};

template<CpuType cpu>
struct MaxVal<double, cpu>
{
    static double get()
    {
        return DBL_MAX;
    }
};

template<CpuType cpu>
struct MaxVal<float, cpu>
{
    static float get()
    {
        return FLT_MAX;
    }
};

template<typename T, CpuType cpu>
struct MinVal
{
    static T get()
    {
        return 0;
    }
};

template<CpuType cpu>
struct MinVal<int, cpu>
{
    static int get()
    {
        return INT_MIN;
    }
};

template<CpuType cpu>
struct MinVal<double, cpu>
{
    static double get()
    {
        return DBL_MIN;
    }
};

template<CpuType cpu>
struct MinVal<float, cpu>
{
    static float get()
    {
        return FLT_MIN;
    }
};

template<typename T, CpuType cpu>
struct EpsilonVal
{
    static T get()
    {
        return 0;
    }
};

template<CpuType cpu>
struct EpsilonVal<double, cpu>
{
    static double get()
    {
        return DBL_EPSILON;
    }
};

template<CpuType cpu>
struct EpsilonVal<float, cpu>
{
    static float get()
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

}
}
}

#endif
