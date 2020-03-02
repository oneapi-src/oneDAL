/* file: service_data_utils.h */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

#include "service/kernel/service_defines.h"

namespace daal
{
namespace services
{
namespace internal
{
template <typename T>
struct MaxVal
{
    DAAL_FORCEINLINE static T get() { return 0; }
};

template <>
struct MaxVal<int>
{
    DAAL_FORCEINLINE static int get() { return INT_MAX; }
};

template <>
struct MaxVal<long long>
{
    DAAL_FORCEINLINE static long long get() { return LLONG_MAX; }
};

template <>
struct MaxVal<double>
{
    DAAL_FORCEINLINE static double get() { return DBL_MAX; }
};

template <>
struct MaxVal<uint32_t>
{
    DAAL_FORCEINLINE static uint32_t get() { return UINT32_MAX; }
};

template <>
struct MaxVal<float>
{
    DAAL_FORCEINLINE static float get() { return FLT_MAX; }
};

template <typename T>
struct MinVal
{
    DAAL_FORCEINLINE static T get() { return 0; }
};

template <>
struct MinVal<int>
{
    DAAL_FORCEINLINE static int get() { return INT_MIN; }
};

template <>
struct MinVal<double>
{
    DAAL_FORCEINLINE static double get() { return DBL_MIN; }
};

template <>
struct MinVal<float>
{
    DAAL_FORCEINLINE static float get() { return FLT_MIN; }
};

template <typename T>
struct EpsilonVal
{
    DAAL_FORCEINLINE static T get() { return 0; }
};

template <>
struct EpsilonVal<double>
{
    DAAL_FORCEINLINE static double get() { return DBL_EPSILON; }
};

template <>
struct EpsilonVal<float>
{
    DAAL_FORCEINLINE static float get() { return FLT_EPSILON; }
};

template <typename T, CpuType cpu>
struct SignBit;

template <CpuType cpu>
struct SignBit<float, cpu>
{
    static int get(float val) { return ((_daal_sp_union_t *)&val)->bits.sign; }
};

template <CpuType cpu>
struct SignBit<double, cpu>
{
    static int get(double val) { return ((_daal_dp_union_t *)&val)->bits.sign; }
};

template <typename T1, typename T2, CpuType cpu>
void vectorConvertFuncCpu(size_t n, void * src, void * dst);

template <typename T1, typename T2, CpuType cpu>
void vectorStrideConvertFuncCpu(size_t n, void * src, size_t srcByteStride, void * dst, size_t dstByteStride);

template<CpuType cpu, typename T, size_t size = sizeof(T)>
struct __clz
{
    size_t operator()(T x)
    {
        constexpr size_t bit_size = 8 * size;
        constexpr size_t minus_one = -1;
        constexpr size_t one = 1;
        size_t i;
        for(i = bit_size; (i != minus_one) && !(x & (one << i)); --i);
        return bit_size - i - 1;
    }
};

template<CpuType cpu, typename T = size_t>
size_t greaterOrEqualPowerOf2Proto(T x)
{
    constexpr size_t bit_size = 8 * sizeof(T);
    constexpr size_t one = 1;
    const size_t leading_ones = __clz<cpu, T, sizeof(T)>()(x);
    return one << (bit_size - leading_ones);
}

#if defined(__INTEL_COMPILER) || defined(_MSC_VER)
#define CLZ_BUILTINS
#include <immintrin.h>

template<CpuType cpu, typename T>
struct __clz<cpu, T, 4>
{
    size_t operator()(T x)
    {
        return _lzcnt_u32(x);
    }
};

template<CpuType cpu, typename T>
struct __clz<cpu, T, 8>
{
    size_t operator()(T x)
    {
        return _lzcnt_u64(x);
    }
};

#elif (defined(__GNUC__) || defined(__clang__)) && !CLZ_BUILTINS
#define CLZ_BUILTINS
template<CpuType cpu>
struct __clz<cpu, unsigned int, sizeof(unsigned int)>
{
    size_t operator()(unsigned int x)
    {
        return __builtin_clz(x);
    }
};
template<CpuType cpu>
struct __clz<cpu, unsigned long, sizeof(unsigned long)>
{
    size_t operator()(unsigned long x)
    {
        return __builtin_clzl(x);
    }
};
template<CpuType cpu>
struct __clz<cpu, unsigned long long, sizeof(unsigned long long)>
{
    size_t operator()(unsigned long long x)
    {
        return __builtin_clzll(x);
    }
};
#endif

template<CpuType cpu>
size_t greaterOrEqualPowerOf2(size_t x)
{
    return (!(x & (x - 1))) ? x : greaterOrEqualPowerOf2Proto<cpu>(x); 
}

} // namespace internal
} // namespace services
} // namespace daal

#endif
