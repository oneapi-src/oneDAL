/** file data_management_utils.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation for data conversions dispatcher
//--
*/

#include "daal_kernel_defines.h"
#include "data_conversion_cpu.h"
#include "data_management/data/internal/conversion.h"

namespace daal
{
namespace data_management
{
namespace internal
{

#define DISPATCH_FUNCTION(funcName, cpuId, ...)                                                                                             \
    switch (cpuId) {                                                                                                                        \
    DAAL_KERNEL_SSSE3_ONLY_CODE(case daal::CpuType::ssse3: ptr = funcName<__VA_ARGS__, daal::CpuType::ssse3>; break;)                       \
    DAAL_KERNEL_SSE42_ONLY_CODE(case daal::CpuType::sse42: ptr = funcName<__VA_ARGS__, daal::CpuType::sse42>; break;)                       \
    DAAL_KERNEL_AVX_ONLY_CODE(case daal::CpuType::avx:   ptr = funcName<__VA_ARGS__, daal::CpuType::avx>; break;)                           \
    DAAL_KERNEL_AVX2_ONLY_CODE(case daal::CpuType::avx2:  ptr = funcName<__VA_ARGS__, daal::CpuType::avx2>; break;)                         \
    DAAL_KERNEL_AVX512_ONLY_CODE(case daal::CpuType::avx512:  ptr = funcName<__VA_ARGS__, daal::CpuType::avx512>; break;)                   \
    DAAL_KERNEL_AVX512_MIC_ONLY_CODE(case daal::CpuType::avx512_mic_e1: ptr = funcName<__VA_ARGS__, daal::CpuType::avx512_mic_e1>; break;)  \
    default: ptr = funcName<__VA_ARGS__, daal::CpuType::sse2>; break;                                                                       \
    }

template<typename T1, typename T2>
static void vectorConvertFunc(size_t n, const void *src, void *dst)
{
    typedef void (*funcType)(size_t n, const void * src, void * dst);
    static funcType ptr = 0;

    if (!ptr)
    {
        const daal::CpuType cpuid = static_cast<daal::CpuType>(daal::services::Environment::getInstance()->getCpuId());
        DISPATCH_FUNCTION(vectorConvertFuncCpu, cpuid, T1, T2);
    }

    ptr(n, src, dst);
}

template <typename T1, typename T2>
static void vectorStrideConvertFunc(size_t n, const void * src, size_t srcByteStride, void * dst, size_t dstByteStride)
{
    typedef void (*funcType)(size_t n, const void * src, size_t srcByteStride, void * dst, size_t dstByteStride);
    static funcType ptr = 0;

    if (!ptr)
    {
        const daal::CpuType cpuid = static_cast<daal::CpuType>(daal::services::Environment::getInstance()->getCpuId());
        DISPATCH_FUNCTION(vectorStrideConvertFuncCpu, cpuid, T1, T2);
    }

    ptr(n, src, srcByteStride, dst, dstByteStride);
}

template <typename T>
DAAL_EXPORT void vectorAssignValueToArray(T * const dataPtr, const size_t n, const T value)
{
    typedef void (*funcType)(void * const, const size_t, const void * const);
    static funcType ptr = 0;

    if (!ptr)
    {
        const daal::CpuType cpuid = static_cast<daal::CpuType>(daal::services::Environment::getInstance()->getCpuId());
        DISPATCH_FUNCTION(vectorAssignValueToArrayCpu, cpuid, T);
    }

    ptr(dataPtr, n, &value);
}

#define DAAL_REGISTER_VECTOR_ASSIGN(Type) \
    template DAAL_EXPORT void vectorAssignValueToArray<Type>(Type * const ptr, const size_t n, const Type value);
DAAL_REGISTER_WITH_HOMOGEN_NT_TYPES(DAAL_REGISTER_VECTOR_ASSIGN)

#undef DAAL_TABLE_UP_ENTRY
#define DAAL_TABLE_UP_ENTRY(F, T)            \
    {                                        \
        F<T, float>, F<T, double>, F<T, int> \
    }

#undef DAAL_TABLE_DOWN_ENTRY
#define DAAL_TABLE_DOWN_ENTRY(F, T)          \
    {                                        \
        F<float, T>, F<double, T>, F<int, T> \
    }

#undef DAAL_CONVERT_UP_TABLE
#define DAAL_CONVERT_UP_TABLE(F)                                                                                                          \
    {                                                                                                                                     \
        DAAL_TABLE_UP_ENTRY(F, float), DAAL_TABLE_UP_ENTRY(F, double), DAAL_TABLE_UP_ENTRY(F, int), DAAL_TABLE_UP_ENTRY(F, unsigned int), \
            DAAL_TABLE_UP_ENTRY(F, DAAL_INT64), DAAL_TABLE_UP_ENTRY(F, DAAL_UINT64), DAAL_TABLE_UP_ENTRY(F, char),                        \
            DAAL_TABLE_UP_ENTRY(F, unsigned char), DAAL_TABLE_UP_ENTRY(F, short), DAAL_TABLE_UP_ENTRY(F, unsigned short),                 \
    }

#undef DAAL_CONVERT_DOWN_TABLE
#define DAAL_CONVERT_DOWN_TABLE(F)                                                                                                                \
    {                                                                                                                                             \
        DAAL_TABLE_DOWN_ENTRY(F, float), DAAL_TABLE_DOWN_ENTRY(F, double), DAAL_TABLE_DOWN_ENTRY(F, int), DAAL_TABLE_DOWN_ENTRY(F, unsigned int), \
            DAAL_TABLE_DOWN_ENTRY(F, DAAL_INT64), DAAL_TABLE_DOWN_ENTRY(F, DAAL_UINT64), DAAL_TABLE_DOWN_ENTRY(F, char),                          \
            DAAL_TABLE_DOWN_ENTRY(F, unsigned char), DAAL_TABLE_DOWN_ENTRY(F, short), DAAL_TABLE_DOWN_ENTRY(F, unsigned short),                   \
    }

DAAL_EXPORT vectorConvertFuncType getVectorUpCast(int idx1, int idx2)
{
    static vectorConvertFuncType table[][3] = DAAL_CONVERT_UP_TABLE(vectorConvertFunc);
    return table[idx1][idx2];
}

DAAL_EXPORT vectorConvertFuncType getVectorDownCast(int idx1, int idx2)
{
    static vectorConvertFuncType table[][3] = DAAL_CONVERT_DOWN_TABLE(vectorConvertFunc);
    return table[idx1][idx2];
}

DAAL_EXPORT vectorStrideConvertFuncType getVectorStrideUpCast(int idx1, int idx2)
{
    static vectorStrideConvertFuncType table[][3] = DAAL_CONVERT_UP_TABLE(vectorStrideConvertFunc);
    return table[idx1][idx2];
}

DAAL_EXPORT vectorStrideConvertFuncType getVectorStrideDownCast(int idx1, int idx2)
{
    static vectorStrideConvertFuncType table[][3] = DAAL_CONVERT_DOWN_TABLE(vectorStrideConvertFunc);
    return table[idx1][idx2];
}

} // namespace internal
namespace data_feature_utils
{
DAAL_EXPORT internal::vectorConvertFuncType getVectorUpCast(int idx1, int idx2)
{
    return internal::getVectorUpCast(idx1, idx2);
}

DAAL_EXPORT internal::vectorConvertFuncType getVectorDownCast(int idx1, int idx2)
{
    return internal::getVectorDownCast(idx1, idx2);
}

DAAL_EXPORT internal::vectorStrideConvertFuncType getVectorStrideUpCast(int idx1, int idx2)
{
    return internal::getVectorStrideUpCast(idx1, idx2);
}

DAAL_EXPORT internal::vectorStrideConvertFuncType getVectorStrideDownCast(int idx1, int idx2)
{
    return internal::getVectorStrideDownCast(idx1, idx2);
}

} // namespace data_feature_utils

} // namespace data_management
} // namespace daal
