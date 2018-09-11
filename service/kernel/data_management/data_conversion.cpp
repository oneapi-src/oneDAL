/** file data_management_utils.cpp */
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

template<typename T1, typename T2>
static void vectorConvertFunc(size_t n, const void *src, void *dst)
{
    typedef void (*funcType)(size_t n, const void *src, void *dst);
    static funcType ptr = 0;

    if(!ptr)
    {
        int cpuid = (int)daal::services::Environment::getInstance()->getCpuId();

        switch(cpuid)
        {
#ifdef DAAL_KERNEL_AVX512
            case avx512    : DAAL_KERNEL_AVX512_ONLY_CODE    (ptr = vectorConvertFuncCpu<T1,T2,avx512    >); break;
#endif
#ifdef DAAL_KERNEL_AVX512_mic
            case avx512_mic: DAAL_KERNEL_AVX512_mic_ONLY_CODE(ptr = vectorConvertFuncCpu<T1,T2,avx512_mic>); break;
#endif
#ifdef DAAL_KERNEL_AVX2
            case avx2      : DAAL_KERNEL_AVX2_ONLY_CODE      (ptr = vectorConvertFuncCpu<T1,T2,avx2      >); break;
#endif
#ifdef DAAL_KERNEL_AVX
            case avx       : DAAL_KERNEL_AVX_ONLY_CODE       (ptr = vectorConvertFuncCpu<T1,T2,avx       >); break;
#endif
#ifdef DAAL_KERNEL_SSE42
            case sse42     : DAAL_KERNEL_SSE42_ONLY_CODE     (ptr = vectorConvertFuncCpu<T1,T2,sse42     >); break;
#endif
#ifdef DAAL_KERNEL_SSSE3
            case ssse3     : DAAL_KERNEL_SSSE3_ONLY_CODE     (ptr = vectorConvertFuncCpu<T1,T2,ssse3     >); break;
#endif
            default        : ptr = vectorConvertFuncCpu<T1,T2,sse2      >; break;
        };
    }

    ptr(n,src,dst);
}

template<typename T1, typename T2>
static void vectorStrideConvertFunc(size_t n, const void *src, size_t srcByteStride, void *dst, size_t dstByteStride)
{
    typedef void (*funcType)(size_t n, const void *src, size_t srcByteStride, void *dst, size_t dstByteStride);
    static funcType ptr = 0;

    if(!ptr)
    {
        int cpuid = (int)daal::services::Environment::getInstance()->getCpuId();

        switch(cpuid)
        {
#ifdef DAAL_KERNEL_AVX512
            case avx512    : DAAL_KERNEL_AVX512_ONLY_CODE    (ptr = vectorStrideConvertFuncCpu<T1,T2,avx512    >); break;
#endif
#ifdef DAAL_KERNEL_AVX512_mic
            case avx512_mic: DAAL_KERNEL_AVX512_mic_ONLY_CODE(ptr = vectorStrideConvertFuncCpu<T1,T2,avx512_mic>); break;
#endif
#ifdef DAAL_KERNEL_AVX2
            case avx2      : DAAL_KERNEL_AVX2_ONLY_CODE      (ptr = vectorStrideConvertFuncCpu<T1,T2,avx2      >); break;
#endif
#ifdef DAAL_KERNEL_AVX
            case avx       : DAAL_KERNEL_AVX_ONLY_CODE       (ptr = vectorStrideConvertFuncCpu<T1,T2,avx       >); break;
#endif
#ifdef DAAL_KERNEL_SSE42
            case sse42     : DAAL_KERNEL_SSE42_ONLY_CODE     (ptr = vectorStrideConvertFuncCpu<T1,T2,sse42     >); break;
#endif
#ifdef DAAL_KERNEL_SSSE3
            case ssse3     : DAAL_KERNEL_SSSE3_ONLY_CODE     (ptr = vectorStrideConvertFuncCpu<T1,T2,ssse3     >); break;
#endif
            default        : ptr = vectorStrideConvertFuncCpu<T1,T2,sse2      >; break;
        };
    }

    ptr(n, src, srcByteStride, dst, dstByteStride);
}

#undef  DAAL_TABLE_UP_ENTRY
#define DAAL_TABLE_UP_ENTRY(F,T) {F<T, float>, F<T, double>, F<T, int> }

#undef  DAAL_TABLE_DOWN_ENTRY
#define DAAL_TABLE_DOWN_ENTRY(F,T) {F<float, T>, F<double, T>, F<int, T> }

#undef  DAAL_CONVERT_UP_TABLE
#define DAAL_CONVERT_UP_TABLE(F) {              \
        DAAL_TABLE_UP_ENTRY(F,float),               \
        DAAL_TABLE_UP_ENTRY(F,double),              \
        DAAL_TABLE_UP_ENTRY(F,int),                 \
        DAAL_TABLE_UP_ENTRY(F,unsigned int),        \
        DAAL_TABLE_UP_ENTRY(F,DAAL_INT64),          \
        DAAL_TABLE_UP_ENTRY(F,DAAL_UINT64),         \
        DAAL_TABLE_UP_ENTRY(F,char),                \
        DAAL_TABLE_UP_ENTRY(F,unsigned char),       \
        DAAL_TABLE_UP_ENTRY(F,short),               \
        DAAL_TABLE_UP_ENTRY(F,unsigned short),      \
    }

#undef  DAAL_CONVERT_DOWN_TABLE
#define DAAL_CONVERT_DOWN_TABLE(F) {           \
        DAAL_TABLE_DOWN_ENTRY(F,float),            \
        DAAL_TABLE_DOWN_ENTRY(F,double),           \
        DAAL_TABLE_DOWN_ENTRY(F,int),              \
        DAAL_TABLE_DOWN_ENTRY(F,unsigned int),     \
        DAAL_TABLE_DOWN_ENTRY(F,DAAL_INT64),       \
        DAAL_TABLE_DOWN_ENTRY(F,DAAL_UINT64),      \
        DAAL_TABLE_DOWN_ENTRY(F,char),             \
        DAAL_TABLE_DOWN_ENTRY(F,unsigned char),    \
        DAAL_TABLE_DOWN_ENTRY(F,short),            \
        DAAL_TABLE_DOWN_ENTRY(F,unsigned short),   \
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
