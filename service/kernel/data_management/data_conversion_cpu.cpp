/** file data_management_utils_cpu.cpp */
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

#include "data_conversion_cpu.h"
#include "internal/conversion.h"
#include "service_memory.h"

namespace daal
{
namespace data_management
{
namespace internal
{

template<typename T1, typename T2, CpuType cpu>
void vectorConvertFuncCpu(size_t n, const void *src, void *dst)
{
    for(size_t i = 0; i < n; i++)
    {
        ((T2 *)dst)[i] = static_cast<T2>(((T1 *)src)[i]);
    }
}

template<typename T1, typename T2, CpuType cpu>
void vectorStrideConvertFuncCpu(size_t n, const void *src, size_t srcByteStride, void *dst, size_t dstByteStride)
{
    for(size_t i = 0; i < n ; i++)
    {
        *(T2 *)(((char *)dst) + i * dstByteStride) = static_cast<T2>(*(T1 *)(((char *)src) + i * srcByteStride));
    }
}

#undef  DAAL_FUNCS_UP_ENTRY
#define DAAL_FUNCS_UP_ENTRY(F,T,A)      \
template void F<T, float , DAAL_CPU> A; \
template void F<T, double, DAAL_CPU> A; \
template void F<T, int   , DAAL_CPU> A;

#undef  DAAL_FUNCS_DOWN_ENTRY
#define DAAL_FUNCS_DOWN_ENTRY(F,T,A)    \
template void F<float , T, DAAL_CPU> A; \
template void F<double, T, DAAL_CPU> A; \
template void F<int   , T, DAAL_CPU> A;

#undef  DAAL_CONVERT_UP_FUNCS
#define DAAL_CONVERT_UP_FUNCS(F,A)                    \
        DAAL_FUNCS_UP_ENTRY(F,float,A)                \
        DAAL_FUNCS_UP_ENTRY(F,double,A)               \
        DAAL_FUNCS_UP_ENTRY(F,int,A)                  \
        DAAL_FUNCS_UP_ENTRY(F,unsigned int,A)         \
        DAAL_FUNCS_UP_ENTRY(F,DAAL_INT64,A)           \
        DAAL_FUNCS_UP_ENTRY(F,DAAL_UINT64,A)          \
        DAAL_FUNCS_UP_ENTRY(F,char,A)                 \
        DAAL_FUNCS_UP_ENTRY(F,unsigned char,A)        \
        DAAL_FUNCS_UP_ENTRY(F,short,A)                \
        DAAL_FUNCS_UP_ENTRY(F,unsigned short,A)

#undef  DAAL_CONVERT_DOWN_FUNCS
#define DAAL_CONVERT_DOWN_FUNCS(F,A)                 \
        DAAL_FUNCS_DOWN_ENTRY(F,unsigned int,A)      \
        DAAL_FUNCS_DOWN_ENTRY(F,DAAL_INT64,A)        \
        DAAL_FUNCS_DOWN_ENTRY(F,DAAL_UINT64,A)       \
        DAAL_FUNCS_DOWN_ENTRY(F,char,A)              \
        DAAL_FUNCS_DOWN_ENTRY(F,unsigned char,A)     \
        DAAL_FUNCS_DOWN_ENTRY(F,short,A)             \
        DAAL_FUNCS_DOWN_ENTRY(F,unsigned short,A)

DAAL_CONVERT_UP_FUNCS(vectorConvertFuncCpu,(size_t n, const void *src, void *dst))
DAAL_CONVERT_DOWN_FUNCS(vectorConvertFuncCpu,(size_t n, const void *src, void *dst))

DAAL_CONVERT_UP_FUNCS(vectorStrideConvertFuncCpu,(size_t n, const void *src, size_t srcByteStride, void *dst, size_t dstByteStride))
DAAL_CONVERT_DOWN_FUNCS(vectorStrideConvertFuncCpu,(size_t n, const void *src, size_t srcByteStride, void *dst, size_t dstByteStride))

template<typename T, CpuType cpu>
void vectorAssignValueToArrayCpu(void* const ptr, const size_t n, const void* const value)
{
    const T& valueT = *((const T*)value);
    T* const ptrT = (T*)ptr;
    services::internal::service_memset<T, cpu>(ptrT, valueT, n);
}

#define DAAL_REGISTER_VECTOR_ASSIGN_CPU(Type) template void vectorAssignValueToArrayCpu<Type, DAAL_CPU>(void* const ptr, const size_t n, const void* const value);
DAAL_REGISTER_WITH_HOMOGEN_NT_TYPES(DAAL_REGISTER_VECTOR_ASSIGN_CPU)

} // namespace internal
} // namespace data_management
} // namespace daal
