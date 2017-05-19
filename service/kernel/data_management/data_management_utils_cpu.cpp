/** file data_management_utils_cpu.cpp */
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

#include "data_utils.h"
#include "service_data_utils.h"

namespace daal
{
namespace data_feature_utils
{
namespace internal
{

template<typename T1, typename T2, CpuType cpu>
void vectorConvertFuncCpu(size_t n, void *src, void *dst)
{
    for(size_t i = 0; i < n; i++)
    {
        ((T2 *)dst)[i] = static_cast<T2>(((T1 *)src)[i]);
    }
}

template<typename T1, typename T2, CpuType cpu>
void vectorStrideConvertFuncCpu(size_t n, void *src, size_t srcByteStride, void *dst, size_t dstByteStride)
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

DAAL_CONVERT_UP_FUNCS(vectorConvertFuncCpu,(size_t n, void *src, void *dst))
DAAL_CONVERT_DOWN_FUNCS(vectorConvertFuncCpu,(size_t n, void *src, void *dst))

DAAL_CONVERT_UP_FUNCS(vectorStrideConvertFuncCpu,(size_t n, void *src, size_t srcByteStride, void *dst, size_t dstByteStride))
DAAL_CONVERT_DOWN_FUNCS(vectorStrideConvertFuncCpu,(size_t n, void *src, size_t srcByteStride, void *dst, size_t dstByteStride))

}
}
}
