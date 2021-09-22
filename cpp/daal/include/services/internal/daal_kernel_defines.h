/* file: daal_kernel_defines.h */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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
//  Common definitions.
//--
*/

#ifndef __DAAL_KERNEL_DEFINES_H__
#define __DAAL_KERNEL_DEFINES_H__

#include "services/env_detect.h"

/** \file daal_kernel_defines.h */
/**
 * @ingroup services
 * @{
 */
#define DAAL_KERNEL_SSSE3
#define DAAL_KERNEL_SSE42
#define DAAL_KERNEL_AVX
#define DAAL_KERNEL_AVX2
#define DAAL_KERNEL_AVX512_MIC
#define DAAL_KERNEL_AVX512

#define DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, cpuType, ...) ContainerTemplate<__VA_ARGS__, cpuType>
#define DAAL_KERNEL_CONTAINER_CASE(ContainerTemplate, cpuType, ...)                              \
case cpuType:                                                                                    \
    _cntr = (new DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, cpuType, __VA_ARGS__)(daalEnv)); \
    break;
#define DAAL_KERNEL_CONTAINER_CASE_SYCL(ContainerTemplate, cpuType, ...)                        \
case cpuType:                                                                                   \
{                                                                                               \
    using contTemplType = DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, cpuType, __VA_ARGS__); \
    static volatile daal::services::internal::GpuSupportRegistrar<contTemplType> registrar;     \
    _cntr = (new contTemplType(daalEnv));                                                       \
    break;                                                                                      \
}

#undef DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID
#define DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID                daal::sse2
#define DAAL_KERNEL_SSE2_ONLY(something)                        , something
#define DAAL_KERNEL_SSE2_ONLY_CODE(...)                         __VA_ARGS__
#define DAAL_KERNEL_SSE2_CONTAINER(ContainerTemplate, ...)      , DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, sse2, __VA_ARGS__)
#define DAAL_KERNEL_SSE2_CONTAINER1(ContainerTemplate, ...)     extern template class DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, sse2, __VA_ARGS__);
#define DAAL_KERNEL_SSE2_CONTAINER_CASE(ContainerTemplate, ...) DAAL_KERNEL_CONTAINER_CASE(ContainerTemplate, sse2, __VA_ARGS__)

#if defined(DAAL_KERNEL_SSSE3)
    #undef DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID
    #define DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID            daal::ssse3
    #define DAAL_KERNEL_SSSE3_ONLY(something)                   , something
    #define DAAL_KERNEL_SSSE3_ONLY_CODE(...)                    __VA_ARGS__
    #define DAAL_KERNEL_SSSE3_CONTAINER(ContainerTemplate, ...) , DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, ssse3, __VA_ARGS__)
    #define DAAL_KERNEL_SSSE3_CONTAINER1(ContainerTemplate, ...) \
        extern template class DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, ssse3, __VA_ARGS__);
    #define DAAL_KERNEL_SSSE3_CONTAINER_CASE(ContainerTemplate, ...)      DAAL_KERNEL_CONTAINER_CASE(ContainerTemplate, ssse3, __VA_ARGS__)
    #define DAAL_KERNEL_SSSE3_CONTAINER_CASE_SYCL(ContainerTemplate, ...) DAAL_KERNEL_CONTAINER_CASE_SYCL(ContainerTemplate, ssse3, __VA_ARGS__)
#else
    #define DAAL_KERNEL_SSSE3_ONLY(something)
    #define DAAL_KERNEL_SSSE3_ONLY_CODE(...)
    #define DAAL_KERNEL_SSSE3_CONTAINER(ContainerTemplate, ...)
    #define DAAL_KERNEL_SSSE3_CONTAINER1(ContainerTemplate, ...)
    #define DAAL_KERNEL_SSSE3_CONTAINER_CASE(ContainerTemplate, ...)
    #define DAAL_KERNEL_SSSE3_CONTAINER_CASE_SYCL(ContainerTemplate, ...)
#endif

#if defined(DAAL_KERNEL_SSE42)
    #undef DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID
    #define DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID            daal::sse42
    #define DAAL_KERNEL_SSE42_ONLY(something)                   , something
    #define DAAL_KERNEL_SSE42_ONLY_CODE(...)                    __VA_ARGS__
    #define DAAL_KERNEL_SSE42_CONTAINER(ContainerTemplate, ...) , DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, sse42, __VA_ARGS__)
    #define DAAL_KERNEL_SSE42_CONTAINER1(ContainerTemplate, ...) \
        extern template class DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, sse42, __VA_ARGS__);
    #define DAAL_KERNEL_SSE42_CONTAINER_CASE(ContainerTemplate, ...)      DAAL_KERNEL_CONTAINER_CASE(ContainerTemplate, sse42, __VA_ARGS__)
    #define DAAL_KERNEL_SSE42_CONTAINER_CASE_SYCL(ContainerTemplate, ...) DAAL_KERNEL_CONTAINER_CASE_SYCL(ContainerTemplate, sse42, __VA_ARGS__)
#else
    #define DAAL_KERNEL_SSE42_ONLY(something)
    #define DAAL_KERNEL_SSE42_ONLY_CODE(...)
    #define DAAL_KERNEL_SSE42_CONTAINER(ContainerTemplate, ...)
    #define DAAL_KERNEL_SSE42_CONTAINER1(ContainerTemplate, ...)
    #define DAAL_KERNEL_SSE42_CONTAINER_CASE(ContainerTemplate, ...)
    #define DAAL_KERNEL_SSE42_CONTAINER_CASE_SYCL(ContainerTemplate, ...)
#endif

#if defined(DAAL_KERNEL_AVX)
    #undef DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID
    #define DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID                    daal::avx
    #define DAAL_KERNEL_AVX_ONLY(something)                             , something
    #define DAAL_KERNEL_AVX_ONLY_CODE(...)                              __VA_ARGS__
    #define DAAL_KERNEL_AVX_CONTAINER(ContainerTemplate, ...)           , DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, avx, __VA_ARGS__)
    #define DAAL_KERNEL_AVX_CONTAINER1(ContainerTemplate, ...)          extern template class DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, avx, __VA_ARGS__);
    #define DAAL_KERNEL_AVX_CONTAINER_CASE(ContainerTemplate, ...)      DAAL_KERNEL_CONTAINER_CASE(ContainerTemplate, avx, __VA_ARGS__)
    #define DAAL_KERNEL_AVX_CONTAINER_CASE_SYCL(ContainerTemplate, ...) DAAL_KERNEL_CONTAINER_CASE_SYCL(ContainerTemplate, avx, __VA_ARGS__)
#else
    #define DAAL_KERNEL_AVX_ONLY(something)
    #define DAAL_KERNEL_AVX_ONLY_CODE(...)
    #define DAAL_KERNEL_AVX_CONTAINER(ContainerTemplate, ...)
    #define DAAL_KERNEL_AVX_CONTAINER1(ContainerTemplate, ...)
    #define DAAL_KERNEL_AVX_CONTAINER_CASE(ContainerTemplate, ...)
    #define DAAL_KERNEL_AVX_CONTAINER_CASE_SYCL(ContainerTemplate, ...)
#endif

#if defined(DAAL_KERNEL_AVX2)
    #undef DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID
    #define DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID           daal::avx2
    #define DAAL_KERNEL_AVX2_ONLY(something)                   , something
    #define DAAL_KERNEL_AVX2_ONLY_CODE(...)                    __VA_ARGS__
    #define DAAL_KERNEL_AVX2_CONTAINER(ContainerTemplate, ...) , DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, avx2, __VA_ARGS__)
    #define DAAL_KERNEL_AVX2_CONTAINER1(ContainerTemplate, ...) \
        extern template class DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, avx2, __VA_ARGS__);
    #define DAAL_KERNEL_AVX2_CONTAINER_CASE(ContainerTemplate, ...)      DAAL_KERNEL_CONTAINER_CASE(ContainerTemplate, avx2, __VA_ARGS__)
    #define DAAL_KERNEL_AVX2_CONTAINER_CASE_SYCL(ContainerTemplate, ...) DAAL_KERNEL_CONTAINER_CASE_SYCL(ContainerTemplate, avx2, __VA_ARGS__)
#else
    #define DAAL_KERNEL_AVX2_ONLY(something)
    #define DAAL_KERNEL_AVX2_ONLY_CODE(...)
    #define DAAL_KERNEL_AVX2_CONTAINER(ContainerTemplate, ...)
    #define DAAL_KERNEL_AVX2_CONTAINER1(ContainerTemplate, ...)
    #define DAAL_KERNEL_AVX2_CONTAINER_CASE(ContainerTemplate, ...)
    #define DAAL_KERNEL_AVX2_CONTAINER_CASE_SYCL(ContainerTemplate, ...)
#endif

#if defined(DAAL_KERNEL_AVX512_MIC)
    #undef DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID
    #define DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID                 daal::avx512_mic
    #define DAAL_KERNEL_AVX512_MIC_ONLY(something)                   , something
    #define DAAL_KERNEL_AVX512_MIC_ONLY_CODE(...)                    __VA_ARGS__
    #define DAAL_KERNEL_AVX512_MIC_CONTAINER(ContainerTemplate, ...) , DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, avx512_mic, __VA_ARGS__)
    #define DAAL_KERNEL_AVX512_MIC_CONTAINER1(ContainerTemplate, ...) \
        extern template class DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, avx512_mic, __VA_ARGS__);
    #define DAAL_KERNEL_AVX512_MIC_CONTAINER_CASE(ContainerTemplate, ...) DAAL_KERNEL_CONTAINER_CASE(ContainerTemplate, avx512_mic, __VA_ARGS__)
    #define DAAL_KERNEL_AVX512_MIC_CONTAINER_CASE_SYCL(ContainerTemplate, ...) \
        DAAL_KERNEL_CONTAINER_CASE_SYCL(ContainerTemplate, avx512_mic, __VA_ARGS__)
#else
    #define DAAL_KERNEL_AVX512_MIC_ONLY(something)
    #define DAAL_KERNEL_AVX512_MIC_ONLY_CODE(...)
    #define DAAL_KERNEL_AVX512_MIC_CONTAINER(ContainerTemplate, ...)
    #define DAAL_KERNEL_AVX512_MIC_CONTAINER1(ContainerTemplate, ...)
    #define DAAL_KERNEL_AVX512_MIC_CONTAINER_CASE(ContainerTemplate, ...)
    #define DAAL_KERNEL_AVX512_MIC_CONTAINER_CASE_SYCL(ContainerTemplate, ...)
#endif

#if defined(DAAL_KERNEL_AVX512)
    #undef DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID
    #define DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID             daal::avx512
    #define DAAL_KERNEL_AVX512_ONLY(something)                   , something
    #define DAAL_KERNEL_AVX512_ONLY_CODE(...)                    __VA_ARGS__
    #define DAAL_KERNEL_AVX512_CONTAINER(ContainerTemplate, ...) , DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, avx512, __VA_ARGS__)
    #define DAAL_KERNEL_AVX512_CONTAINER1(ContainerTemplate, ...) \
        extern template class DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, avx512, __VA_ARGS__);
    #define DAAL_KERNEL_AVX512_CONTAINER_CASE(ContainerTemplate, ...)      DAAL_KERNEL_CONTAINER_CASE(ContainerTemplate, avx512, __VA_ARGS__)
    #define DAAL_KERNEL_AVX512_CONTAINER_CASE_SYCL(ContainerTemplate, ...) DAAL_KERNEL_CONTAINER_CASE_SYCL(ContainerTemplate, avx512, __VA_ARGS__)
#else
    #define DAAL_KERNEL_AVX512_ONLY(something)
    #define DAAL_KERNEL_AVX512_ONLY_CODE(...)
    #define DAAL_KERNEL_AVX512_CONTAINER(ContainerTemplate, ...)
    #define DAAL_KERNEL_AVX512_CONTAINER1(ContainerTemplate, ...)
    #define DAAL_KERNEL_AVX512_CONTAINER_CASE(ContainerTemplate, ...)
    #define DAAL_KERNEL_AVX512_CONTAINER_CASE_SYCL(ContainerTemplate, ...)
#endif

#define DAAL_EXPAND(...) __VA_ARGS__
/** @} */

#endif
