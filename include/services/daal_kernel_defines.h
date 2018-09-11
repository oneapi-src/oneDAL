/* file: daal_kernel_defines.h */
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
//  Common definitions.
//--
*/

#ifndef __DAAL_KERNEL_DEFINES_H__
#define __DAAL_KERNEL_DEFINES_H__

/** \file daal_kernel_defines.h */
/**
 * @ingroup services
 * @{
 */
#define DAAL_KERNELS_ALL

#ifdef DAAL_KERNELS_ALL
#define DAAL_KERNEL_SSSE3
#define DAAL_KERNEL_SSE42
#define DAAL_KERNEL_AVX
#define DAAL_KERNEL_AVX2
#define DAAL_KERNEL_AVX512_mic
#define DAAL_KERNEL_AVX512
#endif

#define DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, cpuType, ...) ContainerTemplate<__VA_ARGS__, cpuType>
#define DAAL_KERNEL_CONTAINER_CASE(ContainerTemplate, cpuType,...)\
    case cpuType: _cntr = (new DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, cpuType, __VA_ARGS__)(daalEnv)); break;

#ifdef DAAL_KERNEL_SSSE3
#define DAAL_KERNEL_SSSE3_ONLY(something) , something
#define DAAL_KERNEL_SSSE3_ONLY_CODE(...) __VA_ARGS__
#define DAAL_KERNEL_SSSE3_CONTAINER(ContainerTemplate, ...) , DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, ssse3, __VA_ARGS__)
#define DAAL_KERNEL_SSSE3_CONTAINER_CASE(ContainerTemplate, ...) DAAL_KERNEL_CONTAINER_CASE(ContainerTemplate, ssse3, __VA_ARGS__)
#else
#define DAAL_KERNEL_SSSE3_ONLY(something)
#define DAAL_KERNEL_SSSE3_ONLY_CODE(...)
#define DAAL_KERNEL_SSSE3_CONTAINER(ContainerTemplate, ...)
#define DAAL_KERNEL_SSSE3_CONTAINER_CASE(ContainerTemplate, ...)
#endif

#ifdef DAAL_KERNEL_SSE42
#define DAAL_KERNEL_SSE42_ONLY(something) , something
#define DAAL_KERNEL_SSE42_ONLY_CODE(...) __VA_ARGS__
#define DAAL_KERNEL_SSE42_CONTAINER(ContainerTemplate, ...) , DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, sse42, __VA_ARGS__)
#define DAAL_KERNEL_SSE42_CONTAINER_CASE(ContainerTemplate, ...) DAAL_KERNEL_CONTAINER_CASE(ContainerTemplate, sse42, __VA_ARGS__)
#else
#define DAAL_KERNEL_SSE42_ONLY(something)
#define DAAL_KERNEL_SSE42_ONLY_CODE(...)
#define DAAL_KERNEL_SSE42_CONTAINER(ContainerTemplate, ...)
#define DAAL_KERNEL_SSE42_CONTAINER_CASE(ContainerTemplate, ...)
#endif

#ifdef DAAL_KERNEL_AVX
#define DAAL_KERNEL_AVX_ONLY(something) , something
#define DAAL_KERNEL_AVX_ONLY_CODE(...) __VA_ARGS__
#define DAAL_KERNEL_AVX_CONTAINER(ContainerTemplate, ...) , DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, avx, __VA_ARGS__)
#define DAAL_KERNEL_AVX_CONTAINER_CASE(ContainerTemplate, ...) DAAL_KERNEL_CONTAINER_CASE(ContainerTemplate, avx, __VA_ARGS__)
#else
#define DAAL_KERNEL_AVX_ONLY(something)
#define DAAL_KERNEL_AVX_ONLY_CODE(...)
#define DAAL_KERNEL_AVX_CONTAINER(ContainerTemplate, ...)
#define DAAL_KERNEL_AVX_CONTAINER_CASE(ContainerTemplate, ...)
#endif

#ifdef DAAL_KERNEL_AVX2
#define DAAL_KERNEL_AVX2_ONLY(something) , something
#define DAAL_KERNEL_AVX2_ONLY_CODE(...) __VA_ARGS__
#define DAAL_KERNEL_AVX2_CONTAINER(ContainerTemplate, ...) , DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, avx2, __VA_ARGS__)
#define DAAL_KERNEL_AVX2_CONTAINER_CASE(ContainerTemplate, ...) DAAL_KERNEL_CONTAINER_CASE(ContainerTemplate, avx2, __VA_ARGS__)
#else
#define DAAL_KERNEL_AVX2_ONLY(something)
#define DAAL_KERNEL_AVX2_ONLY_CODE(...)
#define DAAL_KERNEL_AVX2_CONTAINER(ContainerTemplate, ...)
#define DAAL_KERNEL_AVX2_CONTAINER_CASE(ContainerTemplate, ...)
#endif

#ifdef DAAL_KERNEL_AVX512_mic
#define DAAL_KERNEL_AVX512_mic_ONLY(something) , something
#define DAAL_KERNEL_AVX512_mic_ONLY_CODE(...) __VA_ARGS__
#define DAAL_KERNEL_AVX512_mic_CONTAINER(ContainerTemplate, ...) , DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, avx512_mic, __VA_ARGS__)
#define DAAL_KERNEL_AVX512_mic_CONTAINER_CASE(ContainerTemplate, ...) DAAL_KERNEL_CONTAINER_CASE(ContainerTemplate, avx512_mic, __VA_ARGS__)
#else
#define DAAL_KERNEL_AVX512_mic_ONLY(something)
#define DAAL_KERNEL_AVX512_mic_ONLY_CODE(...)
#define DAAL_KERNEL_AVX512_mic_CONTAINER(ContainerTemplate, ...)
#define DAAL_KERNEL_AVX512_mic_CONTAINER_CASE(ContainerTemplate, ...)
#endif

#ifdef DAAL_KERNEL_AVX512
#define DAAL_KERNEL_AVX512_ONLY(something) , something
#define DAAL_KERNEL_AVX512_ONLY_CODE(...) __VA_ARGS__
#define DAAL_KERNEL_AVX512_CONTAINER(ContainerTemplate, ...) , DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, avx512, __VA_ARGS__)
#define DAAL_KERNEL_AVX512_CONTAINER_CASE(ContainerTemplate, ...) DAAL_KERNEL_CONTAINER_CASE(ContainerTemplate, avx512, __VA_ARGS__)
#else
#define DAAL_KERNEL_AVX512_ONLY(something)
#define DAAL_KERNEL_AVX512_ONLY_CODE(...)
#define DAAL_KERNEL_AVX512_CONTAINER(ContainerTemplate, ...)
#define DAAL_KERNEL_AVX512_CONTAINER_CASE(ContainerTemplate, ...)
#endif
/** @} */

#endif
