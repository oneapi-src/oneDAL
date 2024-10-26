/* file: service_dispatch.h */
/*******************************************************************************
* Copyright 2018 Intel Corporation
* Copyright contributors to the oneDAL project
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
//  Universal dispatcher of abstract function with cpu parameter.
//--
*/

#ifndef __SERVICE_DISPATCH_H__
#define __SERVICE_DISPATCH_H__

#include "services/internal/daal_kernel_defines.h"

#if defined(TARGET_X86_64)
    #define DAAL_DISPATCH_FUNCTION_BY_CPU(func, ...)                                                                    \
        switch (__DAAL_KERNEL_MIN(DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID,                                             \
                                  static_cast<daal::CpuType>(daal::services::Environment::getInstance()->getCpuId())))  \
        {                                                                                                               \
            DAAL_KERNEL_SSE42_ONLY_CODE(case daal::CpuType::sse42 : func(daal::CpuType::sse42, __VA_ARGS__); break;)    \
            DAAL_KERNEL_AVX2_ONLY_CODE(case daal::CpuType::avx2 : func(daal::CpuType::avx2, __VA_ARGS__); break;)       \
            DAAL_KERNEL_AVX512_ONLY_CODE(case daal::CpuType::avx512 : func(daal::CpuType::avx512, __VA_ARGS__); break;) \
            DAAL_EXPAND(default : func(daal::CpuType::sse2, __VA_ARGS__); break;)                                       \
        }

    #define DAAL_DISPATCH_FUNCTION_BY_CPU_SAFE(func, ...)                                                                                  \
        services::Status st;                                                                                                               \
        int cpuid = daal::sse2;                                                                                                            \
        DAAL_SAFE_CPU_CALL(                                                                                                                \
            (cpuid = __DAAL_KERNEL_MIN(DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID, daal::services::Environment::getInstance()->getCpuId())), \
            (cpuid = daal::sse2))                                                                                                          \
        switch (static_cast<daal::CpuType>(cpuid))                                                                                         \
        {                                                                                                                                  \
            DAAL_KERNEL_SSE42_ONLY_CODE(case daal::CpuType::sse42 : st = func(daal::CpuType::sse42, __VA_ARGS__); break;)                  \
            DAAL_KERNEL_AVX2_ONLY_CODE(case daal::CpuType::avx2 : st = func(daal::CpuType::avx2, __VA_ARGS__); break;)                     \
            DAAL_KERNEL_AVX512_ONLY_CODE(case daal::CpuType::avx512 : st = func(daal::CpuType::avx512, __VA_ARGS__); break;)               \
            DAAL_EXPAND(default : st = func(daal::CpuType::sse2, __VA_ARGS__); break;)                                                     \
        }                                                                                                                                  \
        services::throwIfPossible(st);
#elif defined(TARGET_ARM)
    #define DAAL_DISPATCH_FUNCTION_BY_CPU(func, ...)                                                           \
        switch (static_cast<daal::CpuType>(daal::services::Environment::getInstance()->getCpuId()))            \
        {                                                                                                      \
            DAAL_KERNEL_SVE_ONLY_CODE(case daal::CpuType::sve : func(daal::CpuType::sve, __VA_ARGS__); break;) \
        }

    #define DAAL_DISPATCH_FUNCTION_BY_CPU_SAFE(func, ...)                                                           \
        services::Status st;                                                                                        \
        int cpuid = daal::sve;                                                                                      \
        DAAL_SAFE_CPU_CALL((cpuid = daal::services::Environment::getInstance()->getCpuId()), (cpuid = daal::sve))   \
        switch (static_cast<daal::CpuType>(cpuid))                                                                  \
        {                                                                                                           \
            DAAL_KERNEL_SVE_ONLY_CODE(case daal::CpuType::sve : st = func(daal::CpuType::sve, __VA_ARGS__); break;) \
        }                                                                                                           \
        services::throwIfPossible(st);
#elif defined(TARGET_RISCV64)
    #define DAAL_DISPATCH_FUNCTION_BY_CPU(func, ...)                                                              \
        switch (static_cast<daal::CpuType>(daal::services::Environment::getInstance()->getCpuId()))               \
        {                                                                                                         \
            DAAL_KERNEL_RV64_ONLY_CODE(case daal::CpuType::rv64 : func(daal::CpuType::rv64, __VA_ARGS__); break;) \
        }

    #define DAAL_DISPATCH_FUNCTION_BY_CPU_SAFE(func, ...)                                                              \
        services::Status st;                                                                                           \
        int cpuid = daal::rv64;                                                                                        \
        DAAL_SAFE_CPU_CALL((cpuid = daal::services::Environment::getInstance()->getCpuId()), (cpuid = daal::rv64))     \
        switch (static_cast<daal::CpuType>(cpuid))                                                                     \
        {                                                                                                              \
            DAAL_KERNEL_RV64_ONLY_CODE(case daal::CpuType::rv64 : st = func(daal::CpuType::rv64, __VA_ARGS__); break;) \
        }                                                                                                              \
        services::throwIfPossible(st);
#endif
#endif
