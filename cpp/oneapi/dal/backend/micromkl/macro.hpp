/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#pragma once

#include <daal/include/services/daal_defines.h>

#ifndef __MICROMKL_INCLUDE_GUARD__
#error "This header cannot be included outside of micromkl module"
#endif

#define STRINGIFY(x) #x
#define EXPAND(...)  __VA_ARGS__

#ifdef ONEDAL_REF
#define FUNC_NAME(prefix, name)          name
#define FUNC_NAME_CPU(cpu, prefix, name) name
#else
#define FUNC_NAME(prefix, name)          prefix##_##name
#define FUNC_NAME_CPU(cpu, prefix, name) prefix##_##cpu##_##name
#endif

#define DISPATCH_ID_NAME(cpu) oneapi::dal::backend::cpu_dispatch_##cpu

#define FUNC_CPU_DECL(cpu, prefix, name, argdecl) \
    extern "C" void FUNC_NAME_CPU(cpu, prefix, name) argdecl;

#define DISPATCH_FUNC_DECL(prefix, name, arcdecl) \
    template <typename Cpu>                       \
    ONEDAL_FORCEINLINE void FUNC_NAME(prefix, name) arcdecl;

#define DISPATCH_FUNC_CPU(nominal_cpu, actual_cpu, prefix, name, arcdecl, argcall)           \
    template <>                                                                              \
    ONEDAL_FORCEINLINE void FUNC_NAME(prefix, name)<DISPATCH_ID_NAME(nominal_cpu)> arcdecl { \
        FUNC_NAME_CPU(actual_cpu, prefix, name) argcall;                                     \
    }

#define FUNC_CPU(nominal_cpu, actual_cpu, prefix, name, argdecl, argcall) \
    FUNC_CPU_DECL(nominal_cpu, prefix, name, argdecl)                     \
    DISPATCH_FUNC_CPU(nominal_cpu, actual_cpu, prefix, name, argdecl, argcall)

#if defined(TARGET_X86_64)
#define FUNC_AVX512(...) EXPAND(FUNC_CPU(avx512, avx512, __VA_ARGS__))
#define FUNC_AVX2(...)   EXPAND(FUNC_CPU(avx2, avx2, __VA_ARGS__))
#elif defined(TARGET_ARM)
#define FUNC_A8SVE(...) EXPAND(FUNC_CPU(sve, sve, __VA_ARGS__))
#endif

#ifdef __APPLE__
#define FUNC_SSE42(...) EXPAND(FUNC_CPU(sse42, avx2, __VA_ARGS__))
#define FUNC_SSE2(...)  EXPAND(FUNC_CPU(sse2, avx2, __VA_ARGS__))
#else
#define FUNC_SSE42(...) EXPAND(FUNC_CPU(sse42, sse42, __VA_ARGS__))
#define FUNC_SSE2(...)  EXPAND(FUNC_CPU(sse2, sse2, __VA_ARGS__))
#endif

#if defined(TARGET_X86_64)
#define FUNC(prefix, name, argdecl, argcall)    \
    DISPATCH_FUNC_DECL(prefix, name, argdecl)   \
    FUNC_AVX512(prefix, name, argdecl, argcall) \
    FUNC_AVX2(prefix, name, argdecl, argcall)   \
    FUNC_SSE42(prefix, name, argdecl, argcall)  \
    FUNC_SSE2(prefix, name, argdecl, argcall)
#elif defined(TARGET_ARM)
#define FUNC(prefix, name, argdecl, argcall)  \
    DISPATCH_FUNC_DECL(prefix, name, argdecl) \
    FUNC_A8SVE(prefix, name, argdecl, argcall)
#endif

#ifdef ONEDAL_REF
#define FUNC_DECL(prefix, floatabr, name, argdecl, argcall) \
    FUNC(prefix, floatabr##name##_, argdecl, argcall)

#define FUNC_CALL(prefix, floatabr, name, cargcall) floatabr##name##_<Cpu> cargcall;
#else
#define FUNC_DECL(prefix, floatabr, name, argdecl, argcall) \
    FUNC(prefix, floatabr##name, argdecl, argcall)

#define FUNC_CALL(prefix, floatabr, name, cargcall) prefix##_##floatabr##name<Cpu> cargcall;
#endif

#define INSTANTIATE_CPU(cpu, name, Float, argdecl) \
    template void name<DISPATCH_ID_NAME(cpu), Float> argdecl(Float);

#ifdef ONEDAL_CPU_DISPATCH_A8SVE
#define INSTANTIATE_A8SVE(...) EXPAND(INSTANTIATE_CPU(sve, __VA_ARGS__))
#else
#define INSTANTIATE_A8SVE(...)
#endif

#ifdef ONEDAL_CPU_DISPATCH_AVX512
#define INSTANTIATE_AVX512(...) EXPAND(INSTANTIATE_CPU(avx512, __VA_ARGS__))
#else
#define INSTANTIATE_AVX512(...)
#endif

#ifdef ONEDAL_CPU_DISPATCH_AVX2
#define INSTANTIATE_AVX2(...) EXPAND(INSTANTIATE_CPU(avx2, __VA_ARGS__))
#else
#define INSTANTIATE_AVX2(...)
#endif

#ifdef ONEDAL_CPU_DISPATCH_SSE42
#define INSTANTIATE_SSE42(...) EXPAND(INSTANTIATE_CPU(sse42, __VA_ARGS__))
#else
#define INSTANTIATE_SSE42(...)
#endif

#define INSTANTIATE_SSE2(...) EXPAND(INSTANTIATE_CPU(sse2, __VA_ARGS__))

#if defined(TARGET_X86_64)
#define INSTANTIATE_FLOAT(name, Float, argdecl) \
    INSTANTIATE_AVX512(name, Float, argdecl)    \
    INSTANTIATE_AVX2(name, Float, argdecl)      \
    INSTANTIATE_SSE42(name, Float, argdecl)     \
    INSTANTIATE_SSE2(name, Float, argdecl)
#elif defined(TARGET_ARM)
#define INSTANTIATE_FLOAT(name, Float, argdecl) INSTANTIATE_A8SVE(name, Float, argdecl)
#endif

#define FUNC_TEMPLATE(prefix, name, fargdecl, cargdecl, fargcall, cargcall) \
    FUNC_DECL(prefix, s, name, fargdecl(float), fargcall)                   \
    FUNC_DECL(prefix, d, name, fargdecl(double), fargcall)                  \
                                                                            \
    namespace oneapi::dal::backend::micromkl {                              \
                                                                            \
    template <typename Cpu, typename Float>                                 \
    void name cargdecl(Float) {                                             \
        static_assert(sizeof(std::int64_t) == sizeof(DAAL_INT));            \
        if constexpr (std::is_same_v<Float, float>) {                       \
            FUNC_CALL(prefix, s, name, cargcall)                            \
        }                                                                   \
        else {                                                              \
            FUNC_CALL(prefix, d, name, cargcall)                            \
        }                                                                   \
    }                                                                       \
                                                                            \
    INSTANTIATE_FLOAT(name, float, cargdecl)                                \
    INSTANTIATE_FLOAT(name, double, cargdecl)                               \
    }
