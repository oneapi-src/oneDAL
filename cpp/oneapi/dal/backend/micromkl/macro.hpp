/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef __MICROMKL_INCLUDE_GUARD__
#error "This header cannot be included outside of micromkl module"
#endif

#define FUNC_NAME(prefix, name)          prefix##_##name
#define FUNC_NAME_CPU(cpu, prefix, name) prefix##_##cpu##_##name
#define DISPATCH_ID_NAME(cpu)            oneapi::dal::backend::cpu_dispatch_##cpu

#define FUNC_CPU_DECL(cpu, prefix, name, argdecl) \
    extern "C" void FUNC_NAME_CPU(cpu, prefix, name) argdecl;

#define DISPATCH_FUNC_DECL(prefix, name, arcdecl) \
    template <typename Cpu>                       \
    ONEDAL_FORCEINLINE void FUNC_NAME(prefix, name) arcdecl;

#define DISPATCH_FUNC_CPU(cpu, prefix, name, arcdecl, argcall)                       \
    template <>                                                                      \
    ONEDAL_FORCEINLINE void FUNC_NAME(prefix, name)<DISPATCH_ID_NAME(cpu)> arcdecl { \
        FUNC_NAME_CPU(cpu, prefix, name) argcall;                                    \
    }

#define FUNC_CPU(cpu, prefix, name, argdecl, argcall) \
    FUNC_CPU_DECL(cpu, prefix, name, argdecl)         \
    DISPATCH_FUNC_CPU(cpu, prefix, name, argdecl, argcall)

#ifdef __APPLE__
#define FUNC_SSE2(...)  FUNC_CPU(sse42, __VA_ARGS__)
#define FUNC_SSSE3(...) FUNC_CPU(sse42, __VA_ARGS__)
#else
#define FUNC_SSE2(...)  FUNC_CPU(sse2, __VA_ARGS__)
#define FUNC_SSSE3(...) FUNC_CPU(ssse3, __VA_ARGS__)
#endif

#define FUNC_SSE42(...)  FUNC_CPU(sse42, __VA_ARGS__)
#define FUNC_AVX(...)    FUNC_CPU(avx, __VA_ARGS__)
#define FUNC_AVX2(...)   FUNC_CPU(avx2, __VA_ARGS__)
#define FUNC_AVX512(...) FUNC_CPU(avx512, __VA_ARGS__)

#define FUNC(prefix, name, argdecl, argcall)   \
    DISPATCH_FUNC_DECL(prefix, name, argdecl)  \
    FUNC_SSE2(prefix, name, argdecl, argcall)  \
    FUNC_SSSE3(prefix, name, argdecl, argcall) \
    FUNC_SSE42(prefix, name, argdecl, argcall) \
    FUNC_AVX(prefix, name, argdecl, argcall)   \
    FUNC_AVX2(prefix, name, argdecl, argcall)  \
    FUNC_AVX512(prefix, name, argdecl, argcall)

#define INSTANTIATE_CPU(cpu, name, Float, argdecl) \
    template void name<DISPATCH_ID_NAME(cpu), Float> argdecl(Float);

#define INSTANTIATE_SSE2(...) INSTANTIATE_CPU(sse2, __VA_ARGS__)

#ifdef ONEDAL_CPU_DISPATCH_SSSE3
#define INSTANTIATE_SSSE3(...) INSTANTIATE_CPU(ssse3, __VA_ARGS__)
#else
#define INSTANTIATE_SSSE3(...)
#endif

#ifdef ONEDAL_CPU_DISPATCH_SSE42
#define INSTANTIATE_SSE42(...) INSTANTIATE_CPU(sse42, __VA_ARGS__)
#else
#define INSTANTIATE_SSE42(...)
#endif

#ifdef ONEDAL_CPU_DISPATCH_AVX
#define INSTANTIATE_AVX(...) INSTANTIATE_CPU(avx, __VA_ARGS__)
#else
#define INSTANTIATE_AVX(...)
#endif

#ifdef ONEDAL_CPU_DISPATCH_AVX2
#define INSTANTIATE_AVX2(...) INSTANTIATE_CPU(avx2, __VA_ARGS__)
#else
#define INSTANTIATE_AVX2(...)
#endif

#ifdef ONEDAL_CPU_DISPATCH_AVX512
#define INSTANTIATE_AVX512(...) INSTANTIATE_CPU(avx512, __VA_ARGS__)
#else
#define INSTANTIATE_AVX512(...)
#endif

#define INSTANTIATE_FLOAT(name, Float, argdecl) \
    INSTANTIATE_SSE2(name, Float, argdecl)      \
    INSTANTIATE_SSSE3(name, Float, argdecl)     \
    INSTANTIATE_SSE42(name, Float, argdecl)     \
    INSTANTIATE_AVX(name, Float, argdecl)       \
    INSTANTIATE_AVX2(name, Float, argdecl)      \
    INSTANTIATE_AVX512(name, Float, argdecl)

#define FUNC_TEMPLATE(prefix, name, fargdecl, cargdecl, fargcall, cargcall) \
    FUNC(prefix, s##name, fargdecl(float), fargcall)                        \
    FUNC(prefix, d##name, fargdecl(double), fargcall)                       \
                                                                            \
    namespace oneapi::dal::backend::micromkl {                              \
                                                                            \
    template <typename Cpu, typename Float>                                 \
    void name cargdecl(Float) {                                             \
        static_assert(sizeof(std::int64_t) == sizeof(DAAL_INT));            \
        if constexpr (std::is_same_v<Float, float>) {                       \
            prefix##_s##name<Cpu> cargcall;                                 \
        }                                                                   \
        else {                                                              \
            prefix##_d##name<Cpu> cargcall;                                 \
        }                                                                   \
    }                                                                       \
                                                                            \
    INSTANTIATE_FLOAT(name, float, cargdecl)                                \
    INSTANTIATE_FLOAT(name, double, cargdecl)                               \
    }
