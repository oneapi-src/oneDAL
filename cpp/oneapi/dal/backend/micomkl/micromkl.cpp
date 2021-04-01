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

#if defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64)
#define OS_WIN
#endif

#ifdef OS_WIN
#include <windows.h>
#endif

#include <daal/include/services/daal_defines.h>
#include "oneapi/dal/backend/micomkl/micromkl.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

#define FUNC_NAME(prefix, name)          prefix##_##name
#define FUNC_NAME_CPU(cpu, prefix, name) prefix##_##cpu##_##name
#define DISPATCH_ID_NAME(cpu)            oneapi::dal::backend::cpu_dispatch_##cpu

#define DISPATCH_FUNC_DECL(prefix, name, arcdecl, argcall) \
    template <typename Cpu>                                \
    ONEDAL_FORCEINLINE void FUNC_NAME(prefix, name) arcdecl;

#define DISPATCH_FUNC_CPU(cpu, prefix, name, arcdecl, argcall)                       \
    template <>                                                                      \
    ONEDAL_FORCEINLINE void FUNC_NAME(prefix, name)<DISPATCH_ID_NAME(cpu)> arcdecl { \
        FUNC_NAME_CPU(cpu, prefix, name) argcall;                                    \
    }

#define FUNC_CPU_DECL(cpu, prefix, name, argdecl) \
    extern "C" void FUNC_NAME_CPU(cpu, prefix, name) argdecl;

#define FUNC_CPU(cpu, prefix, name, argdecl, argcall) \
    FUNC_CPU_DECL(cpu, prefix, name, argdecl)         \
    DISPATCH_FUNC_CPU(cpu, prefix, name, argdecl, argcall)

#define FUNC(prefix, name, argdecl, argcall)           \
    DISPATCH_FUNC_DECL(prefix, name, argdecl, argcall) \
    FUNC_CPU(sse2, prefix, name, argdecl, argcall)     \
    FUNC_CPU(ssse3, prefix, name, argdecl, argcall)    \
    FUNC_CPU(sse42, prefix, name, argdecl, argcall)    \
    FUNC_CPU(avx, prefix, name, argdecl, argcall)      \
    FUNC_CPU(avx2, prefix, name, argdecl, argcall)     \
    FUNC_CPU(avx512, prefix, name, argdecl, argcall)

#define FUNC_TEMPLATE_INSTANTIATE(name, Float, cpu, argdecl) \
    template void name<DISPATCH_ID_NAME(cpu), Float> argdecl(Float);

#define FUNC_TEMPLATE_INSTANTIATE_FLOAT(name, Float, argdecl) \
    FUNC_TEMPLATE_INSTANTIATE(name, Float, sse2, argdecl)     \
    FUNC_TEMPLATE_INSTANTIATE(name, Float, ssse3, argdecl)    \
    FUNC_TEMPLATE_INSTANTIATE(name, Float, sse42, argdecl)    \
    FUNC_TEMPLATE_INSTANTIATE(name, Float, avx, argdecl)      \
    FUNC_TEMPLATE_INSTANTIATE(name, Float, avx2, argdecl)     \
    FUNC_TEMPLATE_INSTANTIATE(name, Float, avx512, argdecl)

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
    FUNC_TEMPLATE_INSTANTIATE_FLOAT(name, float, cargdecl)                  \
    FUNC_TEMPLATE_INSTANTIATE_FLOAT(name, double, cargdecl)                 \
    }

/* ============================================ SYEVD ============================================ */
#define SYEVD_F_DECLARGS(Float) \
    (const char* jobz,          \
     const char* uplo,          \
     const DAAL_INT* n,         \
     Float* a,                  \
     const DAAL_INT* lda,       \
     Float* w,                  \
     Float* work,               \
     const DAAL_INT* lwork,     \
     DAAL_INT* iwork,           \
     const DAAL_INT* liwork,    \
     DAAL_INT* info,            \
     int ijobz,                 \
     int iuplo)

#define SYEVD_C_DECLARGS(Float) \
    (char jobz,                 \
     char uplo,                 \
     std::int64_t n,            \
     Float* a,                  \
     std::int64_t lda,          \
     Float* w,                  \
     Float* work,               \
     std::int64_t lwork,        \
     std::int64_t* iwork,       \
     std::int64_t liwork,       \
     std::int64_t& info)

#define SYEVD_F_CALLARGS (jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info, ijobz, iuplo)

#define SYEVD_C_CALLARGS                   \
    (&jobz,                                \
     &uplo,                                \
     reinterpret_cast<DAAL_INT*>(&n),      \
     a,                                    \
     reinterpret_cast<DAAL_INT*>(&lda),    \
     w,                                    \
     work,                                 \
     reinterpret_cast<DAAL_INT*>(&lwork),  \
     reinterpret_cast<DAAL_INT*>(iwork),   \
     reinterpret_cast<DAAL_INT*>(&liwork), \
     reinterpret_cast<DAAL_INT*>(&info),   \
     1,                                    \
     1)

FUNC_TEMPLATE(fpk_lapack,
              syevd,
              SYEVD_F_DECLARGS,
              SYEVD_C_DECLARGS,
              SYEVD_F_CALLARGS,
              SYEVD_C_CALLARGS)
