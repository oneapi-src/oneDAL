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

#include "oneapi/dal/backend/micomkl/micromkl.hpp"
#include <daal/include/services/daal_defines.h>

namespace oneapi::dal::backend::micromkl {

#if defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64)
#define OS_WIN
#endif

#define FUNC_NAME(prefix, name) prefix##_##name
#define FUNC_NAME_CPU(cpu, prefix, name) prefix##_##cpu##_##name
#define DISPATCH_ID_NAME(cpu) oneapi::dal::backend::cpu_dispatch_##cpu

#define DISPATCH_FUNC_DECL(prefix, name, arcdecl, argcall) \
    template <typename Cpu>                                \
    void FUNC_NAME(prefix, name) arcdecl;

#define DISPATCH_FUNC_CPU(cpu, prefix, name, arcdecl, argcall)            \
    template <>                                                           \
    inline void FUNC_NAME(prefix, name)<DISPATCH_ID_NAME(cpu)> arcdecl {  \
        FUNC_NAME_CPU(cpu, prefix, name) argcall;                         \
    }

#define FUNC_CPU(cpu, prefix, name, argdecl, argcall)      \
    FUNC_NAME_CPU(cpu, prefix, name) argdecl;              \
    DISPATCH_FUNC_CPU(cpu, prefix, name, argdecl, argcall)

#define FUNC(prefix, name, argdecl, argcall)         \
    FUNC_CPU(sse2, prefix, name, argdecl, argcall)   \
    FUNC_CPU(ssse3, prefix, name, argdecl, argcall)  \
    FUNC_CPU(sse42, prefix, name, argdecl, argcall)  \
    FUNC_CPU(avx, prefix, name, argdecl, argcall)    \
    FUNC_CPU(avx2, prefix, name, argdecl, argcall)   \
    FUNC_CPU(avx512, prefix, name, argdecl, argcall)

#define FUNC_TEMPLATE(name, argdecl, argcall)                     \
    template <typename Float, typename Cpu>                       \
    void name argdecl(Float) {                                    \
        static_assert(sizeof(std::int64_t) == sizeof(DAAL_INT));  \
        if constexpr (std::is_same_v<Float, float>) {             \
            s##name argcalls;                                     \
        }                                                         \
        else {                                                    \
            d##name argcalls;                                     \
        }                                                         \
    }                                                             \
    template void name<float> argdecl(float);                     \
    template void name<double> argdecl(double);

#define SYEVD_F_DECLARGS(Float) (const char* jobz, const char* uplo,                                    \
                                 const std::int64_t* n, Float* a, const std::int64_t* lda,              \
                                 Float* w, Float* work, const std::int64_t* lwork, std::int64_t* iwork, \
                                 const std::int64_t* liwork, std::int64_t* info, int ijobz, int iuplo)
#define SYEVD_F_CALLARGS (jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info, ijobz, iuplo)

#define SYEVD_C_DECLARGS(Float) (char jobz, char uplo, std::int64_t n,                                  \
                                 Float* a, std::int64_t lda, Float* w, Float* work, std::int64_t lwork, \
                                 std::int64_t* iwork, std::int64_t liwork, std::int64_t& info)
#define SYEVD_C_DECLARGS (&jobz, &uplo, &n, a, &lda, &w, work, &lwork, iwork, &liwork, &info, 1, 1)

FUNC(fpk_lapack, ssyevd, SYEVD_F_DECLARGS(float), SYEVD_F_CALLARGS)
FUNC(fpk_lapack, dsyevd, SYEVD_F_DECLARGS(double), SYEVD_F_CALLARGS)
FUNC_TEMPLATE(syevd, SYEVD_C_DECLARGS, SYEVD_C_DECLARGS)

} // namespace oneapi::dal::backend::micromkl
