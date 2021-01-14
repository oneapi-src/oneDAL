/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/backend/dispatcher.hpp"

// Disable clang-format as it dramatically affects redability of macro definitions
// clang-format off

#define DECLARE_EXTERN_FUNCTION_ISA(prefix, isa, ReturnType, name, parameters) \
    ReturnType prefix##isa##_##name parameters;

#define DECLARE_EXTERN_FUNCTION(prefix, ReturnType, name, parameters)             \
    extern "C" {                                                                  \
        DECLARE_EXTERN_FUNCTION_ISA(prefix, sse2, ReturnType, name, parameters)   \
        DECLARE_EXTERN_FUNCTION_ISA(prefix, ssse3, ReturnType, name, parameters)  \
        DECLARE_EXTERN_FUNCTION_ISA(prefix, sse42, ReturnType, name, parameters)  \
        DECLARE_EXTERN_FUNCTION_ISA(prefix, avx, ReturnType, name, parameters)    \
        DECLARE_EXTERN_FUNCTION_ISA(prefix, avx2, ReturnType, name, parameters)   \
        DECLARE_EXTERN_FUNCTION_ISA(prefix, avx512, ReturnType, name, parameters) \
    }

#define DECLARE_FUNCTION_TEMPLATE(prefix, ReturnType, name, parameters) \
    template <typename Cpu>                                             \
    inline ReturnType prefix##name parameters;                          \

#define DEFINE_FUNCTION_TEMPLATE_SPEC_ISA(prefix, isa, ReturnType, name, parameters, args) \
    template <>                                                                            \
    inline ReturnType prefix##name<oneapi::dal::backend::cpu_dispatch_##isa> parameters {  \
        return prefix##isa##_##name args;                                                  \
    }

#define DEFINE_FUNCTION(prefix, ReturnType, name, parameters, args)                           \
    namespace oneapi::dal::backend::mkl {                                                     \
        DECLARE_FUNCTION_TEMPLATE(prefix, ReturnType, name, parameters)                       \
        DEFINE_FUNCTION_TEMPLATE_SPEC_ISA(prefix, sse2, ReturnType, name, parameters, args)   \
        DEFINE_FUNCTION_TEMPLATE_SPEC_ISA(prefix, ssse3, ReturnType, name, parameters, args)  \
        DEFINE_FUNCTION_TEMPLATE_SPEC_ISA(prefix, sse42, ReturnType, name, parameters, args)  \
        DEFINE_FUNCTION_TEMPLATE_SPEC_ISA(prefix, avx, ReturnType, name, parameters, args)    \
        DEFINE_FUNCTION_TEMPLATE_SPEC_ISA(prefix, avx2, ReturnType, name, parameters, args)   \
        DEFINE_FUNCTION_TEMPLATE_SPEC_ISA(prefix, avx512, ReturnType, name, parameters, args) \
    }

#define IMPORT_MKL_FUNCTION(prefix, ReturnType, name, parameters, args) \
    DECLARE_EXTERN_FUNCTION(prefix, ReturnType, name, parameters)       \
    DEFINE_FUNCTION(prefix, ReturnType, name, parameters, args)

#define GEMM_PARAMETERS(Float)                                                             \
    (const char *transa, const char *transb, const std::int64_t *m, const std::int64_t *n, \
     const std::int64_t *k, const Float *alpha, const Float *a, const std::int64_t *lda,   \
     const Float *b, const std::int64_t *ldb, const Float *beta, const Float *c,           \
     const std::int64_t *ldc)

#define GEMM_ARGS(Float) (transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

IMPORT_MKL_FUNCTION(fpk_blas_, void, sgemm, GEMM_PARAMETERS(float), GEMM_ARGS(float))
IMPORT_MKL_FUNCTION(fpk_blas_, void, dgemm, GEMM_PARAMETERS(double), GEMM_ARGS(double))

#undef DECLARE_EXTERN_FUNCTION_ISA
#undef DECLARE_EXTERN_FUNCTION
#undef DECLARE_FUNCTION_TEMPLATE
#undef DEFINE_FUNCTION_TEMPLATE_SPEC_ISA
#undef DEFINE_FUNCTION
#undef IMPORT_MKL_FUNCTION
#undef GEMM_PARAMETERS
#undef GEMM_ARGS
