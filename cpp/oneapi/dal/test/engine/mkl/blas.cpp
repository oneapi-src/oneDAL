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

#include "oneapi/dal/test/engine/mkl/blas.hpp"
#include <type_traits>

#ifdef ONEDAL_REF
#if (defined(DAAL_REF) && (INTPTR_MAX == INT64_MAX))
using GEMM_INT = std::int64_t;
#else
using GEMM_INT = std::int32_t;
#endif

extern "C" {
extern void sgemm_(const char *,
                   const char *,
                   const GEMM_INT *,
                   const GEMM_INT *,
                   const GEMM_INT *,
                   const float *,
                   const float *,
                   const GEMM_INT *,
                   const float *,
                   const GEMM_INT *,
                   const float *,
                   const float *,
                   const GEMM_INT *);
extern void dgemm_(const char *,
                   const char *,
                   const GEMM_INT *,
                   const GEMM_INT *,
                   const GEMM_INT *,
                   const double *,
                   const double *,
                   const GEMM_INT *,
                   const double *,
                   const GEMM_INT *,
                   const double *,
                   const double *,
                   const GEMM_INT *);
}

#define SGEMM(...) sgemm_(__VA_ARGS__)
#define DGEMM(...) dgemm_(__VA_ARGS__)

#else
#include <mkl_blas.h>

#define SGEMM(...) sgemm(__VA_ARGS__)
#define DGEMM(...) dgemm(__VA_ARGS__)

using GEMM_INT = MKL_INT;

#endif

namespace oneapi::dal::test::engine::mkl {

#define GEMM_PARAMETERS_C(Float)                                                                  \
    bool transa, bool transb, std::int64_t m, std::int64_t n, std::int64_t k, Float alpha,        \
        const Float *a, std::int64_t lda, const Float *b, std::int64_t ldb, Float beta, Float *c, \
        std::int64_t ldc

template <typename Float>
void gemm(GEMM_PARAMETERS_C(Float)) {
    const char ta = transa ? 'T' : 'N';
    const char tb = transb ? 'T' : 'N';
    const GEMM_INT _m = static_cast<GEMM_INT>(m);
    const GEMM_INT _n = static_cast<GEMM_INT>(n);
    const GEMM_INT _k = static_cast<GEMM_INT>(k);
    const GEMM_INT _lda = static_cast<GEMM_INT>(lda);
    const GEMM_INT _ldb = static_cast<GEMM_INT>(ldb);
    const GEMM_INT _ldc = static_cast<GEMM_INT>(ldc);
    if constexpr (std::is_same_v<Float, float>) {
        SGEMM(&ta, &tb, &_m, &_n, &_k, &alpha, a, &_lda, b, &_ldb, &beta, c, &_ldc);
    }
    else {
        DGEMM(&ta, &tb, &_m, &_n, &_k, &alpha, a, &_lda, b, &_ldb, &beta, c, &_ldc);
    }
}

#define INSTANTIATE_GEMM(Float) template void gemm<Float>(GEMM_PARAMETERS_C(Float));

INSTANTIATE_GEMM(float)
INSTANTIATE_GEMM(double)

} // namespace oneapi::dal::test::engine::mkl
