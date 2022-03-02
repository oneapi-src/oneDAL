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
#include <mkl_blas.h>

namespace oneapi::dal::test::engine::mkl {

#define GEMM_PARAMETERS_C(Float)                                                                  \
    bool transa, bool transb, std::int64_t m, std::int64_t n, std::int64_t k, Float alpha,        \
        const Float *a, std::int64_t lda, const Float *b, std::int64_t ldb, Float beta, Float *c, \
        std::int64_t ldc

template <typename Float>
void gemm(GEMM_PARAMETERS_C(Float)) {
    const char ta = transa ? 'T' : 'N';
    const char tb = transb ? 'T' : 'N';
    const MKL_INT _m = static_cast<MKL_INT>(m);
    const MKL_INT _n = static_cast<MKL_INT>(n);
    const MKL_INT _k = static_cast<MKL_INT>(k);
    const MKL_INT _lda = static_cast<MKL_INT>(lda);
    const MKL_INT _ldb = static_cast<MKL_INT>(ldb);
    const MKL_INT _ldc = static_cast<MKL_INT>(ldc);
    if constexpr (std::is_same_v<Float, float>) {
        sgemm(&ta, &tb, &_m, &_n, &_k, &alpha, a, &_lda, b, &_ldb, &beta, c, &_ldc);
    }
    else {
        dgemm(&ta, &tb, &_m, &_n, &_k, &alpha, a, &_lda, b, &_ldb, &beta, c, &_ldc);
    }
}

#define INSTANTIATE_GEMM(Float) template void gemm<Float>(GEMM_PARAMETERS_C(Float));

INSTANTIATE_GEMM(float)
INSTANTIATE_GEMM(double)

} // namespace oneapi::dal::test::engine::mkl
