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

#include "oneapi/dal/backend/linalg/dot.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

extern "C" {

#define GEMM_ARGS(Float) \
    const char *transa, const char *transb, \
    const std::int64_t *m, const std::int64_t *n, const std::int64_t *k, \
    const Float *alpha, \
    const Float *a, const std::int64_t *lda, \
    const Float *b, const std::int64_t *ldb, \
    const Float *beta, \
    const Float *c, const std::int64_t *ldc

#define GEMM_ARGS_PASS \
    transa, transb, m, n, k, alpha, \
    a, lda, \
    b, ldb, \
    beta, \
    c, ldc

#define DECLARE_FORTRAN_GEMM(isa) \
    extern void fpk_blas_##isa##_sgemm(GEMM_ARGS(float)); \
    extern void fpk_blas_##isa##_dgemm(GEMM_ARGS(double));

DECLARE_FORTRAN_GEMM(sse2);
DECLARE_FORTRAN_GEMM(ssse3);
DECLARE_FORTRAN_GEMM(sse42);
DECLARE_FORTRAN_GEMM(avx);
DECLARE_FORTRAN_GEMM(avx2);
DECLARE_FORTRAN_GEMM(avx512);

#undef DECLARE_FORTRAN_GEMM

} // extern "C"

namespace oneapi::dal::backend::linalg {

template <typename Float, typename Cpu>
inline void gemm_cpu(GEMM_ARGS(Float));

#define DECLARE_C_GEMM(isa) \
    template <> \
    inline void gemm_cpu<float, cpu_dispatch_##isa>(GEMM_ARGS(float)) { \
        fpk_blas_##isa##_sgemm(GEMM_ARGS_PASS); \
    } \
    template <> \
    inline void gemm_cpu<double, cpu_dispatch_##isa>(GEMM_ARGS(double)) { \
        fpk_blas_##isa##_dgemm(GEMM_ARGS_PASS); \
    }

DECLARE_C_GEMM(sse2);
DECLARE_C_GEMM(ssse3);
DECLARE_C_GEMM(sse42);
DECLARE_C_GEMM(avx);
DECLARE_C_GEMM(avx2);
DECLARE_C_GEMM(avx512);

#undef DECLARE_C_GEMM

template <typename Float>
inline void gemm(const context_cpu& ctx, GEMM_ARGS(Float)) {
    dispatch_by_cpu(ctx, [=](auto cpu) {
        gemm_cpu<Float, decltype(cpu)>(GEMM_ARGS_PASS);
    });
}

template <typename Float>
inline void c_gemm(
        const context_cpu& ctx,
        bool transa, bool transb,
        std::int64_t m, std::int64_t n, std::int64_t k,
        Float alpha,
        const Float *a, std::int64_t lda,
        const Float *b, std::int64_t ldb,
        Float beta,
        Float *c, std::int64_t ldc) {
    const char ta = transa ? 'T' : 'N';
    const char tb = transb ? 'T' : 'N';
    gemm<Float>(ctx, &ta, &tb, &m, &n, &k, &alpha,
                a, &lda, b, &ldb, &beta, c, &ldc);
}

template <typename Float>
void dot_op<Float>::operator()(const context_cpu& ctx,
                               const matrix<Float>& a,
                               const matrix<Float>& b,
                               matrix<Float>& c,
                               Float alpha, Float beta) const {
    const bool is_c_trans = (c.get_layout() == layout::row_major);
    if (is_c_trans) {
        const bool is_a_trans = (a.get_layout() == layout::column_major);
        const bool is_b_trans = (b.get_layout() == layout::column_major);
        c_gemm<Float>(ctx, is_b_trans, is_a_trans,
                      c.get_column_count(), c.get_row_count(), a.get_column_count(),
                      alpha,
                      b.get_data(), b.get_stride(),
                      a.get_data(), a.get_stride(),
                      beta,
                      c.get_mutable_data(), c.get_stride());
    }
    else {
        const bool is_a_trans = (a.get_layout() == layout::row_major);
        const bool is_b_trans = (b.get_layout() == layout::row_major);
        c_gemm<Float>(ctx, is_a_trans, is_b_trans,
                      c.get_row_count(), c.get_column_count(), a.get_column_count(),
                      alpha,
                      a.get_data(), a.get_stride(),
                      b.get_data(), b.get_stride(),
                      beta,
                      c.get_mutable_data(), c.get_stride());
    }
}

template struct dot_op<float>;
template struct dot_op<double>;

} // namespace oneapi::dal::backend::linalg
