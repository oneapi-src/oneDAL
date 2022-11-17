/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/lapack/solve.hpp"

namespace oneapi::dal::backend::primitives {

namespace detail {

struct potrs_params {
    std::int64_t n;
    std::int64_t nrhs;
    std::int64_t lda;
    std::int64_t ldb;
    mkl::uplo uplo;
};

template <mkl::uplo uplo, typename Float, ndorder xlayout, ndorder ylayout>
auto get_potrs_params(const ndview<Float, 2, xlayout>& x, const ndview<Float, 2, ylayout>& y) {
    potrs_params result;

    ONEDAL_ASSERT(xlayout == ndorder::c);
    ONEDAL_ASSERT(ylayout == ndorder::c);

    result.lda = x.get_stride(0);
    result.ldb = y.get_stride(0);
    result.n = x.get_dimension(1);
    result.uplo = ident_uplo(uplo);
    result.nrhs = y.get_dimension(0);

    return result;
}

template <mkl::uplo uplo, typename Float, ndorder xlayout, ndorder ylayout>
std::int64_t potrs_scratchpad_size(sycl::queue& queue,
                                   const ndview<Float, 2, xlayout>& x,
                                   const ndview<Float, 2, ylayout>& y) {
    ONEDAL_ASSERT(x.has_data());
    ONEDAL_ASSERT(y.has_data());
    const auto [ncount, nrhs, nlda, nldb, nuplo] = get_potrs_params<uplo>(x, y);
    return mkl::lapack::potrs_scratchpad_size<Float>(queue, nuplo, ncount, nrhs, nlda, nldb);
}

template <mkl::uplo uplo, typename Float, ndorder xlayout, ndorder ylayout>
sycl::event potrs_solution(sycl::queue& queue,
                           ndview<Float, 2, xlayout>& x,
                           ndview<Float, 2, ylayout>& y,
                           array<Float>& scratchpad,
                           const event_vector& deps) {
    ONEDAL_PROFILER_TASK(potrs_kernel, queue);

    ONEDAL_ASSERT(x.has_mutable_data());
    ONEDAL_ASSERT(y.has_mutable_data());
    ONEDAL_ASSERT(scratchpad.has_mutable_data());
    const auto [ncount, nrhs, nlda, nldb, nuplo] = get_potrs_params<uplo>(x, y);

#ifdef ONEDAL_ENABLE_ASSERT
    const auto scratchpad_real_count = scratchpad.get_count();
    const auto scratchpad_want_count = potrs_scratchpad_size<uplo>(queue, x, y);
    ONEDAL_ASSERT(scratchpad_real_count >= scratchpad_want_count);
#endif

    auto* x_ptr = x.get_mutable_data();
    auto* y_ptr = y.get_mutable_data();
    const auto scount = scratchpad.get_count();
    auto* s_ptr = scratchpad.get_mutable_data();
    return mkl::lapack::potrs(queue,
                              nuplo,
                              ncount,
                              nrhs,
                              x_ptr,
                              nlda,
                              y_ptr,
                              nldb,
                              s_ptr,
                              scount,
                              deps);
}

} // namespace detail

template <mkl::uplo uplo, typename Float, ndorder xlayout, ndorder ylayout>
array<Float> potrs_scratchpad(sycl::queue& q,
                              const ndview<Float, 2, xlayout>& x,
                              const ndview<Float, 2, ylayout>& y,
                              const sycl::usm::alloc& alloc) {
    const auto count = detail::potrs_scratchpad_size<uplo>(q, x, y);
    return array<Float>::empty(q, count, alloc);
}

template <mkl::uplo uplo, typename Float, ndorder xlayout, ndorder ylayout>
sycl::event potrs_solution(sycl::queue& q,
                           ndview<Float, 2, xlayout>& x,
                           ndview<Float, 2, ylayout>& y,
                           opt_array<Float>& scratchpad,
                           const event_vector& dependencies) {
    if (!scratchpad.has_value())
        scratchpad = potrs_scratchpad<uplo>(q, x, y);
    return detail::potrs_solution<uplo>(q, x, y, *scratchpad, dependencies);
}

#define INSTANTIATE(U, F, XL, YL)                                    \
    template array<F> potrs_scratchpad<U>(sycl::queue&,              \
                                          const ndview<F, 2, XL>&,   \
                                          const ndview<F, 2, YL>&,   \
                                          const sycl::usm::alloc&);  \
    template sycl::event potrs_solution<U>(sycl::queue&,             \
                                           ndview<F, 2, XL>&,        \
                                           ndview<F, 2, YL>&,        \
                                           std::optional<array<F>>&, \
                                           const event_vector&);

#define INSTANTIATE_YL(U, F, XL)      \
    INSTANTIATE(U, F, XL, ndorder::f) \
    INSTANTIATE(U, F, XL, ndorder::c)

#define INSTANTIATE_XL(U, F)         \
    INSTANTIATE_YL(U, F, ndorder::f) \
    INSTANTIATE_YL(U, F, ndorder::c)

#define INSTANTIATE_F(U)     \
    INSTANTIATE_XL(U, float) \
    INSTANTIATE_XL(U, double)

INSTANTIATE_F(mkl::uplo::upper)
INSTANTIATE_F(mkl::uplo::lower)

} // namespace oneapi::dal::backend::primitives
