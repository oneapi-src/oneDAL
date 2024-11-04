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
#include <iostream>
namespace oneapi::dal::backend::primitives {

namespace detail {

struct potrf_params {
    std::int64_t n;
    std::int64_t lda;
    mkl::uplo uplo;
};

template <mkl::uplo uplo, typename Float, ndorder layout>
auto get_potrf_params(const ndview<Float, 2, layout>& x) {
    potrf_params result;

    ONEDAL_ASSERT(layout == ndorder::c);

    result.lda = x.get_stride(0);
    result.n = x.get_dimension(1);
    result.uplo = ident_uplo(uplo);

    return result;
}

template <mkl::uplo uplo, typename Float, ndorder layout>
std::int64_t potrf_scratchpad_size(sycl::queue& queue, const ndview<Float, 2, layout>& x) {
    ONEDAL_ASSERT(x.has_data());
    const auto [ncount, nlda, nuplo] = get_potrf_params<uplo>(x);
    return mkl::lapack::potrf_scratchpad_size<Float>(queue, nuplo, ncount, nlda);
}

template <mkl::uplo uplo, typename Float, ndorder layout>
sycl::event potrf_factorization(sycl::queue& queue,
                                ndview<Float, 2, layout>& x,
                                array<Float>& scratchpad,
                                const event_vector& deps) {
    ONEDAL_PROFILER_TASK(potrf_kernel, queue);
    std::cout<<"here potrf_factorization 1"<<std::endl;
    ONEDAL_ASSERT(x.has_mutable_data());
    ONEDAL_ASSERT(scratchpad.has_mutable_data());
    const auto [ncount, nlda, nuplo] = get_potrf_params<uplo>(x);
    std::cout<<"here potrf_factorization 2"<<std::endl;
    [[maybe_unused]] const auto scratchpad_real_count = scratchpad.get_count();
    std::cout<<"here potrf_factorization 3"<<std::endl;
    [[maybe_unused]] const auto scratchpad_want_count = potrf_scratchpad_size<uplo>(queue, x);
    ONEDAL_ASSERT(scratchpad_real_count >= scratchpad_want_count);
    std::cout<<"here potrf_factorization 4"<<std::endl;
    auto* x_ptr = x.get_mutable_data();
    std::cout<<"here potrf_factorization 5"<<std::endl;
    const auto scount = scratchpad.get_count();
    std::cout<<"here potrf_factorization 6"<<std::endl;
    auto* s_ptr = scratchpad.get_mutable_data();
    std::cout<<"here potrf_factorization 7"<<std::endl;
    //std::cout<<"nuplo ="<<nuplo<<std::endl;
    std::cout<<"ncount ="<<ncount<<std::endl;
    std::cout<<"nlda ="<<nlda<<std::endl;
    std::cout<<"scount ="<<scount<<std::endl;
    return mkl::lapack::potrf(queue, mkl::uplo::upper, ncount, x_ptr, nlda, s_ptr, scount, deps);
}

} // namespace detail

template <mkl::uplo uplo, typename Float, ndorder layout>
array<Float> potrf_scratchpad(sycl::queue& q,
                              const ndview<Float, 2, layout>& x,
                              const sycl::usm::alloc& alloc) {
    const auto count = detail::potrf_scratchpad_size<uplo>(q, x);
    return array<Float>::empty(q, count, alloc);
}

template <mkl::uplo uplo, typename Float, ndorder layout>
sycl::event potrf_factorization(sycl::queue& q,
                                ndview<Float, 2, layout>& x,
                                opt_array<Float>& scratchpad,
                                const event_vector& dependencies) {
    if (!scratchpad.has_value())
        scratchpad = potrf_scratchpad<uplo>(q, x);
    return detail::potrf_factorization<uplo>(q, x, *scratchpad, dependencies);
}

#define INSTANTIATE(U, F, L)                                              \
    template array<F> potrf_scratchpad<U>(sycl::queue&,                   \
                                          const ndview<F, 2, L>&,         \
                                          const sycl::usm::alloc&);       \
    template sycl::event potrf_factorization<U>(sycl::queue&,             \
                                                ndview<F, 2, L>&,         \
                                                std::optional<array<F>>&, \
                                                const event_vector&);

#define INSTANTIATE_L(U, F)       \
    INSTANTIATE(U, F, ndorder::f) \
    INSTANTIATE(U, F, ndorder::c)

#define INSTANTIATE_F(U)    \
    INSTANTIATE_L(U, float) \
    INSTANTIATE_L(U, double)

INSTANTIATE_F(mkl::uplo::upper)
INSTANTIATE_F(mkl::uplo::lower)

} // namespace oneapi::dal::backend::primitives
