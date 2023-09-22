/*******************************************************************************
* Copyright 2023 Intel Corporation
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
#include "oneapi/dal/backend/primitives/lapack/syevd.hpp"
#include "oneapi/dal/backend/primitives/blas/misc.hpp"

#include <mkl_dal_sycl.hpp>

namespace oneapi::dal::backend::primitives {

template <typename Float>
static sycl::event syevd_wrapper(sycl::queue& queue,
                                 mkl::job jobz,
                                 mkl::uplo uplo,
                                 std::int64_t n,
                                 Float* a,
                                 std::int64_t lda,
                                 Float* w,
                                 Float* scratchpad,
                                 std::int64_t scratchpad_size,
                                 const event_vector& deps) {
    ONEDAL_ASSERT(lda >= n);

    return mkl::lapack::syevd(queue, jobz, uplo, n, a, lda, w, scratchpad, scratchpad_size, deps);
}

template <mkl::job jobz, mkl::uplo uplo, typename Float>
sycl::event syevd(sycl::queue& queue,
                  ndview<Float, 2>& a,
                  ndview<Float, 1>& w,
                  Float* scratchpad,
                  std::int64_t scratchpad_size,
                  const event_vector& deps) {
    //ONEDAL_PROFILER_TASK(blas.syevd, queue);

    //ONEDAL_ASSERT(a.get_dimension(1) == w.get_dimension(0));
    //ONEDAL_ASSERT(a.has_mutable_data());

    const auto nd = a.get_dimension(0);
    auto* const a_ptr = a.get_mutable_data();
    auto* const w_ptr = w.get_mutable_data();
    const auto a_str = a.get_leading_stride();
    constexpr auto job = ident_job(jobz);
    constexpr auto ul = ident_uplo(uplo);
    return syevd_wrapper(queue,
                         job,
                         ul,
                         nd,
                         a_ptr,
                         a_str,
                         w_ptr,
                         scratchpad,
                         scratchpad_size,
                         deps);
}

#define INSTANTIATE(jobz, ul, F)                                                        \
    template ONEDAL_EXPORT sycl::event syevd<jobz, ul, F>(sycl::queue & queue,          \
                                                          ndview<F, 2> & a,             \
                                                          ndview<F, 1> & w,             \
                                                          F * scratchpad,               \
                                                          std::int64_t scratchpad_size, \
                                                          const event_vector& deps);

#define INSTANTIATE_FLOAT(jobz, ul) \
    INSTANTIATE(jobz, ul, float)    \
    INSTANTIATE(jobz, ul, double)

#define INSTANTIATE_JOB(ul)                \
    INSTANTIATE_FLOAT(mkl::job::novec, ul) \
    INSTANTIATE_FLOAT(mkl::job::vec, ul)

INSTANTIATE_JOB(mkl::uplo::upper)
INSTANTIATE_JOB(mkl::uplo::lower)

} // namespace oneapi::dal::backend::primitives
