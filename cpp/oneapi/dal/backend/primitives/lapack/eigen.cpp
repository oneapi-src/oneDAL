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

#include "oneapi/dal/backend/primitives/lapack/eigen.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/micomkl/micromkl.hpp"

namespace oneapi::dal::backend::primitives {

template <typename... Args>
inline void syevd(Args&&... args) {
    dispatch_by_cpu(context_cpu{}, [&](auto cpu) {
        using dal::backend::micromkl::syevd;
        syevd<decltype(cpu)>(std::forward<Args>(args)...);
    });
}

template <typename Float>
void sym_eigvals_impl(Float* a, std::int64_t n, std::int64_t lda, Float* w) {
    ONEDAL_ASSERT(a);
    ONEDAL_ASSERT(w);
    ONEDAL_ASSERT(n > 0);
    ONEDAL_ASSERT(lda >= n);

    const std::int64_t lwork = 2 * n * n + 6 * n + 1;
    const std::int64_t liwork = 5 * n + 3;

    ONEDAL_ASSERT(lwork > n);
    ONEDAL_ASSERT(liwork > n);

    const auto work = ndarray<Float, 1>::empty(lwork);
    const auto iwork = ndarray<std::int64_t, 1>::empty(liwork);

    Float* work_ptr = work.get_mutable_data();
    std::int64_t* iwork_ptr = iwork.get_mutable_data();

    std::int64_t info;
    syevd('V', 'U', n, a, lda, w, work_ptr, lwork, iwork_ptr, liwork, info);

    if (info != 0) {
        throw internal_error{ dal::detail::error_messages::failed_to_compute_eigenvectors() };
    }
}

template <typename Float>
void flip_eigvals_impl(Float* a, Float* w, std::int64_t n, std::int64_t lda) {
    ONEDAL_ASSERT(a);
    ONEDAL_ASSERT(w);
    ONEDAL_ASSERT(n > 0);
    ONEDAL_ASSERT(lda >= n);

    for (std::int64_t i = 0; i < n / 2; i++) {
        const std::int64_t src_i = i;
        const std::int64_t dst_i = n - i - 1;
        std::swap(w[src_i], w[dst_i]);
        for (std::int64_t j = 0; j < n; j++) {
            std::swap(a[src_i * lda + j], a[dst_i * lda + j]);
        }
    }
}

#define INSTANTIATE(F)                                                            \
    template void sym_eigvals_impl(F* a, std::int64_t n, std::int64_t lda, F* w); \
    template void flip_eigvals_impl(F* a, F* w, std::int64_t n, std::int64_t lda);

INSTANTIATE(float)
INSTANTIATE(double)

} // namespace oneapi::dal::backend::primitives
