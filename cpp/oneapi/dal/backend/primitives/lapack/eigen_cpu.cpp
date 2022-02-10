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

#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/primitives/lapack/eigen.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Cpu, typename Float>
void flip_eigvals_impl_cpu(Float* a,
                           Float* w,
                           std::int64_t n,
                           std::int64_t lda,
                           std::int64_t w_count,
                           Float* a_flipped,
                           std::int64_t lda_flipped,
                           Float* w_flipped) {
    ONEDAL_ASSERT(a);
    ONEDAL_ASSERT(w);
    ONEDAL_ASSERT(a_flipped);
    ONEDAL_ASSERT(w_flipped);
    ONEDAL_ASSERT(n > 0);
    ONEDAL_ASSERT(lda >= n);
    ONEDAL_ASSERT(w_count > 0);
    ONEDAL_ASSERT(w_count <= n);

    if (a == a_flipped) {
        ONEDAL_ASSERT(lda == lda_flipped);

        for (std::int64_t i = 0; i < n / 2; i++) {
            const std::int64_t src_i = i;
            const std::int64_t dst_i = n - i - 1;
            for (std::int64_t j = 0; j < n; j++) {
                std::swap(a[src_i * lda + j], a[dst_i * lda + j]);
            }
        }
    }
    else {
        PRAGMA_IVDEP
        for (std::int64_t i = 0; i < w_count; i++) {
            const std::int64_t src_i = n - i - 1;
            const std::int64_t dst_i = i;
            for (std::int64_t j = 0; j < n; j++) {
                a_flipped[dst_i * lda_flipped + j] = a[src_i * lda + j];
            }
        }
    }

    if (w == w_flipped) {
        ONEDAL_ASSERT(n == w_count);

        for (std::int64_t i = 0; i < n / 2; i++) {
            const std::int64_t src_i = i;
            const std::int64_t dst_i = n - i - 1;
            std::swap(w[src_i], w[dst_i]);
        }
    }
    else {
        PRAGMA_IVDEP
        for (std::int64_t i = 0; i < w_count; i++) {
            const std::int64_t src_i = n - i - 1;
            const std::int64_t dst_i = i;
            w_flipped[dst_i] = w[src_i];
        }
    }
}

#define INSTANTIATE(Cpu, Float)                                   \
    template void flip_eigvals_impl_cpu<Cpu, Float>(Float*,       \
                                                    Float*,       \
                                                    std::int64_t, \
                                                    std::int64_t, \
                                                    std::int64_t, \
                                                    Float*,       \
                                                    std::int64_t, \
                                                    Float*);

INSTANTIATE(__CPU_TAG__, float)
INSTANTIATE(__CPU_TAG__, double)

} // namespace oneapi::dal::backend::primitives
