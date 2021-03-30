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

#include "src/externals/service_lapack.h"

#include "oneapi/dal/backend/primitives/lapack/eigen.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/interop/common.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
static void syevd(char jobz,
                  char uplo,
                  std::int64_t n,
                  Float* a,
                  std::int64_t lda,
                  Float* w,
                  Float* work,
                  std::int64_t lwork,
                  DAAL_INT* iwork,
                  std::int64_t liwork,
                  std::int64_t& info) {
    static_assert(sizeof(std::int64_t) == sizeof(DAAL_INT));

    DAAL_INT daal_n = n;
    DAAL_INT daal_lda = lda;
    DAAL_INT daal_lwork = lwork;
    DAAL_INT daal_liwork = liwork;
    DAAL_INT daal_info;

    dispatch_by_cpu(context_cpu{}, [&](auto cpu) {
        using daal::internal::Lapack;
        using dal::backend::interop::to_daal_cpu_type;
        using lapack_t = Lapack<Float, to_daal_cpu_type<decltype(cpu)>::value>;

        lapack_t::xsyevd(&jobz,
                         &uplo,
                         &daal_n,
                         a,
                         &daal_lda,
                         w,
                         work,
                         &daal_lwork,
                         iwork,
                         &daal_liwork,
                         &daal_info);
    });

    info = daal_info;
}

template <typename Float>
void sym_eigval_impl(Float* a, std::int64_t n, std::int64_t lda, Float* w) {
    ONEDAL_ASSERT(a);
    ONEDAL_ASSERT(w);
    ONEDAL_ASSERT(n > 0);
    ONEDAL_ASSERT(lda >= n);

    const std::int64_t lwork = 2 * n * n + 6 * n + 1;
    const std::int64_t liwork = 5 * n + 3;

    ONEDAL_ASSERT(lwork > n);
    ONEDAL_ASSERT(liwork > n);

    const auto work = ndarray<Float, 1>::empty(lwork);
    const auto iwork = ndarray<DAAL_INT, 1>::empty(liwork);

    Float* work_ptr = work.get_mutable_data();
    DAAL_INT* iwork_ptr = iwork.get_mutable_data();

    std::int64_t info;
    syevd('V', 'U', n, a, lda, w, work_ptr, lwork, iwork_ptr, liwork, info);

    if (info != 0) {
        throw internal_error{ dal::detail::error_messages::failed_to_compute_eigenvectors() };
    }
}

#define INSTANTIATE(F) \
    template void sym_eigval_impl<F>(F * a, std::int64_t n, std::int64_t lda, F * w);

INSTANTIATE(float)
INSTANTIATE(double)

} // namespace oneapi::dal::backend::primitives
