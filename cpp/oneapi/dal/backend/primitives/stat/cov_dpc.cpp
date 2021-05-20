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

#include "oneapi/dal/backend/primitives/stat/cov.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/loops.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include <CL/sycl/ONEAPI/experimental/builtins.hpp>

namespace oneapi::dal::backend::primitives {

template <typename Float>
inline void validate_input(const sycl::queue& q,
                           const ndview<Float, 2>& data,
                           const ndview<Float, 1>& sums,
                           const ndview<Float, 2>& corr,
                           const ndview<Float, 1>& means,
                           const ndview<Float, 1>& vars,
                           const ndview<Float, 1>& tmp) {
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(sums.has_data());
    ONEDAL_ASSERT(corr.has_mutable_data());
    ONEDAL_ASSERT(means.has_mutable_data());
    ONEDAL_ASSERT(vars.has_mutable_data());
    ONEDAL_ASSERT(tmp.has_mutable_data());
    ONEDAL_ASSERT(corr.get_dimension(0) == corr.get_dimension(1),
                  "Correlation matrix must be square");
    ONEDAL_ASSERT(corr.get_dimension(0) == data.get_dimension(1),
                  "Dimensions of correlation matrix must match feature count");
    ONEDAL_ASSERT(sums.get_dimension(0) == data.get_dimension(1),
                  "Element count of sums must match feature count");
    ONEDAL_ASSERT(vars.get_dimension(0) == data.get_dimension(1),
                  "Element count of vars must match feature count");
    ONEDAL_ASSERT(means.get_dimension(0) == data.get_dimension(1),
                  "Element count of means must match feature count");
    ONEDAL_ASSERT(tmp.get_dimension(0) == data.get_dimension(1),
                  "Element count of temporary buffer must match feature count");
    ONEDAL_ASSERT(is_known_usm(q, sums.get_data()));
    ONEDAL_ASSERT(is_known_usm(q, data.get_data()));
    ONEDAL_ASSERT(is_known_usm(q, corr.get_mutable_data()));
    ONEDAL_ASSERT(is_known_usm(q, means.get_mutable_data()));
    ONEDAL_ASSERT(is_known_usm(q, vars.get_mutable_data()));
    ONEDAL_ASSERT(is_known_usm(q, tmp.get_mutable_data()));
}

template <typename Float>
inline sycl::event prepare_correlation(sycl::queue& q,
                                       std::int64_t row_count,
                                       const ndview<Float, 1>& sums,
                                       const ndview<Float, 2>& corr,
                                       ndview<Float, 1>& means,
                                       ndview<Float, 1>& vars,
                                       ndview<Float, 1>& tmp,
                                       const event_vector& deps) {
    const std::int64_t n = row_count;
    const std::int64_t p = sums.get_count();
    const Float inv_n = Float(1.0 / double(n));
    const Float inv_n1 = (n > 1.0f) ? Float(1.0 / double(n - 1)) : 1.0f;

    const Float* sums_ptr = sums.get_data();
    const Float* corr_ptr = corr.get_mutable_data();
    Float* means_ptr = means.get_mutable_data();
    Float* vars_ptr = vars.get_mutable_data();
    Float* tmp_ptr = tmp.get_mutable_data();

    const Float eps = std::numeric_limits<Float>::epsilon();

    return q.submit([&](sycl::handler& cgh) {
        const auto range = make_multiple_nd_range_1d(p, device_max_wg_size(q));

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::nd_item<1> id) {
            const std::int64_t i = id.get_global_id();
            if (i < p) {
                const Float s = sums_ptr[i];
                const Float m = inv_n * s * s;
                const Float c = corr_ptr[i * p + i];
                const Float v = c - m;

                means_ptr[i] = inv_n * s;
                vars_ptr[i] = inv_n1 * v;

                // If $Var[x_i] > 0$ is close to zero, add $\varepsilon$
                // to avoid NaN/Inf in the resulting correlation matrix
                tmp_ptr[i] = v + eps * Float(v < eps);
            }
        });
    });
}

template <typename Float>
inline sycl::event finalize_correlation(sycl::queue& q,
                                        std::int64_t row_count,
                                        const ndview<Float, 1>& sums,
                                        const ndview<Float, 1>& tmp,
                                        ndview<Float, 2>& corr,
                                        const event_vector& deps) {
    const std::int64_t n = row_count;
    const std::int64_t p = sums.get_count();
    const Float inv_n = Float(1.0 / double(n));

    const Float* sums_ptr = sums.get_data();
    const Float* tmp_ptr = tmp.get_mutable_data();
    Float* corr_ptr = corr.get_mutable_data();

    return q.submit([&](sycl::handler& cgh) {
        const auto range = make_range_2d(p, p);

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::nd_item<2> id) {
            const std::int64_t gi = id.get_global_linear_id();
            const std::int64_t i = id.get_global_id(0);
            const std::int64_t j = id.get_global_id(1);

            if (i < p && j < p) {
                const Float is_diag = Float(i == j);

                Float c = corr_ptr[gi];
                c -= inv_n * sums_ptr[i] * sums_ptr[j];
                c *= sycl::rsqrt(tmp_ptr[i] * tmp_ptr[j]);
                corr_ptr[gi] = c * (Float(1.0) - is_diag) + is_diag;
            }
        });
    });
}

template <typename Float>
sycl::event correlation(sycl::queue& q,
                        const ndview<Float, 2>& data,
                        const ndview<Float, 1>& sums,
                        ndview<Float, 2>& corr,
                        ndview<Float, 1>& means,
                        ndview<Float, 1>& vars,
                        ndview<Float, 1>& tmp,
                        const event_vector& deps) {
    validate_input(q, data, sums, corr, means, vars, tmp);

    auto gemm_event = gemm(q, data.t(), data, corr, Float(1), Float(0), deps);

    auto prepare_event =
        prepare_correlation(q, data.get_dimension(0), sums, corr, means, vars, tmp, { gemm_event });

    auto finalize_event =
        finalize_correlation(q, data.get_dimension(0), sums, tmp, corr, { prepare_event });

    return finalize_event;
}

#define INSTANTIATE(F)                                                     \
    template ONEDAL_EXPORT sycl::event correlation<F>(sycl::queue&,        \
                                                      const ndview<F, 2>&, \
                                                      const ndview<F, 1>&, \
                                                      ndview<F, 2>&,       \
                                                      ndview<F, 1>&,       \
                                                      ndview<F, 1>&,       \
                                                      ndview<F, 1>&,       \
                                                      const event_vector&);

INSTANTIATE(float)
INSTANTIATE(double)

} // namespace oneapi::dal::backend::primitives
