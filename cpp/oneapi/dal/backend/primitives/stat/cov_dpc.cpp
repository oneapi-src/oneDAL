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
#include <sycl/ext/oneapi/experimental/builtins.hpp>

namespace oneapi::dal::backend::primitives {

template <typename Float>
sycl::event means(sycl::queue& q,
                  std::int64_t row_count,
                  const ndview<Float, 1>& sums,
                  ndview<Float, 1>& means,
                  const event_vector& deps) {
    ONEDAL_ASSERT(sums.has_data());
    ONEDAL_ASSERT(means.has_mutable_data());
    ONEDAL_ASSERT(is_known_usm(q, sums.get_data()));
    ONEDAL_ASSERT(is_known_usm(q, means.get_mutable_data()));
    ONEDAL_ASSERT(sums.get_dimension(0) == means.get_dimension(0));

    const auto column_count = sums.get_dimension(0);

    const Float inv_n = Float(1.0 / double(row_count));

    const Float* sums_ptr = sums.get_data();
    Float* means_ptr = means.get_mutable_data();

    return q.submit([&](sycl::handler& cgh) {
        const auto range = make_range_1d(column_count);
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            const Float s = sums_ptr[idx];
            means_ptr[idx] = inv_n * s;
        });
    });
}

template <typename Float>
inline sycl::event compute_covariance(sycl::queue& q,
                                      std::int64_t row_count,
                                      const ndview<Float, 1>& sums,
                                      ndview<Float, 2>& cov,
                                      bool bias,
                                      const event_vector& deps) {
    ONEDAL_ASSERT(sums.has_data());
    ONEDAL_ASSERT(cov.has_mutable_data());
    ONEDAL_ASSERT(cov.get_dimension(0) == cov.get_dimension(1), "Covariance matrix must be square");
    ONEDAL_ASSERT(is_known_usm(q, sums.get_data()));
    ONEDAL_ASSERT(is_known_usm(q, cov.get_mutable_data()));

    const std::int64_t n = row_count;
    const std::int64_t p = sums.get_count();
    const Float inv_n = Float(1.0 / double(n));
    const Float inv_n1 = (n > 1) ? Float(1.0 / double(n - 1)) : Float(1);
    const Float multiplier = bias ? inv_n : inv_n1;
    const Float* sums_ptr = sums.get_data();
    Float* cov_ptr = cov.get_mutable_data();

    return q.submit([&](sycl::handler& cgh) {
        const auto range = make_range_2d(p, p);

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::item<2> id) {
            const std::int64_t gi = id.get_linear_id();
            const std::int64_t i = id.get_id(0);
            const std::int64_t j = id.get_id(1);

            if (i < p && j < p) {
                cov_ptr[gi] -= inv_n * sums_ptr[i] * sums_ptr[j];
                cov_ptr[gi] *= multiplier;
            }
        });
    });
}

template <typename Float>
sycl::event covariance(sycl::queue& q,
                       std::int64_t row_count,
                       const ndview<Float, 1>& sums,
                       ndview<Float, 2>& cov,
                       bool bias,
                       const event_vector& deps) {
    ONEDAL_ASSERT(sums.has_data());
    ONEDAL_ASSERT(cov.has_mutable_data());
    ONEDAL_ASSERT(cov.get_dimension(0) == cov.get_dimension(1), "Covariance matrix must be square");
    ONEDAL_ASSERT(is_known_usm(q, sums.get_data()));
    ONEDAL_ASSERT(is_known_usm(q, cov.get_mutable_data()));

    auto compute_event = compute_covariance(q, row_count, sums, cov, bias, deps);
    return compute_event;
}

template <typename Float>
sycl::event variances(sycl::queue& q,
                      const ndview<Float, 2>& cov,
                      ndview<Float, 1>& vars,
                      const event_vector& deps) {
    ONEDAL_ASSERT(cov.has_data());
    ONEDAL_ASSERT(vars.has_mutable_data());
    ONEDAL_ASSERT(cov.get_dimension(0) == cov.get_dimension(1), "Covariance matrix must be square");
    ONEDAL_ASSERT(is_known_usm(q, cov.get_data()));
    ONEDAL_ASSERT(is_known_usm(q, vars.get_mutable_data()));

    const auto p = cov.get_dimension(0);
    const Float* cov_ptr = cov.get_data();
    Float* vars_ptr = vars.get_mutable_data();

    return q.submit([&](sycl::handler& cgh) {
        const auto range = dal::backend::make_range_1d(p);

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            vars_ptr[idx] = cov_ptr[idx * p + idx];
        });
    });
}
template <typename Float>
inline sycl::event prepare_correlation(sycl::queue& q,
                                       std::int64_t row_count,
                                       const ndview<Float, 1>& sums,
                                       const ndview<Float, 2>& corr,
                                       ndview<Float, 1>& tmp,
                                       const event_vector& deps) {
    ONEDAL_ASSERT(sums.has_data());
    ONEDAL_ASSERT(corr.has_data());
    ONEDAL_ASSERT(tmp.has_mutable_data());
    ONEDAL_ASSERT(corr.get_dimension(0) == corr.get_dimension(1),
                  "Correlation matrix must be square");
    ONEDAL_ASSERT(is_known_usm(q, sums.get_data()));
    ONEDAL_ASSERT(is_known_usm(q, corr.get_data()));
    ONEDAL_ASSERT(is_known_usm(q, tmp.get_mutable_data()));

    const auto n = row_count;
    const auto p = sums.get_count();
    const Float inv_n = Float(1.0 / double(n));

    const Float* sums_ptr = sums.get_data();
    const Float* corr_ptr = corr.get_data();

    Float* tmp_ptr = tmp.get_mutable_data();

    const Float eps = std::numeric_limits<Float>::epsilon();

    return q.submit([&](sycl::handler& cgh) {
        const auto range = dal::backend::make_range_1d(p);

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            const Float s = sums_ptr[idx];
            const Float m = inv_n * s * s;
            const Float c = corr_ptr[idx * p + idx];
            const Float v = c - m;

            // If $Var[x_i] > 0$ is close to zero, add $\varepsilon$
            // to avoid NaN/Inf in the resulting correlation matrix
            tmp_ptr[idx] = v + eps * Float(v < eps);
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
    ONEDAL_ASSERT(corr.has_mutable_data());
    ONEDAL_ASSERT(tmp.has_mutable_data());
    ONEDAL_ASSERT(is_known_usm(q, corr.get_mutable_data()));
    ONEDAL_ASSERT(is_known_usm(q, tmp.get_mutable_data()));

    const auto n = row_count;
    const auto p = sums.get_count();
    const Float inv_n = Float(1.0 / double(n));

    const Float* sums_ptr = sums.get_data();
    const Float* tmp_ptr = tmp.get_mutable_data();
    Float* corr_ptr = corr.get_mutable_data();

    return q.submit([&](sycl::handler& cgh) {
        const auto range = make_range_2d(p, p);

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<2> idx) {
            const std::int64_t i = idx[0];
            const std::int64_t j = idx[1];
            const std::int64_t gi = i * p + j;

            const Float is_diag = Float(i == j);

            Float c = corr_ptr[gi];
            c -= inv_n * sums_ptr[i] * sums_ptr[j];
            c *= sycl::rsqrt(tmp_ptr[i] * tmp_ptr[j]);
            corr_ptr[gi] = c * (Float(1.0) - is_diag) + is_diag;
        });
    });
}

template <typename Float>
sycl::event correlation(sycl::queue& q,
                        std::int64_t row_count,
                        const ndview<Float, 1>& sums,
                        ndview<Float, 2>& corr,
                        ndview<Float, 1>& tmp,
                        const event_vector& deps) {
    ONEDAL_ASSERT(sums.has_data());
    ONEDAL_ASSERT(corr.has_mutable_data());
    ONEDAL_ASSERT(tmp.has_mutable_data());
    ONEDAL_ASSERT(corr.get_dimension(0) == corr.get_dimension(1),
                  "Correlation matrix must be square");
    ONEDAL_ASSERT(is_known_usm(q, sums.get_data()));
    ONEDAL_ASSERT(is_known_usm(q, corr.get_mutable_data()));
    ONEDAL_ASSERT(is_known_usm(q, tmp.get_mutable_data()));

    auto prepare_event = prepare_correlation(q, row_count, sums, corr, tmp, deps);
    auto finalize_event = finalize_correlation(q, row_count, sums, tmp, corr, { prepare_event });
    return finalize_event;
}

template <typename Float>
inline sycl::event prepare_correlation_from_covariance(sycl::queue& q,
                                                       std::int64_t row_count,
                                                       const ndview<Float, 2>& cov,
                                                       ndview<Float, 1>& tmp,
                                                       bool bias,
                                                       const event_vector& deps) {
    ONEDAL_ASSERT(cov.has_data());
    ONEDAL_ASSERT(tmp.has_mutable_data());
    ONEDAL_ASSERT(cov.get_dimension(0) == cov.get_dimension(1), "Covariance matrix must be square");
    ONEDAL_ASSERT(is_known_usm(q, cov.get_data()));
    ONEDAL_ASSERT(is_known_usm(q, tmp.get_mutable_data()));

    const auto n = row_count;
    const auto p = cov.get_dimension(1);
    const Float unbiased_multiplier = (n > 1) ? Float(n - 1) : Float(1);
    const Float biased_multiplier = Float(n);
    const Float multiplier = bias ? biased_multiplier : unbiased_multiplier;

    const Float* cov_ptr = cov.get_data();

    Float* tmp_ptr = tmp.get_mutable_data();

    const Float eps = std::numeric_limits<Float>::epsilon();

    return q.submit([&](sycl::handler& cgh) {
        const auto range = dal::backend::make_range_1d(p);

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            Float c = cov_ptr[idx * p + idx] * multiplier;

            // If $Var[x_i] > 0$ is close to zero, add $\varepsilon$
            // to avoid NaN/Inf in the resulting correlation matrix
            tmp_ptr[idx] = c + eps * Float(c < eps);
        });
    });
}

template <typename Float>
inline sycl::event finalize_correlation_from_covariance(sycl::queue& q,
                                                        std::int64_t row_count,
                                                        const ndview<Float, 2>& cov,
                                                        const ndview<Float, 1>& tmp,
                                                        ndview<Float, 2>& corr,
                                                        bool bias,
                                                        const event_vector& deps) {
    ONEDAL_ASSERT(cov.has_data());
    ONEDAL_ASSERT(corr.has_mutable_data());
    ONEDAL_ASSERT(tmp.has_data());
    ONEDAL_ASSERT(corr.get_dimension(0) == corr.get_dimension(1),
                  "Correlation matrix must be square");
    ONEDAL_ASSERT(cov.get_dimension(0) == cov.get_dimension(1), "Covariance matrix must be square");
    ONEDAL_ASSERT(is_known_usm(q, corr.get_mutable_data()));
    ONEDAL_ASSERT(is_known_usm(q, cov.get_data()));
    ONEDAL_ASSERT(is_known_usm(q, tmp.get_data()));

    const auto n = row_count;
    const auto p = cov.get_dimension(1);
    const Float unbiased_multiplier = (n > 1) ? Float(n - 1) : Float(1);
    const Float biased_multiplier = Float(n);
    const Float multiplier = bias ? biased_multiplier : unbiased_multiplier;
    const Float* tmp_ptr = tmp.get_data();
    Float* corr_ptr = corr.get_mutable_data();
    const Float* cov_ptr = cov.get_data();
    return q.submit([&](sycl::handler& cgh) {
        const auto range = make_range_2d(p, p);

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<2> idx) {
            const std::int64_t i = idx[0];
            const std::int64_t j = idx[1];
            const std::int64_t gi = i * p + j;
            const Float is_diag = Float(i == j);
            Float c = cov_ptr[gi] * multiplier * sycl::rsqrt(tmp_ptr[i] * tmp_ptr[j]);
            corr_ptr[gi] = c * (Float(1.0) - is_diag) + is_diag;
        });
    });
}

template <typename Float>
sycl::event correlation_from_covariance(sycl::queue& q,
                                        std::int64_t row_count,
                                        const ndview<Float, 2>& cov,
                                        ndview<Float, 2>& corr,
                                        ndview<Float, 1>& tmp,
                                        bool bias,
                                        const event_vector& deps) {
    ONEDAL_ASSERT(cov.has_mutable_data());
    ONEDAL_ASSERT(corr.has_mutable_data());
    ONEDAL_ASSERT(tmp.has_mutable_data());
    ONEDAL_ASSERT(corr.get_dimension(0) == corr.get_dimension(1),
                  "Correlation matrix must be square");
    ONEDAL_ASSERT(cov.get_dimension(0) == cov.get_dimension(1), "Covariance matrix must be square");
    ONEDAL_ASSERT(is_known_usm(q, corr.get_mutable_data()));
    ONEDAL_ASSERT(is_known_usm(q, cov.get_mutable_data()));
    ONEDAL_ASSERT(is_known_usm(q, tmp.get_mutable_data()));

    auto prepare_event = prepare_correlation_from_covariance(q, row_count, cov, tmp, bias, deps);
    auto finalize_event =
        finalize_correlation_from_covariance(q, row_count, cov, tmp, corr, bias, { prepare_event });
    finalize_event.wait_and_throw();
    return finalize_event;
}

#define INSTANTIATE_MEANS(F)                                         \
    template ONEDAL_EXPORT sycl::event means<F>(sycl::queue&,        \
                                                std::int64_t,        \
                                                const ndview<F, 1>&, \
                                                ndview<F, 1>&,       \
                                                const event_vector&);

INSTANTIATE_MEANS(float)
INSTANTIATE_MEANS(double)

#define INSTANTIATE_COV(F)                                                \
    template ONEDAL_EXPORT sycl::event covariance<F>(sycl::queue&,        \
                                                     std::int64_t,        \
                                                     const ndview<F, 1>&, \
                                                     ndview<F, 2>&,       \
                                                     bool,                \
                                                     const event_vector&);

INSTANTIATE_COV(float)
INSTANTIATE_COV(double)

#define INSTANTIATE_COR_FROM_COV(F)                                                        \
    template ONEDAL_EXPORT sycl::event correlation_from_covariance<F>(sycl::queue&,        \
                                                                      std::int64_t,        \
                                                                      const ndview<F, 2>&, \
                                                                      ndview<F, 2>&,       \
                                                                      ndview<F, 1>&,       \
                                                                      bool,                \
                                                                      const event_vector&);

INSTANTIATE_COR_FROM_COV(float)
INSTANTIATE_COR_FROM_COV(double)

#define INSTANTIATE_COR(F)                                                 \
    template ONEDAL_EXPORT sycl::event correlation<F>(sycl::queue&,        \
                                                      std::int64_t,        \
                                                      const ndview<F, 1>&, \
                                                      ndview<F, 2>&,       \
                                                      ndview<F, 1>&,       \
                                                      const event_vector&);

INSTANTIATE_COR(float)
INSTANTIATE_COR(double)

#define INSTANTIATE_VARS(F)                                              \
    template ONEDAL_EXPORT sycl::event variances<F>(sycl::queue&,        \
                                                    const ndview<F, 2>&, \
                                                    ndview<F, 1>&,       \
                                                    const event_vector&);

INSTANTIATE_VARS(float)
INSTANTIATE_VARS(double)

} // namespace oneapi::dal::backend::primitives
