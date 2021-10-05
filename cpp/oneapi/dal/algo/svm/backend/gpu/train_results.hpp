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

#pragma once

#include "oneapi/dal/backend/primitives/ndarray.hpp"
// #include "oneapi/dal/backend/primitives/common.hpp" // ?
#include "oneapi/dal/backend/primitives/selection/select_flagged.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/algo/svm/backend/gpu/misc.hpp"

namespace oneapi::dal::svm::backend {

namespace pr = dal::backend::primitives;

template <typename Float>
sycl::event check_coeffs_border(sycl::queue& q,
                                const pr::ndview<Float, 1>& coeffs,
                                pr::ndview<std::uint8_t, 1>& indicator,
                                const Float C,
                                const dal::backend::event_vector& deps = {}) {
    ONEDAL_ASSERT(indicator.has_mutable_data());
    ONEDAL_ASSERT(indicator.get_dimension(0) == coeffs.get_dimension(0));

    const Float* coeffs_ptr = coeffs.get_data();
    std::uint8_t* indicator_ptr = indicator.get_mutable_data();

    return q.submit([&](sycl::handler& cgh) {
        const auto range = dal::backend::make_range_1d(coeffs.get_count());
        cgh.depends_on(deps);

        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            const Float coeffs_i = coeffs_ptr[idx];
            indicator_ptr[idx] = 0 < coeffs_i && coeffs_i < C;
        });
    });
}

template <typename Float>
Float compute_bias(sycl::queue& q,
                   const pr::ndview<Float, 1>& labels,
                   const pr::ndview<Float, 1>& f,
                   const pr::ndview<Float, 1>& coeffs,
                   pr::ndview<Float, 1>& tmp_values,
                   pr::ndview<std::uint8_t, 1>& indicator,
                   const Float C) {
    Float bias = 0;
    auto reduce_res = pr::ndarray<Float, 1>::empty(q, { 1 }, sycl::usm::alloc::device);

    /* free SV: (0 < coeffs < C)*/
    std::int64_t free_sv_count = 0;
    auto check_coeffs_border_event = check_coeffs_border<Float>(q, coeffs, indicator, C);
    auto select_flagged = pr::select_flagged<Float, std::uint8_t>{ q };
    select_flagged(indicator, f, tmp_values, free_sv_count, { check_coeffs_border_event })
        .wait_and_throw();
    if (free_sv_count > 0) {
        auto reduce_event = pr::reduce_by_columns(
            q,
            tmp_values.reshape(pr::ndshape<2>{ tmp_values.get_dimension(0), 1 }),
            reduce_res,
            pr::sum<Float>{},
            pr::identity<Float>{});
        auto reduce_res_host = reduce_res.to_host(q, { reduce_event }).flatten();
        bias = -*reduce_res_host.get_data() / Float(free_sv_count);
    }
    else {
        Float ub = dal::detail::limits<Float>::min();
        auto check_edge_event =
            check_violating_edge(q, labels, coeffs, indicator, C, violating_edge::up);
        std::int64_t up_edge_count = 0;
        auto select_flagged_event =
            select_flagged(indicator, f, tmp_values, up_edge_count, { check_edge_event });
        auto reduce_event = pr::reduce_by_columns(
            q,
            tmp_values.reshape(pr::ndshape<2>{ tmp_values.get_dimension(0), 1 }),
            reduce_res,
            pr::min<Float>{},
            pr::identity<Float>{},
            { select_flagged_event });
        auto reduce_res_up_host = reduce_res.to_host(q, { reduce_event }).flatten();
        ub = *reduce_res_up_host.get_data();

        Float lb = dal::detail::limits<Float>::max();
        check_edge_event =
            check_violating_edge(q, labels, coeffs, indicator, C, violating_edge::low);
        std::int64_t low_edge_count = 0;
        select_flagged_event =
            select_flagged(indicator, f, tmp_values, low_edge_count, { check_edge_event });
        reduce_event = pr::reduce_by_columns(
            q,
            tmp_values.reshape(pr::ndshape<2>{ tmp_values.get_dimension(0), 1 }),
            reduce_res,
            pr::max<Float>{},
            pr::identity<Float>{},
            { select_flagged_event });
        auto reduce_res_low_host = reduce_res.to_host(q, { reduce_event }).flatten();
        lb = *reduce_res_low_host.get_data();

        bias = -0.5 * (ub + lb);
    }

    return bias;
}

template <typename Float>
auto compute_dual_coeffs(sycl::queue& q,
                         const pr::ndview<Float, 1>& labels,
                         pr::ndview<Float, 1>& coeffs,
                         const dal::backend::event_vector& deps = {}) {
    ONEDAL_ASSERT(labels.get_dimension(0) == coeffs.get_dimension(0));
    ONEDAL_ASSERT(coeffs.has_mutable_data());

    const Float* labels_ptr = labels.get_data();
    Float* coeffs_ptr = coeffs.get_mutable_data();

    return q.submit([&](sycl::handler& cgh) {
        const auto range = dal::backend::make_range_1d(coeffs.get_count());
        cgh.depends_on(deps);

        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            coeffs_ptr[idx] = coeffs_ptr[idx] * labels_ptr[idx];
        });
    });
}

template <typename Float>
auto check_non_zero_binary(sycl::queue& q,
                           const pr::ndview<Float, 1>& coeffs,
                           pr::ndview<std::uint8_t, 1>& indicator,
                           const dal::backend::event_vector& deps = {}) {
    ONEDAL_ASSERT(indicator.get_dimension(0) == coeffs.get_dimension(0));
    ONEDAL_ASSERT(indicator.has_mutable_data());

    const Float* coeffs_ptr = coeffs.get_data();
    std::uint8_t* indicator_ptr = indicator.get_mutable_data();

    return q.submit([&](sycl::handler& cgh) {
        const auto range = dal::backend::make_range_1d(indicator.get_count());
        cgh.depends_on(deps);

        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            indicator_ptr[idx] = coeffs_ptr[idx] != Float(0);
        });
    });
}

template <typename Float>
auto compute_sv_coeffs(sycl::queue& q,
                       const pr::ndview<Float, 1>& coeffs,
                       pr::ndview<Float, 1>& tmp_values,
                       pr::ndview<std::uint8_t, 1>& indicator,
                       const dal::backend::event_vector& deps = {}) {
    auto check_non_zero_binary_event = check_non_zero_binary(q, coeffs, indicator, deps);

    std::int64_t sv_count = 0;
    auto select_flagged = pr::select_flagged<Float, std::uint8_t>{ q };
    select_flagged(indicator, coeffs, tmp_values, sv_count, { check_non_zero_binary_event })
        .wait_and_throw();

    auto sv_coeffs = pr::ndarray<Float, 1>::empty(q, { sv_count }, sycl::usm::alloc::device);

    sycl::event copy_event;
    if (sv_count != 0) {
        const Float* tmp_values_ptr = tmp_values.get_data();
        Float* sv_coeffs_ptr = sv_coeffs.get_mutable_data();
        copy_event = dal::backend::copy(q, sv_coeffs_ptr, tmp_values_ptr, sv_count);
    }

    return std::make_tuple(sv_coeffs, sv_count, copy_event);
}

template <typename Float>
auto compute_support_indices(sycl::queue& q,
                             pr::ndview<std::uint8_t, 1>& indicator,
                             const std::int64_t sv_count,
                             const dal::backend::event_vector& deps = {}) {
    auto support_indices =
        pr::ndarray<std::uint32_t, 1>::empty(q, { sv_count }, sycl::usm::alloc::device);
    auto tmp_index =
        pr::ndarray<std::uint32_t, 1>::empty(q, { sv_count }, sycl::usm::alloc::device);

    sycl::event copy_event;
    if (sv_count != 0) {
        auto row_count = indicator.get_dimension(0);
        auto tmp_range =
            pr::ndarray<std::uint32_t, 1>::empty(q, { row_count }, sycl::usm::alloc::device);
        auto arange_event = tmp_range.arange(q);

        std::int64_t check_sv_count = 0;
        auto select_flagged = pr::select_flagged_index<std::uint32_t, std::uint8_t>{ q };
        select_flagged(indicator, tmp_range, tmp_index, check_sv_count, { arange_event })
            .wait_and_throw();

        ONEDAL_ASSERT(check_sv_count == sv_count);

        const std::uint32_t* tmp_index_ptr = tmp_index.get_data();
        std::uint32_t* support_indices_ptr = support_indices.get_mutable_data();

        copy_event = dal::backend::copy(q, support_indices_ptr, tmp_index_ptr, sv_count);
    }

    return std::make_tuple(support_indices, copy_event);
}

template <typename Float>
auto compute_support_vectors(sycl::queue& q,
                             const pr::ndview<Float, 2>& x,
                             const pr::ndview<std::uint32_t, 1>& support_indices,
                             const std::int64_t sv_count,
                             const dal::backend::event_vector& deps = {}) {
    auto support_vectors =
        pr::ndarray<Float, 2>::empty(q, { sv_count, x.get_dimension(1) }, sycl::usm::alloc::device);

    sycl::event copy_by_indices_event;
    if (sv_count != 0) {
        copy_by_indices_event = copy_by_indices(q,
                                                x,
                                                support_indices,
                                                support_vectors,
                                                sv_count,
                                                x.get_dimension(1),
                                                deps);
    }
    return std::make_tuple(support_vectors, copy_by_indices_event);
}

template <typename Float>
auto compute_train_results(sycl::queue& q,
                           const pr::ndview<Float, 2>& x,
                           const pr::ndview<Float, 1>& labels,
                           const pr::ndview<Float, 1>& f,
                           pr::ndview<Float, 1>& coeffs,
                           const Float C) {
    auto row_count = labels.get_dimension(0);
    auto tmp_values = pr::ndarray<Float, 1>::empty(q, { row_count }, sycl::usm::alloc::device);
    auto indicator =
        pr::ndarray<std::uint8_t, 1>::empty(q, { row_count }, sycl::usm::alloc::device);

    Float bias = compute_bias<Float>(q, labels, f, coeffs, tmp_values, indicator, C);

    auto compute_dual_coeffs_event = compute_dual_coeffs<Float>(q, labels, coeffs);

    auto [sv_coeffs, sv_count, compute_sv_coeffs_event] =
        compute_sv_coeffs<Float>(q, coeffs, tmp_values, indicator, { compute_dual_coeffs_event });

    auto [support_indices, compute_support_indices_event] =
        compute_support_indices<Float>(q, indicator, sv_count, { compute_sv_coeffs_event });

    auto [support_vectors, compute_support_vectors_event] =
        compute_support_vectors<Float>(q,
                                       x,
                                       support_indices,
                                       sv_count,
                                       { compute_support_indices_event });

    return std::make_tuple(bias,
                           //coeffs,
                           sv_coeffs,
                           support_indices,
                           support_vectors,
                           compute_support_vectors_event);
}

} // namespace oneapi::dal::svm::backend