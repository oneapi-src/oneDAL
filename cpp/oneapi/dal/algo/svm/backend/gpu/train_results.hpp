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

namespace oneapi::dal::svm::backend {

namespace pr = dal::backend::primitives;

template <typename Float>
sycl::event check_coeffs_border(sycl::queue& q,
                                const pr::ndview<Float, 1> coeffs,
                                pr::ndview<std::uint8_t, 1>& indicator,
                                const Float C,
                                const dal::backend::event_vector& deps = {}) {
    ONEDAL_ASSERT(indicator.has_mutable_data());
    ONEDAL_ASSERT(indicator.get_dimension(0) == coeffs.get_dimension(0));

    const Float* coeffs_ptr = coeffs.get_data();
    std::uint8_t indicator_ptr = indicator.get_mutable_data();

    return q.submit([&](sycl::handler& cgh) {
        const auto range = dal::backend::make_range_1d(data.get_count());
        cgh.depends_on(deps);

        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            const Float coeffs_i = coeffs_ptr[idx];
            indicator_ptr[idx] = 0 < coeffs_i && coeffs_i < C;
        });
    });
}

template <typename Float>
Float compute_bias(const sycl::queue& queue,
                   const pr::ndview<Float, 1>& labels,
                   const pr::ndview<Float, 1> f,
                   const pr::ndview<Float, 1> coeffs,
                   pr::ndview<Float, 1> tmp_values,
                   pr::ndview<std::uint8_t, 1> indicator,
                   const Float C,
                   const std::int64_t row_count) {
    Float bias = 0;
    auto reduce_res = pr::ndarray<Float, 1>::empty(queue, { 1 }, sycl::usm::alloc::device);

    /* free SV: (0 < coeffs < C)*/
    std::int64_t free_sv_count = 0;
    auto check_coeffs_border_event = check_coeffs_border(queue, coeffs, indicator, C);
    auto select_flagged = pr::select_flagged<Float, std::uint8_t>{ queue_ };
    select_flagged(indicator, f, tmp_values, free_sv_count, { check_coeffs_border_event })
        .wait_and_throw();
    if (free_sv_count > 0) {
        auto reduce_event = pr::reduce_by_rows(queue,
                                               tmp_values,
                                               reduce_res,
                                               pr::sum<Float>{},
                                               pr::identity<Float>{});
        auto reduce_res_host = reduce_res.to_host(queue, { reduce_event }).flatten();
        bias = -*reduce_res_host / Float(free_sv_count);
    }
    else {
        Float ub = dal::detail::limits<Float>::min();
        check_edge(queue, labels, coeffs, indicator)


        Float lb = dal::detail::limits<Float>::max();
    }
}

template <typename Float>
auto compute_train_results(const sycl::queue& queue,
                           const pr::ndview<Float, 2>& x,
                           const pr::ndview<Float, 1>& labels,
                           const pr::ndview<Float, 1> f,
                           const pr::ndview<Float, 1> coeffs,
                           const Float C,
                           const std::int64_t row_count) {
    ONEDAL_ASSERT(row_count > 0);
    ONEDAL_ASSERT(row_count <= dal::detail::limits<std::uint32_t>::max());
    auto tmp_values = pr::ndarray<Float, 1>::empty(queue, { row_count }, sycl::usm::alloc::device);
    auto indicator =
        pr::ndarray<std::uint8_t, 1>::empty(queue, { row_count }, sycl::usm::alloc::device);

    Float bias = compute_bias(const sycl::queue& queue,
                              const pr::ndview<Float, 1>& labels,
                              const pr::ndview<Float, 1> f,
                              const pr::ndview<Float, 1> coeffs,
                              const pr::ndview<Float, 1> tmp_values,
                              const pr::ndview<std::uint8_t, 1> indicator,
                              const Float C,
                              const std::int64_t row_count);
}

// template <typename Float>
// class train_result {
// public:
//     train_result(const sycl::queue& queue,
//                  const pr::ndarray<Float, 1>& labels,
//                  const pr::ndarray<Float, 1> f,
//                  const pr::ndarray<Float, 1> coeffs,
//                  const Float C,
//                  const std::int64_t row_count)
//             : queue_(queue),
//               row_count_(row_count),
//               C_(C),
//               labels_(labels),
//               f_(f),
//               coeffs_(coeffs) {
//         ONEDAL_ASSERT(row_count > 0);
//         ONEDAL_ASSERT(row_count <= dal::detail::limits<std::uint32_t>::max());
//         tmp_values_ =
//             pr::ndarray<Float, 1>::empty(queue_, { row_count_ }, sycl::usm::alloc::device);
//         mask_ =
//             pr::ndarray<std::uint8_t, 1>::empty(queue_, { row_count_ }, sycl::usm::alloc::device);
//     }
//     sycl::event compute_classification_coeffs(sycl::queue& queue, ) {}

// private:
//     sycl::queue queue_;

//     const std::int64_t row_count_;
//     const Float C_;

//     pr::ndarray<Float, 1> labels_;
//     pr::ndarray<Float, 1> f_;
//     pr::ndarray<Float, 1> coeffs_;
//     pr::ndarray<Float, 1> tmp_values_;
//     pr::ndarray<std::uint8_t, 1> mask_;
// }

} // namespace oneapi::dal::svm::backend