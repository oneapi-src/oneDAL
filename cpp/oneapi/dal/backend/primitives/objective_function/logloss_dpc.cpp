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

#include "oneapi/dal/backend/primitives/objective_function/logloss.hpp"
#include "oneapi/dal/backend/primitives/blas/gemv.hpp"
#include "oneapi/dal/backend/primitives/loops.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include <sycl/ext/oneapi/experimental/builtins.hpp>

namespace oneapi::dal::backend::primitives {

template <typename Float>
sycl::event compute_predictions(sycl::queue& q,
                                const ndview<Float, 1>& parameters,
                                const ndview<Float, 2>& data,
                                ndview<Float, 1>& predictions,
                                const event_vector& deps) {
    auto fill_event = fill<Float>(q, predictions, Float(1), {});
    return gemv(data,
                parameters.get_slice(1, parameters.get_dimesion(0)),
                predictions,
                Float(1),
                parameters.at(0),
                { fill_event });
}

template <typename Float>
Float compute_logloss(sycl::queue& q,
                      const ndview<Float, 1>& parameters,
                      const ndview<Float, 2>& data,
                      const ndview<Float, 1>& labels,
                      const event_vector& deps) {
    const int64_t n = data.get_dimension(0);
    const int64_t p = data.get_dimension(1);
    ONEDAL_ASSERT(parameters.get_dimension(0) == p + 1);
    ONEDAL_ASSERT(labels.get_dimension(0) == n);
    ONEDAL_ASSERT(labels.has_data());
    ONEDAL_ASSERT(parameters.has_data());
    ONEDAL_ASSERT(data.has_data());
    auto losses = ndarray<float_t, 1>::empty(q, { n });
    auto prediction_event = compute_predictions(q, parameters, data, losses, deps);

    const Float* labels_ptr = labels.get_data();
    Float* loss_ptr = losses.get_data();

    auto [out, out_event] =
        ndarray<Float, 1>::full(q, std::int64_t(1), Float(0), sycl::usm::alloc::device);

    auto out_ptr = out.get_mutable_data();

    auto loss_event = q.submit([&](sycl::handler& cgh) {
        const auto range = make_range_1d(n);
        // cgh.depends_on(deps); unnessary because prediction event already as dependance on deps
        using oneapi::dal::backend::operator+;

        const event_vector& full_deps = deps + out_event + prediction_event;
        cgh.depends_on(full_deps);

        auto sumReduction = reduction(out_ptr, cgh, sycl::plus<>());

        // if y in {-1, 1} then loss function would be log(1 + exp(-y * pred))
        // so we need to compute log(1 + exp(-(y * 2 - 1) * pred)) as our original labels are in {0, 1}

        cgh.parallel_for(range, sumReduction, [=](sycl::id<1> idx, auto& sum) {
            const Float pred = loss_ptr[idx];
            const Float label = labels_ptr[idx] * 2 - 1;
            sum += sycl::log(1 + sycl::exp(-label * pred));
        });
    });

    return out_ptr.to_host(q, { loss_event }).at(0);
}

template <typename Float>
Float compute_logloss_with_regularization(sycl::queue& q,
                                          const ndview<Float, 1>& parameters,
                                          const ndview<Float, 2>& data,
                                          const ndview<Float, 1>& labels,
                                          const Float L1,
                                          const Float L2,
                                          const Float tol,
                                          const event_vector& deps) {
    Float L1_norm = Float(0);
    Float L2_norm = Float(0);
    if (abs(L1 - Float(0)) > tol) {
    }
}

} // namespace oneapi::dal::backend::primitives
