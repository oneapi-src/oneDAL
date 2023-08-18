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

#pragma once

#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
sycl::event compute_probabilities(sycl::queue& q,
                                  const ndview<Float, 1>& parameters,
                                  const ndview<Float, 2>& data,
                                  ndview<Float, 1>& predictions,
                                  bool fit_intercept = true,
                                  const event_vector& deps = {});

template <typename Float>
sycl::event compute_logloss(sycl::queue& q,
                            const ndview<Float, 1>& parameters,
                            const ndview<Float, 2>& data,
                            const ndview<std::int32_t, 1>& labels,
                            ndview<Float, 1>& out,
                            Float L1 = Float(0),
                            Float L2 = Float(0),
                            bool fit_intercept = true,
                            const event_vector& deps = {});

template <typename Float>
sycl::event compute_logloss(sycl::queue& q,
                            const ndview<Float, 1>& parameters,
                            const ndview<Float, 2>& data,
                            const ndview<std::int32_t, 1>& labels,
                            const ndview<Float, 1>& probabilities,
                            ndview<Float, 1>& out,
                            Float L1 = Float(0),
                            Float L2 = Float(0),
                            bool fit_intercept = true,
                            const event_vector& deps = {});

template <typename Float>
sycl::event compute_logloss_with_der(sycl::queue& q,
                                     const ndview<Float, 1>& parameters,
                                     const ndview<Float, 2>& data,
                                     const ndview<std::int32_t, 1>& labels,
                                     const ndview<Float, 1>& probabilities,
                                     ndview<Float, 1>& out,
                                     ndview<Float, 1>& out_derivative,
                                     Float L1 = Float(0),
                                     Float L2 = Float(0),
                                     bool fit_intercept = true,
                                     const event_vector& deps = {});

template <typename Float>
sycl::event compute_derivative(sycl::queue& q,
                               const ndview<Float, 1>& parameters,
                               const ndview<Float, 2>& data,
                               const ndview<std::int32_t, 1>& labels,
                               const ndview<Float, 1>& probabilities,
                               ndview<Float, 1>& out_derivative,
                               Float L1 = Float(0),
                               Float L2 = Float(0),
                               bool fit_intercept = true,
                               const event_vector& deps = {});

template <typename Float>
sycl::event compute_hessian(sycl::queue& q,
                            const ndview<Float, 1>& parameters,
                            const ndview<Float, 2>& data,
                            const ndview<std::int32_t, 1>& labels,
                            const ndview<Float, 1>& probabilities,
                            ndview<Float, 2>& out_hessian,
                            Float L1 = Float(0),
                            Float L2 = Float(0),
                            bool fit_intercept = true,
                            const event_vector& deps = {});

template <typename Float>
sycl::event compute_raw_hessian(sycl::queue& q,
                                const ndview<Float, 1>& probabilities,
                                ndview<Float, 1>& out_hessian,
                                const event_vector& deps = {});

template <typename Float>
class logloss_hessian_product {
public:
    logloss_hessian_product(sycl::queue& q,
                            const ndview<Float, 2>& data,
                            const Float L2 = Float(0),
                            const bool fit_intercept = true);

    sycl::event set_raw_hessian(const ndview<Float, 1>& raw_hessian, const event_vector& deps = {});

    ndview<Float, 1>& get_raw_hessian();
    sycl::event operator()(const ndview<Float, 1>& vec,
                           ndview<Float, 1>& out,
                           const event_vector& deps = {});

private:
    sycl::event compute_with_fit_intercept(const ndview<Float, 1>& vec,
                                           ndview<Float, 1>& out,
                                           const event_vector& deps);
    sycl::event compute_without_fit_intercept(const ndview<Float, 1>& vec,
                                              ndview<Float, 1>& out,
                                              const event_vector& deps);

    sycl::queue q_;
    ndarray<Float, 1> raw_hessian_;
    const ndview<Float, 2> data_;
    ndarray<Float, 1> buffer_;
    const Float L2_;
    const bool fit_intercept_;
    const std::int64_t n_;
    const std::int64_t p_;
};

} // namespace oneapi::dal::backend::primitives
