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

#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/optimizers/common.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
sycl::event compute_probabilities(sycl::queue& q,
                                  const ndview<Float, 1>& parameters,
                                  const ndview<Float, 2>& data,
                                  ndview<Float, 1>& probabilities,
                                  bool fit_intercept = true,
                                  const event_vector& deps = {});

template <typename Float>
sycl::event compute_logloss(sycl::queue& q,
                            const ndview<std::int32_t, 1>& labels,
                            const ndview<Float, 1>& probabilities,
                            ndview<Float, 1>& out,
                            bool fit_intercept = true,
                            const event_vector& deps = {});

template <typename Float>
sycl::event compute_logloss_with_der(sycl::queue& q,
                                     const ndview<Float, 2>& data,
                                     const ndview<std::int32_t, 1>& labels,
                                     const ndview<Float, 1>& probabilities,
                                     ndview<Float, 1>& out,
                                     ndview<Float, 1>& out_derivative,
                                     bool fit_intercept = true,
                                     const event_vector& deps = {});

template <typename Float>
sycl::event compute_derivative(sycl::queue& q,
                               const ndview<Float, 2>& data,
                               const ndview<std::int32_t, 1>& labels,
                               const ndview<Float, 1>& probabilities,
                               ndview<Float, 1>& out_derivative,
                               bool fit_intercept = true,
                               const event_vector& deps = {});

template <typename Float>
sycl::event add_regularization_loss(sycl::queue& q,
                                    const ndview<Float, 1>& parameters,
                                    ndview<Float, 1>& out,
                                    Float L1 = Float(0),
                                    Float L2 = Float(0),
                                    bool fit_intercept = true,
                                    const event_vector& deps = {});

template <typename Float>
sycl::event add_regularization_gradient_loss(sycl::queue& q,
                                             const ndview<Float, 1>& parameters,
                                             ndview<Float, 1>& out,
                                             ndview<Float, 1>& out_derivative,
                                             Float L1 = Float(0),
                                             Float L2 = Float(0),
                                             bool fit_intercept = true,
                                             const event_vector& deps = {});

template <typename Float>
sycl::event add_regularization_gradient(sycl::queue& q,
                                        const ndview<Float, 1>& parameters,
                                        ndview<Float, 1>& out_derivative,
                                        Float L1 = Float(0),
                                        Float L2 = Float(0),
                                        bool fit_intercept = true,
                                        const event_vector& deps = {});

template <typename Float>
sycl::event compute_hessian(sycl::queue& q,
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
class LogLossHessianProduct : public BaseMatrixOperator<Float> {
public:
    LogLossHessianProduct(sycl::queue& q,
                          table& data,
                          Float L2 = Float(0),
                          bool fit_intercept = true);
    sycl::event operator()(const ndview<Float, 1>& vec,
                           ndview<Float, 1>& out,
                           const event_vector& deps) final;
    ndview<Float, 1>& get_raw_hessian();

private:
    sycl::event compute_with_fit_intercept(const ndview<Float, 1>& vec,
                                           ndview<Float, 1>& out,
                                           const event_vector& deps);
    sycl::event compute_without_fit_intercept(const ndview<Float, 1>& vec,
                                              ndview<Float, 1>& out,
                                              const event_vector& deps);

    sycl::queue q_;
    table& data_;
    Float L2_;
    bool fit_intercept_;
    ndarray<Float, 1> raw_hessian_;
    ndarray<Float, 1> buffer_;
    const std::int64_t n_;
    const std::int64_t p_;
};

template <typename Float>
class LogLossFunction : public BaseFunction<Float> {
public:
    LogLossFunction(sycl::queue queue,
                    table& data,
                    ndview<std::int32_t, 1>& labels,
                    Float L2 = 0.0,
                    bool fit_intercept = true);
    Float get_value() final;
    ndview<Float, 1>& get_gradient() final;
    BaseMatrixOperator<Float>& get_hessian_product() final;

    event_vector update_x(const ndview<Float, 1>& x,
                          bool need_hessp = false,
                          const event_vector& deps = {}) final;

private:
    sycl::queue q_;
    table data_;
    ndview<std::int32_t, 1> labels_;
    const std::int64_t n_;
    const std::int64_t p_;
    Float L2_;
    bool fit_intercept_;
    const std::int64_t bsz_;
    ndarray<Float, 1> probabilities_;
    ndarray<Float, 1> gradient_;
    ndarray<Float, 1> buffer_;
    LogLossHessianProduct<Float> hessp_;
    const std::int64_t dimension_;
    Float value_;
};

} // namespace oneapi::dal::backend::primitives
