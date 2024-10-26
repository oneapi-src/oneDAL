/*******************************************************************************
* Copyright contributors to the oneDAL project
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
#include "oneapi/dal/backend/communicator.hpp"
#include "oneapi/dal/backend/primitives/sparse_blas/handle.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
sycl::event compute_probabilities(sycl::queue& q,
                                  const ndview<Float, 1>& parameters,
                                  const ndview<Float, 2>& data,
                                  ndview<Float, 1>& probabilities,
                                  bool fit_intercept = true,
                                  const event_vector& deps = {});

template <typename Float>
sycl::event compute_probabilities_sparse(sycl::queue& q,
                                         const ndview<Float, 1>& parameters,
                                         sparse_matrix_handle& sp_handler,
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
sycl::event compute_logloss_with_der_sparse(sycl::queue& q,
                                            sparse_matrix_handle& sp_handler,
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

} // namespace oneapi::dal::backend::primitives
