/*******************************************************************************
* Copyright 2022 Intel Corporation
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
sycl::event compute_predictions(sycl::queue& q,
                                const ndview<Float, 1>& parameters,
                                const ndview<Float, 2>& data,
                                ndview<Float, 1>& predictions,
                                const event_vector& deps = {});

template <typename Float>
Float compute_logloss(sycl::queue& q,
                      const ndview<Float, 1>& parameters,
                      const ndview<Float, 2>& data,
                      const ndview<Float, 1>& labels,
                      const event_vector& deps = {});

template <typename Float>
Float compute_logloss_with_regularization(sycl::queue& q,
                                          const ndview<Float, 1>& parameters,
                                          const ndview<Float, 2>& data,
                                          const ndview<Float, 1>& labels,
                                          const Float L1 = Float(0),
                                          const Float L2 = Float(0),
                                          const Float tol = float(1e-7),
                                          const event_vector& deps = {});

} // namespace oneapi::dal::backend::primitives
