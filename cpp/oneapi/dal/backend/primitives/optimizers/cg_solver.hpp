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
#include "oneapi/dal/backend/primitives/optimizers/common.hpp"

namespace oneapi::dal::backend::primitives {

// Method of Conjugate Gradients for Solving Linear Systems
// https://nvlpubs.nist.gov/nistpubs/jres/049/jresv49n6p409_a1b.pdf
template <typename Float>
std::pair<sycl::event, std::int64_t> cg_solve(sycl::queue& queue,
                                              base_matrix_operator<Float>& mul_operator,
                                              const ndview<Float, 1>& b,
                                              ndview<Float, 1>& x,
                                              ndview<Float, 1>& residual,
                                              ndview<Float, 1>& conj_vector,
                                              ndview<Float, 1>& buffer,
                                              Float tol = 1.0e-5,
                                              Float atol = -1.0,
                                              std::int64_t maxiter = 100l,
                                              const event_vector& deps = {});

} // namespace oneapi::dal::backend::primitives
