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

// Decrease alpha while Armicho condition is not satisfied
// Armicho condition phi(alpha) <= phi(0) + c1 * alpha * phi'(0)
// phi(alpha) = f(x + alpha * d)
// phi'(alpha) = <f'(x + alpha * d), d>
/// @brief
/// @tparam Float
/// @param queue - sycl::queue
/// @param f - function with update_x, get_value, get_gradient methods
/// @param x - current point
/// @param direction - descent direction
/// @param result - x + alpha * direction will be written here
/// @param alpha - initial constant, should be equal to 1.0 for newton methods
/// @param c1 - constant for Armicho condition
/// @param x0_initialized - if function f is computed at point x
/// @param deps
/// @return
template <typename Float>
Float backtracking(sycl::queue queue,
                   BaseFunction<Float>& f,
                   const ndview<Float, 1>& x,
                   const ndview<Float, 1>& direction,
                   ndview<Float, 1>& result,
                   Float alpha = 1.0,
                   Float c1 = 1.0e-4,
                   bool x0_initialized = false,
                   const event_vector& deps = {});

} // namespace oneapi::dal::backend::primitives
