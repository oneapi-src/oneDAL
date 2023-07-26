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

// Nocedal, J, and S J Wright. 2006. Numerical Optimization. Springer New York.
// pp. 168 (also known as the truncated Newton method)
// https://link.springer.com/book/10.1007/978-0-387-40065-5
template <typename Float>
sycl::event newton_cg(sycl::queue& queue,
                      BaseFunction<Float>& f,
                      ndview<Float, 1>& x,
                      Float tol = 1.0e-5,
                      std::int64_t maxiter = 100l,
                      const event_vector& deps = {});

} // namespace oneapi::dal::backend::primitives
