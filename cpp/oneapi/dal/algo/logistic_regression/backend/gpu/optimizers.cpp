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

#include "oneapi/dal/algo/logistic_regression/backend/gpu/optimizers.hpp"
#include "oneapi/dal/backend/primitives/optimizers.hpp"

namespace oneapi::dal::logistic_regression::backend {

template <typename Float>
newton_cg_optimizer<Float>::newton_cg_optimizer(double tol, std::int32_t maxiter)
        : tol_(tol),
          maxiter_(maxiter) {}

template <typename Float>
sycl::event newton_cg_optimizer<Float>::minimize(sycl::queue& q,
                                                 pr::BaseFunction<Float>& f,
                                                 pr::ndview<Float, 1>& x,
                                                 const be::event_vector& deps) {
    return pr::newton_cg(q, f, x, Float(tol_), maxiter_, deps);
}

template class optimizer_iface<float>;
template class optimizer_iface<double>;
template class newton_cg_optimizer<float>;
template class newton_cg_optimizer<double>;

} // namespace oneapi::dal::logistic_regression::backend
