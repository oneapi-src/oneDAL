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

#include "oneapi/dal/backend/primitives/optimizers/common.hpp"
#include "oneapi/dal/backend/primitives/optimizers/cg_solver.hpp"
#include "oneapi/dal/backend/primitives/optimizers/newton_cg.hpp"
#include "oneapi/dal/backend/primitives/blas/gemv.hpp"
#include "oneapi/dal/backend/primitives/element_wise.hpp"
#include "oneapi/dal/backend/primitives/objective_function/logloss.hpp"
#include "oneapi/dal/backend/primitives/blas/gemv.hpp"
#include <cmath>

namespace oneapi::dal::backend::primitives {

template <typename Float, typename Function>
sycl::event newton_cg(sycl::queue& queue,
                      Function& f,
                      ndview<Float, 1>& x,
                      Float tol,
                      std::int32_t maxiter,
                      const event_vector& deps) {
    std::int64_t n = x.get_dimension(0);

    const auto kernel_minus = [=](const Float& val, Float*) -> Float {
        return -val;
    };
    auto buffer = ndarray<Float, 1>::empty(queue, { 3 * n + 1 }, sycl::usm::alloc::device);

    auto buffer1 = buffer.get_slice(0, n);
    auto buffer2 = buffer.get_slice(n, 2 * n);
    auto buffer3 = buffer.get_slice(2 * n, 3 * n);
    Float* norm_gpu = buffer.get_mutable_data();
    norm_gpu += n * 3;

    for (std::int32_t i = 0; i < maxiter; ++i) {
        auto update_event = f.update_x(x, deps);
        auto gradient = f.get_gradient();
        Float grad_norm = 0;
        l1_norm(queue, gradient, norm_gpu, &grad_norm, { update_event }).wait_and_throw();
        Float tol_k = std::min(sqrt(grad_norm), 0.5); //

        auto prepare_grad_event =
            element_wise(queue, kernel_minus, gradient, nullptr, gradient, { update_event });
        auto solve_event = cg_solve(queue,
                                    f.get_hessian_product(),
                                    gradient,
                                    x,
                                    buffer1,
                                    buffer2,
                                    buffer3,
                                    tol_k,
                                    Float(0),
                                    n * 20,
                                    { prepare_grad_event });
        solve_event.wait_and_throw();
    }

    return {};
}

#define INSTANTIATE(F, Function)                               \
    template sycl::event newton_cg<F, Function>(sycl::queue&,  \
                                                Function&,     \
                                                ndview<F, 1>&, \
                                                F,             \
                                                std::int32_t,  \
                                                const event_vector&);

INSTANTIATE(float, convex_function<float>);
INSTANTIATE(double, convex_function<double>);

} // namespace oneapi::dal::backend::primitives
