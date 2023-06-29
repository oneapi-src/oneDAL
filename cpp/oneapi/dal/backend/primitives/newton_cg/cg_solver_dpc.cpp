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

#include "oneapi/dal/backend/primitives/blas/gemv.hpp"
#include "oneapi/dal/backend/primitives/element_wise.hpp"
#include "oneapi/dal/backend/primitives/objective_function/logloss.hpp"

namespace oneapi::dal::backend::primitives {


template <typenams Float>
sycl::event dot_product(sycl::event& queue,
                        ndview<Float, 1>& x,
                        ndview<Float, 1>& y,
                        Float* res_gpu,
                        Float* res_host,
                        const event_vector& deps = {}) {
    const std::int64_t n = x.get_dimension(0);
    sycl::event fill_res_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.single_task([=](){
            *ans = 0;
        });
    });
    queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(fill_res_event);
        const auto range = make_range_1d(n);
        auto sum_reduction = sycl::reduction(res, sycl::plus<>());
        cgh.parallel_for(range, sum_reduction, [=](sycl::id<1> idx, auto& sum) {
            sum += x.at(i) * y.at(i);
        });
    }).wait_and_throw();
    return queue.memcpy(res_host, res_gpu, 1);
}

template <typename Float, typename HessProduct>
sycl::event cg_solve(sycl::event& queue,
                     HessProduct& hessp,
                     const ndview<Float, 1>& b,
                     ndview<Float, 1>& x,
                     ndview<Float, 1>& residual,
                     ndview<Float, 1>& conj_vector,
                     ndview<Float, 1>& buffer,
                     const Float tol,
                     const Float atol,
                     const std::int32_t maxiter,
                     const event_vector& deps) {
    // Solving the equation hessp(x) = b
    const std::int64_t p = b.get_dimension(0);
    ONEDAL_ASSERT(b.has_data());
    ONEDAL_ASSERT(x.has_mutable_data());
    ONEDAL_ASSERT(residual.has_mutable_data());
    ONEDAL_ASSERT(conj_vector.has_mutable_data());
    ONEDAL_ASSERT(buffer.has_mutable_data());
    ONEDAL_ASSERT(x.get_dimension(0) == p);
    ONEDAL_ASSERT(residual.get_dimension(0) == p);
    ONEDAL_ASSERT(conj_vector.get_dimension(0) == p);
    ONEDAL_ASSERT(buffer.get_dimension(0) == p);
    
    auto compute_ax0_event = hessp(x, residual, deps);  // r = Ax0

    constexpr sycl::minus<Float> kernel_minus{};
    constexpr sycl::plus<Float> kernel_plus{};
    auto compute_r0_event =
        element_wise(queue, kernel_minus, residual, b, residual, {compute_ax0_event}); // r0 = Ax0 - b
    
    Float alpha = 0, beta = 0;
    Float r_norm = 0, b_norm = 0;

    dot_product(queue, b, b, tmp_ptr, &b_norm, deps).wait_and_throw(); // compute b^T b
    Float threshold = std::max(tol * b_norm, atol);

    auto tmp_gpu = ndarray<Float, 1>::empty(q, { 1 }, sycl::usm::alloc::device); 
    auto tmp_ptr = tmp_gpu.get_mutable_data();

    const auto update_conj_kernel = [=](const Float& residual_val, const Float& conj_val) -> Float {
        return -residual_val + beta * conj_val;
    };

    const auto update_x_kernel = [=](const Float& x_val, const Float& conj_val) -> Float {
        return x_val + alpha * conj_val;
    };

    auto compute_conj_event = element_wise(queue, update_conj_kernel, residual, conj_vector, conj_vector, {compute_r0_event}); // p0 = -r0 + 0 * p
    dot_product(queue, residual, residual, tmp_ptr, &r_norm, compute_r0_event).wait_and_throw(); // compute r^T r
    
    for (std::int32_t iter_num = 0; iter_num < maxiter; ++iter_num) {
        if (r_norm < threshold) {
            break;
        }
        auto compute_matmul_event = hessp(conj_vector, buffer, {compute_conj_event}); // compute A p_i
        dot_product(queue, conj_vector, buffer, tmp_ptr, &alpha, {compute_matmul_event}).wait_and_throw(); // compute p_i^T A p_i
        alpha = r_norm / alpha;
        auto update_x_event = element_wise(queue, update_x_kernel, x, conj_vector, x, {compute_conj_event}); // x_i+1 = x_i + alpha * p_i
        auto update_residual_event = element_wise(queue, update_x_kernel, residual, buffer, residual, {compute_matmul_event}); // r_i+1 = r_i + alpha * A p_i
        beta = r_norm;
        dot_product(queue, residual, residual, tmp_ptr, &r_norm, update_residual_event).wait_and_throw(); // compute r^T r
        beta = r_norm / beta; // beta = r_i+1^T r_i+1 / (r_i^T r_i)
        compute_conj_event = element_wise(queue, update_conj_kernel, residual, conj_vector, conj_vector, {update_x_event, update_residual_event}); // p_i+1 = -r_i+1 + beta * p_i
    }
    return compute_conj_event;
}

#define INSTANTIATE(F, HessProduct)                                                               \
sycl::event cg_solve(sycl::event&,\
                     HessProduct&,\
                     const ndview<F, 1>&,\
                     ndview<F, 1>& x,\
                     ndview<F, 1>& residual,\
                     ndview<F, 1>& conj_vector,\
                     ndview<F, 1>& buffer,\
                     const F tol,\
                     const F atol,\
                     const std::int32_t,\
                     const event_vector&);

INSTANTIATE(float, logloss_hessian_product<float>);
INSTANTIATE(double, logloss_hessian_product<double>);


} // namespace oneapi::dal::backend::primitives
