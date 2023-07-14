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

#include "oneapi/dal/backend/primitives/optimizers/cg_solver.hpp"
#include "oneapi/dal/backend/primitives/optimizers/common.hpp"
#include "oneapi/dal/backend/primitives/blas/gemv.hpp"
#include "oneapi/dal/backend/primitives/element_wise.hpp"
#include "oneapi/dal/backend/primitives/objective_function/logloss.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
matrix_operator<Float>::matrix_operator(sycl::queue& q, const ndview<Float, 2>& A) : q_(q),
                                                                                     A_(A) {}

template <typename Float>
sycl::event matrix_operator<Float>::operator()(const ndview<Float, 1>& vec,
                                               ndview<Float, 1>& out,
                                               const event_vector& deps) {
    ONEDAL_ASSERT(A_.get_dimension(1) == vec.get_dimension(0));
    ONEDAL_ASSERT(out.get_dimension(0) == vec.get_dimension(0));
    sycl::event fill_out_event = fill<Float>(q_, out, Float(0), deps);
    return gemv(q_, A_, vec, out, Float(1), Float(0), { fill_out_event });
}

template <typename Float>
sycl::event dot_product(sycl::queue& queue,
                        const ndview<Float, 1>& x,
                        const ndview<Float, 1>& y,
                        Float* res_gpu,
                        Float* res_host,
                        const event_vector& deps) {
    const std::int64_t n = x.get_dimension(0);
    auto* x_ptr = x.get_mutable_data();
    auto* y_ptr = y.get_mutable_data();
    sycl::event fill_res_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.single_task([=]() {
            *res_gpu = 0;
        });
    });
    fill_res_event.wait_and_throw();

    queue
        .submit([&](sycl::handler& cgh) {
            cgh.depends_on(fill_res_event);
            const auto range = make_range_1d(n);
            auto sum_reduction = sycl::reduction(res_gpu, sycl::plus<>());
            cgh.parallel_for(range, sum_reduction, [=](sycl::id<1> idx, auto& sum) {
                sum += x_ptr[idx] * y_ptr[idx];
            });
        })
        .wait_and_throw();

    return queue.submit([&](sycl::handler& cgh) {
        cgh.memcpy(res_host, res_gpu, sizeof(Float));
    });
}

template <typename Float, typename MatrixOperator>
sycl::event cg_solve(sycl::queue& queue,
                     MatrixOperator& mul_operator,
                     const ndview<Float, 1>& b,
                     ndview<Float, 1>& x,
                     ndview<Float, 1>& residual,
                     ndview<Float, 1>& conj_vector,
                     ndview<Float, 1>& buffer,
                     const Float tol,
                     const Float atol,
                     const std::int32_t maxiter,
                     const event_vector& deps) {
    // Solving the equation mul_operator(x) = b
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

    Float alpha = 0, beta = 0;
    Float r_norm = 0, b_norm = 0;
    Float r_l1_norm = 0;

    const auto kernel_minus = [=](const Float a, const Float b) -> Float {
        return a - b;
    };

    auto compute_ax0_event = mul_operator(x, residual, deps); // r = Ax0

    auto compute_r0_event = element_wise(queue,
                                         kernel_minus,
                                         residual,
                                         b,
                                         residual,
                                         { compute_ax0_event }); // r0 = Ax0 - b
    // compute_r0_event.wait_and_throw();
    auto tmp_gpu = ndarray<Float, 1>::empty(queue, { 1 }, sycl::usm::alloc::device);
    auto tmp_ptr = tmp_gpu.get_mutable_data();

    l1_norm<Float>(queue, b, tmp_ptr, &b_norm, deps).wait_and_throw(); // compute norm(b)

    // Tolerances for convergence norm(residual) <= max(tol*norm(b), atol)
    Float threshold = std::max(tol * b_norm, atol);

    const auto init_conj_kernel = [=](const Float residual_val, const Float conj_val) -> Float {
        return -residual_val;
    };
    auto compute_conj_event = element_wise(queue,
                                           init_conj_kernel,
                                           residual,
                                           conj_vector,
                                           conj_vector,
                                           { compute_r0_event }); // p0 = -r0 + 0 * p
    auto conj_host = conj_vector.to_host(queue, {});
    dot_product<Float>(queue, residual, residual, tmp_ptr, &r_norm, { compute_r0_event })
        .wait_and_throw(); // compute r^T r

    for (std::int32_t iter_num = 0; iter_num < maxiter; ++iter_num) {
        l1_norm<Float>(queue, residual, tmp_ptr, &r_l1_norm, { compute_conj_event })
            .wait_and_throw(); // compute norm(residual)
        if (r_l1_norm < threshold) {
            break;
        }
        auto compute_matmul_event =
            mul_operator(conj_vector, buffer, { compute_conj_event }); // compute A p_i
        dot_product<Float>(queue, conj_vector, buffer, tmp_ptr, &alpha, { compute_matmul_event })
            .wait_and_throw(); // compute p_i^T A p_i
        ONEDAL_ASSERT(alpha > 1e-15);
        alpha = r_norm / alpha;
        const auto update_x_kernel = [=](const Float x_val, const Float conj_val) -> Float {
            return x_val + alpha * conj_val;
        };
        auto update_x_event = element_wise(queue,
                                           update_x_kernel,
                                           x,
                                           conj_vector,
                                           x,
                                           { compute_conj_event }); // x_i+1 = x_i + alpha * p_i
        update_x_event.wait_and_throw();
        auto update_residual_event =
            element_wise(queue,
                         update_x_kernel,
                         residual,
                         buffer,
                         residual,
                         { compute_matmul_event }); // r_i+1 = r_i + alpha * A p_i
        update_residual_event.wait_and_throw();
        beta = r_norm;
        dot_product<Float>(queue, residual, residual, tmp_ptr, &r_norm, { update_residual_event })
            .wait_and_throw(); // compute r^T r
        beta = r_norm / beta; // beta = r_i+1^T r_i+1 / (r_i^T r_i)

        const auto update_conj_kernel = [=](const Float residual_val,
                                            const Float conj_val) -> Float {
            return -residual_val + beta * conj_val;
        };
        compute_conj_event =
            element_wise(queue,
                         update_conj_kernel,
                         residual,
                         conj_vector,
                         conj_vector,
                         { update_x_event, update_residual_event }); // p_i+1 = -r_i+1 + beta * p_i
    }
    return compute_conj_event;
}

#define INSTANTIATE_SOLVER(F, MatrixOperator)                             \
    template sycl::event cg_solve<F, MatrixOperator>(sycl::queue&,        \
                                                     MatrixOperator&,     \
                                                     const ndview<F, 1>&, \
                                                     ndview<F, 1>&,       \
                                                     ndview<F, 1>&,       \
                                                     ndview<F, 1>&,       \
                                                     ndview<F, 1>&,       \
                                                     const F,             \
                                                     const F,             \
                                                     const std::int32_t,  \
                                                     const event_vector&);

INSTANTIATE_SOLVER(float, logloss_hessian_product<float>);
INSTANTIATE_SOLVER(double, logloss_hessian_product<double>);
INSTANTIATE_SOLVER(float, matrix_operator<float>);
INSTANTIATE_SOLVER(double, matrix_operator<double>);

} // namespace oneapi::dal::backend::primitives
