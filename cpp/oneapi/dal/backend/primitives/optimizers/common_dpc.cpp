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
#include "oneapi/dal/backend/primitives/blas/gemv.hpp"
#include "oneapi/dal/backend/primitives/element_wise.hpp"

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
convex_function<Float>::convex_function(sycl::queue& q,
                                        const ndview<Float, 2>& A,
                                        const ndview<Float, 1>& b)
        : q_(q),
          A_(A),
          b_(b),
          hessp_(q, A) {
    std::int64_t n = A.get_dimension(0);
    ONEDAL_ASSERT(n == A.get_dimension(1));
    ONEDAL_ASSERT(n == b.get_dimension(0));
    gradient_ = ndarray<Float, 1>::empty(q_, { n }, sycl::usm::alloc::device);
}

template <typename Float>
ndview<Float, 1>& convex_function<Float>::get_gradient() {
    return gradient_;
}

template <typename Float>
matrix_operator<Float>& convex_function<Float>::get_hessian_product() {
    return hessp_;
}

template <typename Float>
sycl::event convex_function<Float>::update_x(const ndview<Float, 1>& x, const event_vector& deps) {
    auto fill_gradient_event = fill<Float>(q_, gradient_, Float(0), deps);
    auto gemv_event = gemv(q_, A_, x, gradient_, Float(1), Float(0), { fill_gradient_event });
    auto kernel_plus = sycl::plus<>();
    auto bias_event = element_wise(q_, kernel_plus, gradient_, b_, gradient_, { gemv_event });
    return bias_event;
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

template <typename Float>
sycl::event l1_norm(sycl::queue& queue,
                    const ndview<Float, 1>& x,
                    Float* res_gpu,
                    Float* res_host,
                    const event_vector& deps) {
    const std::int64_t n = x.get_dimension(0);
    auto* x_ptr = x.get_mutable_data();
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
                Float val = x_ptr[idx];
                sum += val >= 0 ? val : -val;
            });
        })
        .wait_and_throw();

    return queue.submit([&](sycl::handler& cgh) {
        cgh.memcpy(res_host, res_gpu, sizeof(Float));
    });
}

#define INSTANTIATE(F)                                        \
    template sycl::event dot_product<F>(sycl::queue&,         \
                                        const ndview<F, 1>&,  \
                                        const ndview<F, 1>&,  \
                                        F*,                   \
                                        F*,                   \
                                        const event_vector&); \
    template sycl::event l1_norm<F>(sycl::queue&,             \
                                    const ndview<F, 1>&,      \
                                    F*,                       \
                                    F*,                       \
                                    const event_vector&);     \
    template class matrix_operator<F>;                        \
    template class convex_function<F>;

INSTANTIATE(float);
INSTANTIATE(double);

} // namespace oneapi::dal::backend::primitives
