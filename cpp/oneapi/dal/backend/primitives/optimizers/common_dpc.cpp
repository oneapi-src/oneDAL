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
LinearMatrixOperator<Float>::LinearMatrixOperator(sycl::queue& q, const ndview<Float, 2>& A)
        : BaseMatrixOperator<Float>(),
          q_(q),
          A_(A) {}

template <typename Float>
sycl::event dot_product(sycl::queue& queue,
                        const ndview<Float, 1>& x,
                        const ndview<Float, 1>& y,
                        Float* res_gpu,
                        Float* res_host,
                        const event_vector& deps) {
    const std::int64_t n = x.get_dimension(0);
    auto* const x_ptr = x.get_mutable_data();
    auto* const y_ptr = y.get_mutable_data();
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
    auto* const x_ptr = x.get_mutable_data();

    sycl::event fill_res_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.fill(res_gpu, Float(0), 1);
    });

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
    template class BaseMatrixOperator<F>;                     \
    template class LinearMatrixOperator<F>;                   \
    template class BaseFunction<F>;                           \
    template class QuadraticFunction<F>;

INSTANTIATE(float);
INSTANTIATE(double);

} // namespace oneapi::dal::backend::primitives
