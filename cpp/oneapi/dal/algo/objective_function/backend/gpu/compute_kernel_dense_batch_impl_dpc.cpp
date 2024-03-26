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

#include "oneapi/dal/algo/objective_function/backend/gpu/compute_kernel_dense_batch_impl.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/algo/objective_function/backend/objective_impl.hpp"
#include "oneapi/dal/backend/primitives/objective_function.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::objective_function::backend {

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

using alloc = sycl::usm::alloc;

using bk::context_gpu;
using task_t = task::compute;
using input_t = compute_input<task_t>;
using result_t = compute_result<task_t>;

std::int64_t get_block_size(std::int64_t n, std::int64_t p) {
    constexpr std::int64_t max_alloc_size = 1 << 21;
    return p > max_alloc_size ? 512 : max_alloc_size / p;
}

template <typename Float>
void add_regularization(sycl::queue& q_,
                        const detail::descriptor_base<task_t>& desc,
                        result_t& result,
                        const std::int64_t n,
                        const std::int64_t p,
                        const Float* params_ptr,
                        pr::ndarray<Float, 1>& ans_loss,
                        pr::ndarray<Float, 1>& ans_gradient,
                        pr::ndarray<Float, 2>& ans_hessian,
                        const Float L1,
                        const Float L2,
                        const bk::event_vector& deps) {
    const Float inv_n = Float(1) / n;

    auto* const ans_loss_ptr = ans_loss.get_mutable_data();
    auto* const ans_grad_ptr = ans_gradient.get_mutable_data();

    sycl::event prev_logloss_e, prev_grad_e, prev_hess_e;

    if (desc.get_result_options().test(result_options::value)) {
        prev_logloss_e = q_.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.single_task([=] {
                ans_loss_ptr[0] *= inv_n;
                for (std::int64_t i = 1; i < p + 1; ++i) {
                    ans_loss_ptr[0] +=
                        L2 * params_ptr[i] * params_ptr[i] + L1 * sycl::fabs(params_ptr[i]);
                }
            });
        });
    }
    if (desc.get_result_options().test(result_options::gradient)) {
        prev_grad_e = q_.submit([&](sycl::handler& cgh) {
            const auto range = oneapi::dal::backend::make_range_1d(p + 1);
            cgh.depends_on(deps);
            cgh.parallel_for(range, [=](sycl::id<1> idx) {
                ans_grad_ptr[idx[0]] *= inv_n;
                if (idx > 0) {
                    ans_grad_ptr[idx] += L2 * params_ptr[idx] * 2;
                }
            });
        });
    }

    if (desc.get_result_options().test(result_options::hessian)) {
        prev_hess_e = q_.submit([&](sycl::handler& cgh) {
            auto* const ans_hess_ptr = ans_hessian.get_mutable_data();
            const auto range2d = oneapi::dal::backend::make_range_2d(p + 1, p + 1);
            cgh.depends_on(deps);
            cgh.parallel_for(range2d, [=](sycl::id<2> idx) {
                ans_hess_ptr[idx[0] * (p + 1) + idx[1]] *= inv_n;
                if (idx[0] == idx[1] && idx[0] > 0) {
                    ans_hess_ptr[idx[0] * (p + 1) + idx[1]] += L2 * 2;
                }
            });
        });
    }

    if (desc.get_result_options().test(result_options::value)) {
        result.set_value(homogen_table::wrap(ans_loss.flatten(q_, { prev_logloss_e }), 1, 1));
    }
    if (desc.get_result_options().test(result_options::gradient)) {
        result.set_gradient(
            homogen_table::wrap(ans_gradient.flatten(q_, { prev_grad_e }), p + 1, 1));
    }
    if (desc.get_result_options().test(result_options::hessian)) {
        result.set_hessian(
            homogen_table::wrap(ans_hessian.flatten(q_, { prev_hess_e }), p + 1, p + 1));
    }
}

template <typename Float>
sycl::event value_and_gradient_iter(sycl::queue& q_,
                                    std::int64_t p,
                                    const pr::ndview<Float, 2>& data_nd,
                                    const pr::ndview<std::int32_t, 1>& responses_nd,
                                    const pr::ndview<Float, 1>& probabilities,
                                    pr::ndview<Float, 1>& out,
                                    pr::ndview<Float, 1>& ans,
                                    bool fit_intercept,
                                    sycl::event& prev_iter) {
    auto fill_event = fill(q_, out, Float(0), {});

    auto out_loss = out.get_slice(0, 1);
    auto out_gradient = out.get_slice(1, p + 2);
    auto out_gradient_suf = fit_intercept ? out_gradient : out_gradient.get_slice(1, p + 1);

    auto loss_event = compute_logloss_with_der(q_,
                                               data_nd,
                                               responses_nd,
                                               probabilities,
                                               out_loss,
                                               out_gradient_suf,
                                               fit_intercept,
                                               { fill_event });

    const auto* const out_ptr = out.get_data();
    auto* const ans_ptr = ans.get_mutable_data();
    return q_.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ loss_event, prev_iter });
        const auto range = oneapi::dal::backend::make_range_1d(p + 2);
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            ans_ptr[idx] += out_ptr[idx];
        });
    });
}

template <typename Float>
sycl::event value_iter(sycl::queue& q_,
                       const pr::ndview<std::int32_t, 1>& responses_nd,
                       const pr::ndview<Float, 1>& probabilities,
                       pr::ndview<Float, 1>& out_loss,
                       pr::ndview<Float, 1>& ans_loss,
                       bool fit_intercept,
                       sycl::event& prev_iter) {
    auto fill_event = fill(q_, out_loss, Float(0), {});
    auto loss_event =
        compute_logloss(q_, responses_nd, probabilities, out_loss, fit_intercept, { fill_event });
    const auto* const out_ptr = out_loss.get_data();
    auto* const ans_loss_ptr = ans_loss.get_mutable_data();
    return q_.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ loss_event, prev_iter });
        cgh.single_task([=] {
            *ans_loss_ptr += *out_ptr;
        });
    });
}

template <typename Float>
sycl::event gradient_iter(sycl::queue& q_,
                          std::int64_t p,
                          const pr::ndarray<Float, 2>& data_nd,
                          const pr::ndarray<std::int32_t, 1>& responses_nd,
                          const pr::ndarray<Float, 1>& probabilities,
                          pr::ndarray<Float, 1>& out_gradient,
                          pr::ndarray<Float, 1>& ans_gradient,
                          bool fit_intercept,
                          sycl::event& prev_iter) {
    auto fill_event = fill(q_, out_gradient, Float(0), {});
    auto out_grad_suf = fit_intercept ? out_gradient : out_gradient.get_slice(1, p + 1);
    auto grad_event = compute_derivative(q_,
                                         data_nd,
                                         responses_nd,
                                         probabilities,
                                         out_grad_suf,
                                         fit_intercept,
                                         { fill_event });
    grad_event.wait_and_throw();
    const auto* const grad_ptr = out_gradient.get_data();
    auto* const ans_grad_ptr = ans_gradient.get_mutable_data();
    return q_.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ grad_event, prev_iter });

        const auto range = oneapi::dal::backend::make_range_1d(p + 1);

        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            ans_grad_ptr[idx] += grad_ptr[idx];
        });
    });
}

template <typename Float>
sycl::event hessian_iter(sycl::queue& q_,
                         std::int64_t p,
                         const pr::ndarray<Float, 2>& data_nd,
                         const pr::ndarray<std::int32_t, 1>& responses_nd,
                         const pr::ndarray<Float, 1>& probabilities,
                         pr::ndarray<Float, 2>& out_hessian,
                         pr::ndarray<Float, 2>& ans_hessian,
                         bool fit_intercept,
                         sycl::event& prev_iter) {
    auto fill_event = fill(q_, out_hessian, Float(0), {});
    auto hess_event = compute_hessian(q_,
                                      data_nd,
                                      responses_nd,
                                      probabilities,
                                      out_hessian,
                                      Float(0),
                                      Float(0),
                                      fit_intercept,
                                      { fill_event });
    const auto* const hess_ptr = out_hessian.get_data();
    auto* const ans_hess_ptr = ans_hessian.get_mutable_data();
    return q_.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ hess_event, prev_iter });
        const auto range = oneapi::dal::backend::make_range_2d(p + 1, p + 1);
        cgh.parallel_for(range, [=](sycl::id<2> idx) {
            ans_hess_ptr[idx[0] * (p + 1) + idx[1]] += hess_ptr[idx[0] * (p + 1) + idx[1]];
        });
    });
}

template <typename Float>
result_t compute_kernel_dense_batch_impl<Float>::operator()(
    const detail::descriptor_base<task_t>& desc,
    const input_t& input) {
    ONEDAL_ASSERT(input.get_data().has_data());
    ONEDAL_ASSERT(input.get_parameters().has_data());
    ONEDAL_ASSERT(input.get_responses().has_data());
    const auto data = input.get_data();
    const auto params = input.get_parameters();
    const auto responses = input.get_responses();
    const std::int64_t n = data.get_row_count();
    const std::int64_t p = data.get_column_count();
    ONEDAL_ASSERT(responses.get_row_count() == n);
    ONEDAL_ASSERT(responses.get_column_count() == 1);
    ONEDAL_ASSERT(params.get_row_count() == p + 1);
    ONEDAL_ASSERT(params.get_column_count() == 1);

    auto obj_impl = detail::get_objective_impl(desc);

    const Float L1 = obj_impl->get_l1_regularization_coefficient();
    const Float L2 = obj_impl->get_l2_regularization_coefficient();
    bool fit_intercept = obj_impl->get_intercept_flag();

    const std::int64_t bsz = get_block_size(n, p);
    const bk::uniform_blocking blocking(n, bsz);

    const auto params_nd = pr::table2ndarray_1d<Float>(q_, params, alloc::device);
    const auto params_nd_suf = fit_intercept ? params_nd : params_nd.slice(1, p);
    const auto* const params_ptr = params_nd.get_data();

    const auto responses_nd_big = pr::table2ndarray_1d<std::int32_t>(q_, responses, alloc::device);

    auto result = compute_result<task_t>{}.set_result_options(desc.get_result_options());

    auto [out, out_e] = pr::ndarray<Float, 1>::zeros(q_, { p + 2 }, sycl::usm::alloc::device);
    out_e.wait_and_throw();

    auto [ans, ans_e] = pr::ndarray<Float, 1>::zeros(q_, { p + 2 }, sycl::usm::alloc::device);
    ans_e.wait_and_throw();

    auto out_loss = out.slice(0, 1);
    auto out_gradient = out.slice(1, p + 1);
    auto ans_loss = ans.slice(0, 1);
    auto ans_gradient = ans.slice(1, p + 1);

    pr::ndarray<Float, 2> out_hessian, ans_hessian;

    if (desc.get_result_options().test(result_options::hessian)) {
        ONEDAL_ASSERT_MUL_OVERFLOW(std::int64_t, p + 1, p + 1);
        out_hessian = pr::ndarray<Float, 2>::empty(q_, { p + 1, p + 1 }, sycl::usm::alloc::device);
        ans_hessian = pr::ndarray<Float, 2>::empty(q_, { p + 1, p + 1 }, sycl::usm::alloc::device);
        auto fill_event = fill(q_, ans_hessian, Float(0), {});
        fill_event.wait_and_throw();
    }

    auto probabilities_big = pr::ndarray<Float, 1>::empty(q_, { bsz }, sycl::usm::alloc::device);

    sycl::event prev_logloss_e, prev_grad_e, prev_hess_e;

    for (std::int64_t b = 0; b < blocking.get_block_count(); ++b) {
        const auto first = blocking.get_block_start_index(b);
        const auto last = blocking.get_block_end_index(b);
        const std::int64_t cursize = last - first;

        auto probabilities = probabilities_big.slice(0, cursize);

        const auto data_rows =
            row_accessor<const Float>(data).pull(q_, { first, last }, sycl::usm::alloc::device);
        const auto data_nd = pr::ndarray<Float, 2>::wrap(data_rows, { cursize, p });
        const auto responses_nd = responses_nd_big.slice(first, cursize);

        sycl::event prob_e =
            compute_probabilities(q_, params_nd_suf, data_nd, probabilities, fit_intercept, {});
        prob_e.wait_and_throw();

        if (desc.get_result_options().test(result_options::value) &&
            desc.get_result_options().test(result_options::gradient)) {
            prev_logloss_e = value_and_gradient_iter(q_,
                                                     p,
                                                     data_nd,
                                                     responses_nd,
                                                     probabilities,
                                                     out,
                                                     ans,
                                                     fit_intercept,
                                                     prev_logloss_e);
        }
        else {
            if (desc.get_result_options().test(result_options::value)) {
                prev_logloss_e = value_iter(q_,
                                            responses_nd,
                                            probabilities,
                                            out_loss,
                                            ans_loss,
                                            fit_intercept,
                                            prev_logloss_e);
            }
            if (desc.get_result_options().test(result_options::gradient)) {
                prev_grad_e = gradient_iter(q_,
                                            p,
                                            data_nd,
                                            responses_nd,
                                            probabilities,
                                            out_gradient,
                                            ans_gradient,
                                            fit_intercept,
                                            prev_grad_e);
            }
        }
        if (desc.get_result_options().test(result_options::hessian)) {
            prev_hess_e = hessian_iter(q_,
                                       p,
                                       data_nd,
                                       responses_nd,
                                       probabilities,
                                       out_hessian,
                                       ans_hessian,
                                       fit_intercept,
                                       prev_hess_e);
        }
    }
    add_regularization<Float>(q_,
                              desc,
                              result,
                              n,
                              p,
                              params_ptr,
                              ans_loss,
                              ans_gradient,
                              ans_hessian,
                              L1,
                              L2,
                              { prev_logloss_e, prev_grad_e, prev_hess_e });

    return result;
}

template class compute_kernel_dense_batch_impl<float>;
template class compute_kernel_dense_batch_impl<double>;

} // namespace oneapi::dal::objective_function::backend

#endif // ONEDAL_DATA_PARALLEL
