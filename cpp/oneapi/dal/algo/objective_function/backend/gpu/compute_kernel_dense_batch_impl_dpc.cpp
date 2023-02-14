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
// #include "oneapi/dal/backend/primitives/stat.hpp"
// #include "oneapi/dal/backend/primitives/blas.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::objective_function::backend {

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

using alloc = sycl::usm::alloc;

using bk::context_gpu;
using task_t = task::compute;
using input_t = compute_input<task_t>;
using result_t = compute_result<task_t>;
//using descriptor_t = detail::descriptor_base<task_t>;


template <typename Float>
result_t compute_kernel_dense_batch_impl<Float>::operator()(const detail::descriptor_base<task_t>& desc,
                                                      const input_t& input) {
    ONEDAL_ASSERT(input.get_data().has_data());
    ONEDAL_ASSERT(input.get_parameters().has_data());
    ONEDAL_ASSERT(input.get_responses().has_data());
    const auto data = input.get_data();
    const auto params = input.get_parameters();
    const auto responses = input.get_responses();
    const std::int64_t n = data.get_row_count();
    const std::int64_t p = data.get_column_count();
    ONEDAL_ASSERT(responses.get_row_count() == 1);
    ONEDAL_ASSERT(responses.get_column_count() == n);
    ONEDAL_ASSERT(params.get_row_count() == 1);
    ONEDAL_ASSERT(params.get_column_count() == p + 1);

    auto obj_impl = detail::get_objective_impl(desc);

    const Float L1 = obj_impl->get_l1_regularization_coefficient();
    const Float L2 = obj_impl->get_l2_regularization_coefficient();

    const std::int64_t bsz = 4;

    const auto params_nd = pr::table2ndarray_1d<Float>(q_, params, alloc::device);
    const auto responses_nd_big = pr::table2ndarray_1d<std::int32_t>(q_, responses, alloc::device);
    
    auto result = compute_result<task_t>{}.set_result_options(desc.get_result_options());

    

    auto [out, out_e] = pr::ndarray<Float, 1>::zeros(q_, {p + 2}, sycl::usm::alloc::device);
    out_e.wait_and_throw();

    auto [ans, ans_e] = pr::ndarray<Float, 1>::zeros(q_, {p + 2}, sycl::usm::alloc::device);
    ans_e.wait_and_throw();

    auto* const ans_ptr = ans.get_mutable_data();

    auto out_loss = out.slice(0, 1);
    auto out_gradient = out.slice(1, p + 1);
    auto ans_loss = ans.slice(0, 1);
    auto* const ans_loss_ptr = ans_loss.get_mutable_data();
    auto ans_gradient = ans.slice(1, p + 1);
    auto* const ans_grad_ptr = ans_gradient.get_mutable_data();

    pr::ndarray<Float, 2> out_hessian, ans_hessian;

    if (desc.get_result_options().test(result_options::hessian)) {
        out_hessian = pr::ndarray<Float, 2>::empty(q_, {p + 1, p + 1}, sycl::usm::alloc::device);
        ans_hessian = pr::ndarray<Float, 2>::empty(q_, {p + 1, p + 1}, sycl::usm::alloc::device);
        auto fill_event = fill(q_, ans_hessian, Float(0), {});
        fill_event.wait_and_throw();
    }

    auto probabilities_big = pr::ndarray<Float, 1>::empty(q_, {bsz}, sycl::usm::alloc::device); 

    for (std::int64_t i = 0; i < n; i += bsz) {
        std::int64_t cursize = std::min(n - i, bsz);
        
        auto probabilities = probabilities_big.slice(0, cursize);

        const auto data_rows = row_accessor<const Float>(data).pull(q_, { i, i + cursize }, sycl::usm::alloc::device);
        const auto data_nd = pr::ndarray<Float, 2>::wrap(data_rows, {cursize, p});
        const auto responses_nd = responses_nd_big.slice(i, i + cursize);
    
        sycl::event prob_e = compute_probabilities(q_, params_nd, data_nd, probabilities, {});

        // We only need to add regularization once so we do it only on the first iteration
        const Float L1_ = i == 0 ? L1 : 0;
        const Float L2_ = i == 0 ? L2 : 0; 


        if (desc.get_result_options().test(result_options::value) && desc.get_result_options().test(result_options::gradient)) {
            auto fill_event = fill(q_, out, Float(0), {});
            
            auto loss_event = compute_logloss_with_der(q_,
                                        params_nd,
                                        data_nd,
                                        responses_nd,
                                        probabilities,
                                        out_loss,
                                        out_gradient,
                                        L1_,
                                        L2_,
                                        {prob_e, fill_event});
            loss_event.wait_and_throw();
            
            const auto* const out_ptr = out.get_data();
            // const Float last_loss = i == 0 ? 0 : row_accessor<const Float>(result.get_value()).pull({0, -1})[0]; // create only one
            // auto grad_nd = pr::table2ndarray_1d<Float>(q_, result.get_gradient(), alloc::device);
            // auto* const ans_ptr = ans.get_mutable_data(); // pointer to rvalue!!!
           
            auto update_event = q_.submit([&](sycl::handler& cgh) {
                cgh.depends_on({ loss_event });
                const auto range = oneapi::dal::backend::make_range_1d(p + 2);
                cgh.parallel_for(range, [=](sycl::id<1> idx) {
                    ans_ptr[idx] += out_ptr[idx];
                    //Float last_val = idx == 0 ? last_loss : grad_ptr[idx - 1];
                    //out_ptr[idx] += last_val;
                });
            });
            update_event.wait_and_throw();
            //result.set_value(homogen_table::wrap(out_loss.flatten(q_, {}), 1, 1)); // do only once at the end of cycle
            //result.set_gradient(homogen_table::wrap(out_gradient.flatten(q_, {}), 1, p + 1));
            
        } else {
            
            if (desc.get_result_options().test(result_options::value)) {
                //auto out_loss = out.slice(0, 1);
                auto fill_event = fill(q_, out_loss, Float(0), {});
                auto loss_event = compute_logloss(q_,
                                                    params_nd,
                                                    data_nd,
                                                    responses_nd,
                                                    probabilities,
                                                    out_loss,
                                                    L1_,
                                                    L2_,
                                                    { prob_e, fill_event });
                loss_event.wait_and_throw();

                // auto out_host = out_loss.to_host(q_);
                // std::cout << i << ": " << out_host.get_data()[0] << std::endl;

                
                const auto* const out_ptr = out_loss.get_data();
                // Float last_val = i == 0 ? 0 : row_accessor<const Float>(result.get_value()).pull({0, -1})[0];
                auto update_event = q_.submit([&](sycl::handler& cgh) {
                    cgh.depends_on({ loss_event });
                    cgh.single_task([=] {
                        ans_loss_ptr[0] += out_ptr[0];
                    });
                });
                update_event.wait_and_throw();
                // result.set_value(homogen_table::wrap(out_loss.flatten(q_, { update_event }), 1, 1));
                //*/
            }

            if (desc.get_result_options().test(result_options::gradient)) {
                //auto out_gradient = out.slice(1, p + 1);
                auto fill_event = fill(q_, out_gradient, Float(0), {});
                auto grad_event = compute_derivative(q_,
                                                    params_nd,
                                                    data_nd,
                                                    responses_nd,
                                                    probabilities,
                                                    out_gradient,
                                                    L1_,
                                                    L2_,
                                                    { prob_e, fill_event });
                grad_event.wait_and_throw();
                const auto* const grad_ptr = out_gradient.get_data();
                auto update_event = q_.submit([&](sycl::handler& cgh) {
                    cgh.depends_on({ grad_event });

                    const auto range = oneapi::dal::backend::make_range_1d(p + 1);
                    
                    cgh.parallel_for(range, [=](sycl::id<1> idx) {
                        ans_grad_ptr[idx] += grad_ptr[idx];
                    });
                });
                update_event.wait_and_throw();
                /*
                auto* const out_ptr = out_gradient.get_mutable_data();
                auto grad_ptr = pr::table2ndarray_1d<Float>(q_, result.get_gradient(), alloc::device).get_mutable_data();
                if (i > 0) {
                    auto update_event = q_.submit([&](sycl::handler& cgh) {
                        cgh.depends_on({ grad_event });

                        const auto range = oneapi::dal::backend::make_range_1d(p + 1);
                        
                        cgh.parallel_for(range, [=](sycl::id<1> idx) {
                            out_ptr[idx] += grad_ptr[idx];
                        });
                    });
                    update_event.wait_and_throw();
                }
                result.set_gradient(homogen_table::wrap(out_gradient.flatten(q_, { }), 1, p + 1));
                */
            }
            
        }
        
        if (desc.get_result_options().test(result_options::hessian)) {
            //auto [out_hessian, out_hess_e] = pr::ndarray<Float, 2>::zeros(q_, { p + 1, p + 1 }, sycl::usm::alloc::device);
            auto fill_event = fill(q_, out_hessian, Float(0), {});
            auto hess_event = compute_hessian(q_,
                                                params_nd,
                                                data_nd,
                                                responses_nd,
                                                probabilities,
                                                out_hessian,
                                                L1_,
                                                L2_,
                                                { prob_e, fill_event });
            hess_event.wait_and_throw();
            auto* const ans_hess_ptr = ans_hessian.get_mutable_data();
            const auto* const hess_ptr = out_hessian.get_data();
            auto update_event = q_.submit([&](sycl::handler& cgh) {
                cgh.depends_on({ hess_event });
                const auto range = oneapi::dal::backend::make_range_2d(p + 1, p + 1);          
                cgh.parallel_for(range, [=](sycl::id<2> idx) {
                    ans_hess_ptr[idx[0] * (p + 1) + idx[1]] += hess_ptr[idx[0] * (p + 1) + idx[1]];
                    //for (std::int64_t i = 0; i < p + 1; ++i) {
                        
                    //}
                });
            });
            update_event.wait_and_throw();
            /*
            auto* const out_ptr = out_hessian.get_mutable_data();
            auto hess_ptr = pr::table2ndarray<Float>(q_, result.get_hessian(), alloc::device).get_mutable_data();
            if (i > 0) {
                auto update_event = q_.submit([&](sycl::handler& cgh) {
                    cgh.depends_on({ hess_event });
                    const auto range = oneapi::dal::backend::make_range_1d(p + 1);          
                    cgh.parallel_for(range, [=](sycl::id<1> idx) {
                        for (std::int64_t i = 0; i < p + 1; ++i) {
                            out_ptr[idx * (p + 1) + i] += hess_ptr[idx * (p + 1) + i];
                        }
                    });
                });
                update_event.wait_and_throw();
            }
            result.set_hessian(homogen_table::wrap(out_hessian.flatten(q_, { }), p + 1, p + 1));
            */
        }
        if (desc.get_result_options().test(result_options::value)) {
            result.set_value(homogen_table::wrap(ans_loss.flatten(q_, {  }), 1, 1));
        }
        if (desc.get_result_options().test(result_options::gradient)) {
            result.set_gradient(homogen_table::wrap(ans_gradient.flatten(q_, { }), 1, p + 1));
        }
        if (desc.get_result_options().test(result_options::hessian)) {
            result.set_hessian(homogen_table::wrap(ans_hessian.flatten(q_, { }), p + 1, p + 1));
        }

    } 


    /*
    
    if (desc.get_result_options().test(result_options::value) && desc.get_result_options().test(result_options::gradient)) {
        auto [out, out_e] = pr::ndarray<Float, 1>::zeros(q_, {p + 2}, sycl::usm::alloc::device);
        auto out_loss = out.slice(0, 1);
        auto out_gradient = out.slice(1, p + 1);
        auto loss_event = compute_logloss_with_der(q_,
                                     params_nd,
                                     data_nd,
                                     responses_nd,
                                     probabilities,
                                     out_loss,
                                     out_gradient,
                                     L1,
                                     L2,
                                     {prob_e, out_e});
        result.set_value(homogen_table::wrap(out_loss.flatten(q_, { loss_event }), 1, 1));
        result.set_gradient(homogen_table::wrap(out_gradient.flatten(q_, { loss_event }), 1, p + 1));
    } else {
        if (desc.get_result_options().test(result_options::value)) {
            auto [out_loss, out_loss_e] = pr::ndarray<Float, 1>::zeros(q_, { 1 }, sycl::usm::alloc::device);
            auto loss_event = compute_logloss(q_,
                                                params_nd,
                                                data_nd,
                                                responses_nd,
                                                probabilities,
                                                out_loss,
                                                L1,
                                                L2,
                                                { prob_e, out_loss_e });
            result.set_value(homogen_table::wrap(out_loss.flatten(q_, { loss_event }), 1, 1));
        }
        

        if (desc.get_result_options().test(result_options::gradient)) {
            auto [out_gradient, out_grad_e] = pr::ndarray<Float, 1>::zeros(q_, { p + 1 }, sycl::usm::alloc::device);
            auto grad_event = compute_derivative(q_,
                                                params_nd,
                                                data_nd,
                                                responses_nd,
                                                probabilities,
                                                out_gradient,
                                                L1,
                                                L2,
                                                { prob_e, out_grad_e });
            result.set_gradient(homogen_table::wrap(out_gradient.flatten(q_, { grad_event }), 1, p + 1));
        }
    }
    if (desc.get_result_options().test(result_options::hessian)) {
        auto [out_hessian, out_hess_e] = pr::ndarray<Float, 2>::zeros(q_, { p + 1, p + 1 }, sycl::usm::alloc::device);
        auto hes_event = compute_hessian(q_,
                                            params_nd,
                                            data_nd,
                                            responses_nd,
                                            probabilities,
                                            out_hessian,
                                            L1,
                                            L2,
                                            { prob_e, out_hess_e });
        result.set_hessian(homogen_table::wrap(out_hessian.flatten(q_, { hes_event }), p + 1, p + 1));
    }

    */
    return result;
    
}

template class compute_kernel_dense_batch_impl<float>;
template class compute_kernel_dense_batch_impl<double>;

} // namespace oneapi::dal::objective_function::backend

#endif // ONEDAL_DATA_PARALLEL
