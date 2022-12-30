/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/backend/primitives/objective_function/logloss.hpp"
#include "oneapi/dal/backend/primitives/blas/gemv.hpp"
#include "oneapi/dal/backend/primitives/loops.hpp"
// #include "oneapi/dal/backend/primitives/reduction/reduction_1d.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include <sycl/ext/oneapi/experimental/builtins.hpp>
#include "oneapi/dal/backend/primitives/debug.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
sycl::event compute_predictions(sycl::queue& q,
                                const ndview<Float, 1>& parameters,
                                const ndview<Float, 2>& data,
                                ndview<Float, 1>& predictions,
                                const event_vector& deps) {
    auto fill_event = fill<Float>(q, predictions, Float(1), {});
    using oneapi::dal::backend::operator+;

    const int64_t n = data.get_dimension(0);

    auto param_arr = ndarray<Float, 1>::wrap(parameters.get_data(), 1);
    Float w0 = param_arr.to_host(q, deps).at(0); // Poor perfomance

    auto event =  gemv(q, 
                data,
                parameters.get_slice(1, parameters.get_dimension(0)),
                predictions,
                Float(1),
                w0,
                {fill_event});
    auto pred_ptr = predictions.get_mutable_data();

    return q.submit([&](sycl::handler& cgh){
        cgh.depends_on(event);
        const auto range = make_range_1d(n);
        cgh.parallel_for(range, [=](sycl::id<1> idx){
            pred_ptr[idx] = 1 / (1 + sycl::exp(-pred_ptr[idx]));
        });
    });
    // return event;
}

template <typename Float>
sycl::event compute_logloss(sycl::queue& q,
                      const ndview<Float, 1>& parameters,
                      const ndview<Float, 2>& data,
                      const ndview<std::int32_t, 1>& labels,
                      ndview<Float, 1>& out,
                      Float L1,
                      Float L2,
                      const event_vector& deps) {
    const std::int64_t n = data.get_dimension(0);
    const std::int64_t p = data.get_dimension(1);
    ONEDAL_ASSERT(parameters.get_dimension(0) == p + 1);
    ONEDAL_ASSERT(labels.get_dimension(0) == n);
    ONEDAL_ASSERT(labels.has_data());
    ONEDAL_ASSERT(parameters.has_data());
    ONEDAL_ASSERT(data.has_data());
    auto probabilities = ndarray<Float, 1>::empty(q, { n }, sycl::usm::alloc::device);
    auto prediction_event = compute_predictions(q, parameters, data, probabilities, deps);

    const std::int32_t* labels_ptr = labels.get_data();
    auto prob_ptr = probabilities.get_data();

    // auto [out, out_event] =
    //    ndarray<Float, 1>::full(q, std::int64_t(1), Float(0), sycl::usm::alloc::device);

    auto out_ptr = out.get_mutable_data();

    // auto tmp_arr = ndarray<Float, 1>::empty(q, {n});
    // auto tmp_ptr = tmp_arr.get_mutable_data();

    auto loss_event = q.submit([&](sycl::handler& cgh) {
        const auto range = make_range_1d(n);
        // cgh.depends_on(deps); unnessary because prediction event already as dependance on deps
        using oneapi::dal::backend::operator+;
        using sycl::reduction;

        // const event_vector& full_deps = deps + out_event + prediction_event;
        // const event_vector& full_deps = deps + prediction_event;
        cgh.depends_on({prediction_event});

        auto sumReduction = reduction(out_ptr, sycl::plus<>());

        // if y in {-1, 1} then loss function would be log(1 + exp(-y * pred))
        // so we need to compute log(1 + exp(-(y * 2 - 1) * pred)) as our original labels are in {0, 1}
        

        cgh.parallel_for(range, sumReduction, [=](sycl::id<1> idx, auto& sum) {
            const Float prob = prob_ptr[idx];
            const int32_t label = labels_ptr[idx];
            sum += -label * sycl::log(prob) - (1 - label) * sycl::log(1 - prob);
            //sum += sycl::log(1 + sycl::exp(-label * pred));
        });
    });
    

    auto [out_reg, out_reg_e] = ndarray<Float, 1>::zeros(q, {1}, sycl::usm::alloc::device);
    auto reg_ptr = out_reg.get_mutable_data();
    event_vector vector_out_reg = {out_reg_e};
    
    auto param_ptr = parameters.get_data();

    auto reg_event = q.submit([&](sycl::handler& cgh){
        cgh.depends_on(vector_out_reg);
        const auto range = make_range_1d(p + 1);
        auto sumReduction = sycl::reduction(reg_ptr, sycl::plus<>());
        cgh.parallel_for(range, sumReduction, [=](sycl::id<1> idx, auto& sum) {
            const Float param = param_ptr[idx];
            sum += L1 * sycl::abs(param) + L2 * param * param;
        });
    });

    auto final_event = q.submit([&](sycl::handler& cgh){
        cgh.depends_on({reg_event, loss_event});
        cgh.single_task([=]{
            out_ptr[0] += reg_ptr[0];
        });
    });

    return final_event;
}




template<typename Float>
sycl::event compute_logloss_with_der(sycl::queue& q,
                      const ndview<Float, 1>& parameters,
                      const ndview<Float, 2>& data,
                      const ndview<std::int32_t, 1>& labels,
                      ndview<Float, 1>& out,
                      ndview<Float, 1>& out_derivative,
                      Float L1,
                      Float L2,
                      const event_vector& deps) {

    const int64_t n = data.get_dimension(0);
    const int64_t p = data.get_dimension(1);

    ONEDAL_ASSERT(out.get_count() == 1);
    ONEDAL_ASSERT(out_derivative.get_count() == p + 1);


    // regression output
    auto probabilities = ndarray<Float, 1>::empty(q, { n }, sycl::usm::alloc::device);

    // d loss_i / d pred_i
    auto derivative_object = ndarray<Float, 1>::empty(q, { n }, sycl::usm::alloc::device);
    
    // auto der_obj_ptr = derivative_object.get_mutable_data();


    auto prediction_event = compute_predictions(q, parameters, data, probabilities, deps);


    auto der_obj_ptr = derivative_object.get_mutable_data();
    auto proba_ptr = probabilities.get_mutable_data();
    auto labels_ptr = labels.get_data();
    auto param_ptr = parameters.get_data();
    auto out_ptr = out.get_mutable_data();
    auto out_derivative_ptr = out_derivative.get_mutable_data();


    auto loss_event = q.submit([&](sycl::handler& cgh) {
        using oneapi::dal::backend::operator+;
        using sycl::reduction;

        // const event_vector& full_deps = deps + out_event + prediction_event;
        // const event_vector& full_deps = deps + prediction_event;
        cgh.depends_on({prediction_event});

        auto sumReductionLogLoss = reduction(out_ptr, sycl::plus<>());
        auto sumReductionDerivativeW0 = reduction(out_derivative_ptr, sycl::plus<>());
        // if y in {-1, 1} then loss function would be log(1 + exp(-y * pred))
        // so we need to compute log(1 + exp(-(y * 2 - 1) * pred)) as our original labels are in {0, 1}

        // const auto range = make_range_1d(n);
        const auto wg_size = propose_wg_size(q);
        const auto range = make_multiple_nd_range_1d(n, wg_size);

        
        cgh.parallel_for(range, sumReductionLogLoss, sumReductionDerivativeW0, [=](sycl::nd_item<1> id, auto& sum_logloss, auto& sum_Dw0) {
            auto idx = id.get_group_linear_id() * wg_size + id.get_local_linear_id();
            if (idx >= std::size_t(n)) 
                return;
            const Float prob = proba_ptr[idx];
            const float label = labels_ptr[idx];
            sum_logloss += -label * sycl::log(prob) - (1 - label) * sycl::log(1 - prob);
            der_obj_ptr[idx] = prob - label;
            sum_Dw0 += der_obj_ptr[idx];
        });
        
    });

    auto out_der_suffix = out_derivative.get_slice(1, p + 1);
    
    auto der_event = gemv(q, data.t(),
                derivative_object,
                out_der_suffix,
                {loss_event});

    auto [reg_val, reg_val_e] = ndarray<Float, 1>::zeros(q, {1}, sycl::usm::alloc::device);

    event_vector vec = {reg_val_e};
    auto reg_ptr = reg_val.get_mutable_data();

    auto reg_event = q.submit([&](sycl::handler& cgh){
        using oneapi::dal::backend::operator+;
        cgh.depends_on(vec + der_event);
        const auto range = make_range_1d(p + 1);
        auto sumReduction = sycl::reduction(reg_ptr, sycl::plus<>());
        sycl::stream ss(16384, 16, cgh);
        cgh.parallel_for(range, sumReduction, [=](sycl::id<1> idx, auto& sum) {
            const Float param = param_ptr[idx];
            sum += L1 * sycl::abs(param) + L2 * param * param;
            out_derivative_ptr[idx] += L2 * 2 * param;
            if (param > 0) {
                out_derivative_ptr[idx] += L1;
            } else if (param < 0) {
                out_derivative_ptr[idx] -= L1;
            }
        });
    });


    auto final_event = q.submit([&](sycl::handler& cgh){
        cgh.depends_on({reg_event, loss_event});
        cgh.single_task([=]{
            out_ptr[0] += reg_ptr[0];
        });
    });

    // std::cout << out.to_host(q, {final_event});
    return final_event;
}




template<typename Float>
sycl::event compute_hessian(sycl::queue& q,
                      const ndview<Float, 1>& parameters,
                      const ndview<Float, 2>& data,
                      const ndview<std::int32_t, 1>& labels,
                      ndview<Float, 2>& out_hessian,
                      Float L1,
                      Float L2,
                      const event_vector& deps) {

    const int64_t n = data.get_dimension(0);
    const int64_t p = data.get_dimension(1);

   // ONEDAL_ASSERT(out.get_count() == 1);
    ONEDAL_ASSERT(out_hessian.get_dimension(0) == (p + 1));
    ONEDAL_ASSERT(out_hessian.get_dimension(1) == (p + 1));


    // regression output
    auto probabilities = ndarray<Float, 1>::empty(q, { n }, sycl::usm::alloc::device);


    auto prediction_event = compute_predictions(q, parameters, data, probabilities, deps);
    
    auto data_ptr = data.get_data();
    auto hes_ptr = out_hessian.get_mutable_data();
    auto proba_ptr = probabilities.get_mutable_data();


    event_vector hess_deps = {};
    
    for (std::int64_t j = 0; j < p; ++j) {
        for (std::int64_t k = j; k < p; ++k) {
            using oneapi::dal::backend::operator+;
            auto event = q.submit([&](sycl::handler& cgh) {
                cgh.depends_on({prediction_event});
                const auto range = make_range_1d(n);
                auto sumReduction = sycl::reduction(hes_ptr + (j + 1) * (p + 1) + (k + 1), sycl::plus<>());
                cgh.parallel_for(range, sumReduction, [=](sycl::id<1> idx, auto& sum) {
                    const Float prob = proba_ptr[idx]; 
                    sum += data_ptr[idx * p + j] * data_ptr[idx * p + k] * prob * (1 - prob);
                });
            });
            hess_deps = hess_deps + event;
        }
        auto event = q.submit([&](sycl::handler& cgh) {
            cgh.depends_on({prediction_event});
            const auto range = make_range_1d(n);
            auto sumReduction = sycl::reduction(hes_ptr + (j + 1), sycl::plus<>());
            cgh.parallel_for(range, sumReduction, [=](sycl::id<1> idx, auto& sum) {
                const Float prob = proba_ptr[idx]; 
                sum += data_ptr[idx * p + j] * prob * (1 - prob);
            });
        });
        hess_deps = hess_deps + event;
    }
    auto event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on({prediction_event});
        const auto range = make_range_1d(n);
        auto sumReduction = sycl::reduction(hes_ptr, sycl::plus<>());
        cgh.parallel_for(range, sumReduction, [=](sycl::id<1> idx, auto& sum) {
            const Float prob = proba_ptr[idx]; 
            sum += prob * (1 - prob);
        });
    });
    hess_deps = hess_deps + event;

    auto copy_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(hess_deps);
        const auto range = make_range_2d(p + 1, p + 1);
        cgh.parallel_for(range, [=](sycl::id<2> idx) {
            auto j = idx[0];
            auto k = idx[1];
            if (j < k) {
                hes_ptr[k * (p + 1) + j] = hes_ptr[j * (p + 1) + k];
            }
            if (j == k) {
                hes_ptr[j * (p + 1) + j] += 2 * L2;
            }
        });             
    });

    return copy_event;


    /*
    auto hess_event = q.submit([&](sycl::handler& cgh) {
        const auto range = make_range_2d(p, p);

        cgh.depends_on({prediction_event});
        cgh.parallel_for(range, [=](sycl::id<2> idx) {
            // can make more parallelization with atomics!
            const std::int64_t j = idx[0];
            const std::int64_t k = idx[1];
            const std::int64_t gi = (j + 1) * (p + 1) + (k + 1);
            for (int64_t i = 0; i < n; ++i) {
                const Float prob = proba_ptr[i];
                hes_ptr[gi] += data_ptr[i * p + j] * data_ptr[i * p + k] * prob * (1 - prob);
                if (k == 0) {
                    hes_ptr[j + 1] += prob * (1 - prob) * data_ptr[i * p + j];
                }
                if (j == 0 && k == 0) {
                    hes_ptr[0] += prob * (1 - prob);
                }
            }
            if (k == 0) {
                hes_ptr[(j + 1) * (p + 1)] = hes_ptr[j + 1]; 
            }
        });
    });
    

    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on({hess_event});
        const auto range = make_range_1d(p + 1);
        cgh.parallel_for(range, [=](sycl::id<1> idx){
            hes_ptr[idx * (p + 1) + idx] += 2 * L2;
        });
    });
    */
}




#define INSTANTIATE(F)  template sycl::event compute_predictions<F>(sycl::queue&, \
                                const ndview<F, 1>&, \
                                const ndview<F, 2>&, \
                                ndview<F, 1>&, \
                                const event_vector&); \
                        template sycl::event compute_logloss<F>(sycl::queue&, \
                                const ndview<F, 1>&, \
                                const ndview<F, 2>&, \
                                const ndview<std::int32_t, 1>&, \
                                ndview<F, 1>&, \
                                F, F, \
                                const event_vector&); \
                        template sycl::event compute_logloss_with_der<F>(sycl::queue&, \
                                const ndview<F, 1>&, \
                                const ndview<F, 2>&, \
                                const ndview<std::int32_t, 1>&, \
                                ndview<F, 1>&, \
                                ndview<F, 1>&, \
                                F, F, \
                                const event_vector&); \
                        template sycl::event compute_hessian<F>(sycl::queue&, \
                                const ndview<F, 1>&, \
                                const ndview<F, 2>&, \
                                const ndview<std::int32_t, 1>&, \
                                ndview<F, 2>&, \
                                F, F, \
                                const event_vector&);

INSTANTIATE(float);
INSTANTIATE(double);                                   



} // namespace oneapi::dal::backend::primitives
