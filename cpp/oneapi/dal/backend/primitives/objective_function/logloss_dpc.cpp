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
#include "oneapi/dal/table/row_accessor.hpp"
#include <sycl/ext/oneapi/experimental/builtins.hpp>
#include "oneapi/dal/backend/primitives/debug.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
sycl::event compute_probabilities(sycl::queue& q,
                                const ndview<Float, 1>& parameters,
                                const ndview<Float, 2>& data,
                                ndview<Float, 1>& probabilities,
                                const event_vector& deps) {
    auto fill_event = fill<Float>(q, probabilities, Float(1), {});
    using oneapi::dal::backend::operator+;

    const std::int64_t n = data.get_dimension(0);

    auto param_arr = ndarray<Float, 1>::wrap(parameters.get_data(), 1);
    Float w0 = param_arr.to_host(q, deps).at(0); // Poor perfomance

    auto event = gemv(q,
                      data,
                      parameters.get_slice(1, parameters.get_dimension(0)),
                      probabilities,
                      Float(1),
                      w0,
                      { fill_event });
    auto prob_ptr = probabilities.get_mutable_data();

    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(event);
        const auto range = make_range_1d(n);
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            prob_ptr[idx] = 1 / (1 + sycl::exp(-prob_ptr[idx]));
        });
    });
    // return event;
}

template <typename Float>
sycl::event compute_logloss(sycl::queue& q,
                            const ndview<Float, 1>& parameters,
                            const ndview<Float, 2>& data,
                            const ndview<std::int32_t, 1>& labels,
                            const ndview<Float, 1>& probabilities,
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


    auto labels_ptr = labels.get_data();
    auto prob_ptr = probabilities.get_data();

    auto out_ptr = out.get_mutable_data();

    auto loss_event = q.submit([&](sycl::handler& cgh) {
        const auto range = make_range_1d(n);
        using oneapi::dal::backend::operator+;
        using sycl::reduction;

        cgh.depends_on(deps);

        auto sumReduction = reduction(out_ptr, sycl::plus<>());

        cgh.parallel_for(range, sumReduction, [=](sycl::id<1> idx, auto& sum) {
            const Float prob = prob_ptr[idx];
            const std::int32_t label = labels_ptr[idx];
            sum += -label * sycl::log(prob) - (1 - label) * sycl::log(1 - prob);
        });
    });

    auto [out_reg, out_reg_e] = ndarray<Float, 1>::zeros(q, { 1 }, sycl::usm::alloc::device);
    auto reg_ptr = out_reg.get_mutable_data();
    event_vector vector_out_reg = { out_reg_e };

    auto param_ptr = parameters.get_data();

    auto reg_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(vector_out_reg);
        const auto range = make_range_1d(p + 1);
        auto sumReduction = sycl::reduction(reg_ptr, sycl::plus<>());
        cgh.parallel_for(range, sumReduction, [=](sycl::id<1> idx, auto& sum) {
            const Float param = param_ptr[idx];
            sum += L1 * sycl::abs(param) + L2 * param * param;
        });
    });

    auto final_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ reg_event, loss_event });
        cgh.single_task([=] {
            out_ptr[0] += reg_ptr[0];
        });
    });

    return final_event;
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

    // out should be filled with zero

    auto probabilities = ndarray<Float, 1>::empty(q, { n }, sycl::usm::alloc::device);
    auto prediction_event = compute_probabilities(q, parameters, data, probabilities, deps);

    return compute_logloss(q, parameters, data, labels, probabilities, out, L1, L2, {prediction_event});
}

template <typename Float>
sycl::event compute_logloss_with_der(sycl::queue& q,
                                     const ndview<Float, 1>& parameters,
                                     const ndview<Float, 2>& data,
                                     const ndview<std::int32_t, 1>& labels,
                                     const ndview<Float, 1>& probabilities,
                                     ndview<Float, 1>& out,
                                     ndview<Float, 1>& out_derivative,
                                     Float L1,
                                     Float L2,
                                     const event_vector& deps) {
    // out, out_derivative should be filled with zeros

    const std::int64_t n = data.get_dimension(0);
    const std::int64_t p = data.get_dimension(1);

    ONEDAL_ASSERT(out.get_count() == 1);
    ONEDAL_ASSERT(out_derivative.get_count() == p + 1);

    // d loss_i / d pred_i
    auto derivative_object = ndarray<Float, 1>::empty(q, { n }, sycl::usm::alloc::device);

    auto der_obj_ptr = derivative_object.get_mutable_data();
    auto proba_ptr = probabilities.get_data();
    auto labels_ptr = labels.get_data();
    auto param_ptr = parameters.get_data();
    auto out_ptr = out.get_mutable_data();
    auto out_derivative_ptr = out_derivative.get_mutable_data();

    auto loss_event = q.submit([&](sycl::handler& cgh) {
        using oneapi::dal::backend::operator+;
        using sycl::reduction;

        cgh.depends_on(deps);
        auto sumReductionLogLoss = reduction(out_ptr, sycl::plus<>());
        auto sumReductionDerivativeW0 = reduction(out_derivative_ptr, sycl::plus<>());
        const auto wg_size = propose_wg_size(q);
        const auto range = make_multiple_nd_range_1d(n, wg_size);

        cgh.parallel_for(
            range,
            sumReductionLogLoss,
            sumReductionDerivativeW0,
            [=](sycl::nd_item<1> id, auto& sum_logloss, auto& sum_Dw0) {
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

    auto der_event = gemv(q, data.t(), derivative_object, out_der_suffix, { loss_event });

    auto [reg_val, reg_val_e] = ndarray<Float, 1>::zeros(q, { 1 }, sycl::usm::alloc::device);

    event_vector vec = { reg_val_e };
    auto reg_ptr = reg_val.get_mutable_data();

    auto reg_event = q.submit([&](sycl::handler& cgh) {
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
            }
            else if (param < 0) {
                out_derivative_ptr[idx] -= L1;
            }
        });
    });

    auto final_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ reg_event, loss_event });
        cgh.single_task([=] {
            out_ptr[0] += reg_ptr[0];
        });
    });

    return final_event;
}


template <typename Float>
sycl::event compute_derivative(sycl::queue& q,
                                     const ndview<Float, 1>& parameters,
                                     const ndview<Float, 2>& data,
                                     const ndview<std::int32_t, 1>& labels,
                                     const ndview<Float, 1>& probabilities,
                                     ndview<Float, 1>& out_derivative,
                                     Float L1,
                                     Float L2,
                                     const event_vector& deps) {
    // out_derivative should be filled with zeros

    const std::int64_t n = data.get_dimension(0);
    const std::int64_t p = data.get_dimension(1);

    ONEDAL_ASSERT(out_derivative.get_count() == p + 1);

    // d loss_i / d pred_i
    auto derivative_object = ndarray<Float, 1>::empty(q, { n }, sycl::usm::alloc::device);

    auto der_obj_ptr = derivative_object.get_mutable_data();
    auto proba_ptr = probabilities.get_data();
    auto labels_ptr = labels.get_data();
    auto param_ptr = parameters.get_data();
    auto out_derivative_ptr = out_derivative.get_mutable_data();

    auto loss_event = q.submit([&](sycl::handler& cgh) {
        using oneapi::dal::backend::operator+;
        using sycl::reduction;

        cgh.depends_on(deps);
        auto sumReductionDerivativeW0 = reduction(out_derivative_ptr, sycl::plus<>());
        const auto wg_size = propose_wg_size(q);
        const auto range = make_multiple_nd_range_1d(n, wg_size);

        cgh.parallel_for(
            range,
            sumReductionDerivativeW0,
            [=](sycl::nd_item<1> id, auto& sum_Dw0) {
                auto idx = id.get_group_linear_id() * wg_size + id.get_local_linear_id();
                if (idx >= std::size_t(n))
                    return;
                const Float prob = proba_ptr[idx];
                const float label = labels_ptr[idx];
                der_obj_ptr[idx] = prob - label;
                sum_Dw0 += der_obj_ptr[idx];
            });
    });

    auto out_der_suffix = out_derivative.get_slice(1, p + 1);

    auto der_event = gemv(q, data.t(), derivative_object, out_der_suffix, { loss_event });

    auto reg_event = q.submit([&](sycl::handler& cgh) {
        using oneapi::dal::backend::operator+;
        cgh.depends_on({der_event});
        const auto range = make_range_1d(p + 1);
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            const Float param = param_ptr[idx];
            out_derivative_ptr[idx] += L2 * 2 * param;
            if (param > 0) {
                out_derivative_ptr[idx] += L1;
            }
            else if (param < 0) {
                out_derivative_ptr[idx] -= L1;
            }
        });
    });

    return reg_event;
}

template<typename Float>

sycl::event compute_hessian1(sycl::queue& q,
                            const ndview<Float, 1>& parameters,
                            const ndview<Float, 2>& data,
                            const ndview<std::int32_t, 1>& labels,
                            const ndview<Float, 1>& probabilities,
                            ndview<Float, 2>& out_hessian,
                            Float L1,
                            Float L2,
                            const event_vector& deps) {
    const int64_t n = data.get_dimension(0);
    const int64_t p = data.get_dimension(1);

    ONEDAL_ASSERT(out_hessian.get_dimension(0) == (p + 1));
    ONEDAL_ASSERT(out_hessian.get_dimension(1) == (p + 1));

    auto data_ptr = data.get_data();
    auto hes_ptr = out_hessian.get_mutable_data();
    auto proba_ptr = probabilities.get_mutable_data();

    const auto max_wg = device_max_wg_size(q);
    const auto wg = std::min(p + 1, max_wg);
    const auto inp_str = data.get_leading_stride();
    const auto out_str = out_hessian.get_leading_stride();

    const std::int64_t block_size = 32;
    const auto num_blocks = (n + block_size - 1) / block_size;
    const auto range = make_multiple_nd_range_3d({num_blocks, p + 1, wg}, {1, 1, wg});

    auto hes_event = q.submit([&](sycl::handler& cgh){
        // my optimized version 

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::nd_item<3> item) {
            const std::int64_t obj_ind = item.get_global_id(0);
            const auto j = item.get_global_id(1);
            const auto param_ind_2 = item.get_global_id(2);
            Float val = 0;
            for (auto k = param_ind_2; k <= j; k += wg) {
                val = 0;
                for (auto i = obj_ind * block_size; i < std::min((obj_ind + 1) * block_size, n); ++i){
                    //Float x1 = 1, x2 = 1;
                    Float x1 = j > 0 ? data_ptr[i * inp_str + (j - 1)] : 1;
                    Float x2 = k > 0 ? data_ptr[i * inp_str + (k - 1)] : 1;
                    //if (j > 0) {
                    //    x1 = data_ptr[i * inp_str + (j - 1)];
                    //}
                    // if (k > 0) {
                    //    x2 = data_ptr[i * inp_str + (k - 1)];
                    //}
                    auto prob = proba_ptr[i] * (1 - proba_ptr[i]);
                    val += x1 * x2 * prob;
                }
                Float& out = hes_ptr[j * out_str + k];
                sycl::atomic_ref<Float, sycl::memory_order::relaxed, sycl::memory_scope::device, 
                    sycl::access::address_space::ext_intel_global_device_space>(out).fetch_add(val);
            }

        });
    });

    auto copy_event = q.submit([&](sycl::handler& cgh) {
        //cgh.depends_on(hess_deps);
        cgh.depends_on({hes_event});
        const auto range = make_range_2d(p + 1, p + 1);
        cgh.parallel_for(range, [=](sycl::id<2> idx) {
            auto j = idx[0];
            auto k = idx[1];
            if (j > k) {
                hes_ptr[k * out_str + j] = hes_ptr[j * out_str + k];
            }
            if (j == k) {
                hes_ptr[j * out_str + j] += 2 * L2;
            }
        });
    });

    return copy_event;

}

template <typename Float>
sycl::event compute_hessian2(sycl::queue& q,
                            const ndview<Float, 1>& parameters,
                            const ndview<Float, 2>& data,
                            const ndview<std::int32_t, 1>& labels,
                            const ndview<Float, 1>& probabilities,
                            ndview<Float, 2>& out_hessian,
                            Float L1,
                            Float L2,
                            const event_vector& deps) {
    const int64_t n = data.get_dimension(0);
    const int64_t p = data.get_dimension(1);

    ONEDAL_ASSERT(out_hessian.get_dimension(0) == (p + 1));
    ONEDAL_ASSERT(out_hessian.get_dimension(1) == (p + 1));

    auto data_ptr = data.get_data();
    auto hes_ptr = out_hessian.get_mutable_data();
    auto proba_ptr = probabilities.get_mutable_data();

    const auto max_wg = device_max_wg_size(q);
    const auto wg = std::min(p + 1, max_wg);
    const auto inp_str = data.get_leading_stride();
    const auto out_str = out_hessian.get_leading_stride();
    
    const auto range =  make_multiple_nd_range_2d({n, wg}, {1l, wg}); //make_range_2d({1l, wg}, {n, wg});
    auto hes_event = q.submit([&](sycl::handler& cgh){
        // basic nikita's version

        cgh.depends_on(deps);

        cgh.parallel_for(range, [=](sycl::nd_item<2> item) {
            const auto rowi = item.get_global_id(0);
            const auto coli = item.get_global_id(1);
            const auto wg2 = wg;//item.get_local_range()[1];
            const auto prob = proba_ptr[rowi] * (1 - proba_ptr[rowi]);
            for(std::int32_t i = 0; i <= p; ++i) {
                // Float x1 = 1;
                Float x1 = i > 0 ? data_ptr[rowi * inp_str + (i - 1)] : 1;
                // if (i > 0) {
                //    x1 = data_ptr[rowi * inp_str + (i - 1)];
                // }
                for(std::int32_t j = coli; j <= i; j += wg2) {
                    Float x2 = j > 0 ? data_ptr[rowi  * inp_str + (j - 1)] : 1;

                    //if (j > 0) {
                    //    x2 = data_ptr[rowi  * inp_str + (j - 1)];
                    //}
                    // const auto x2 = data_ptr[rowi * inp_str + j];
                    Float& out = hes_ptr[i * out_str + j];
                    const auto val = prob * x1 * x2;
                    sycl::atomic_ref<Float, sycl::memory_order::relaxed, sycl::memory_scope::device, 
                    sycl::access::address_space::ext_intel_global_device_space>(out).fetch_add(val);
                }
            }
        });
    });

    auto copy_event = q.submit([&](sycl::handler& cgh) {
        //cgh.depends_on(hess_deps);
        cgh.depends_on({hes_event});
        const auto range = make_range_2d(p + 1, p + 1);
        cgh.parallel_for(range, [=](sycl::id<2> idx) {
            auto j = idx[0];
            auto k = idx[1];
            if (j > k) {
                hes_ptr[k * out_str + j] = hes_ptr[j * out_str + k];
            }
            if (j == k) {
                hes_ptr[j * out_str + j] += 2 * L2;
            }
        });
    });

    return copy_event;

}


template <typename Float>
sycl::event compute_hessian3(sycl::queue& q,
                            const ndview<Float, 1>& parameters,
                            const ndview<Float, 2>& data,
                            const ndview<std::int32_t, 1>& labels,
                            const ndview<Float, 1>& probabilities,
                            ndview<Float, 2>& out_hessian,
                            Float L1,
                            Float L2,
                            const event_vector& deps) {
    // slow version 
    const int64_t n = data.get_dimension(0);
    const int64_t p = data.get_dimension(1);

    ONEDAL_ASSERT(out_hessian.get_dimension(0) == (p + 1));
    ONEDAL_ASSERT(out_hessian.get_dimension(1) == (p + 1));

    auto data_ptr = data.get_data();
    auto hes_ptr = out_hessian.get_mutable_data();
    auto proba_ptr = probabilities.get_mutable_data();

    const auto inp_str = data.get_leading_stride();
    const auto out_str = out_hessian.get_leading_stride();

    event_vector hess_deps = {};

    for (std::int64_t j = 0; j < p; ++j) {
        for (std::int64_t k = j; k < p; ++k) {
            using oneapi::dal::backend::operator+;
            auto event = q.submit([&](sycl::handler& cgh) {
                cgh.depends_on(deps);
                
                const auto range = make_range_1d(n);
                auto sumReduction =
                    sycl::reduction(hes_ptr + (j + 1) * out_str + (k + 1), sycl::plus<>());
                cgh.parallel_for(range, sumReduction, [=](sycl::id<1> idx, auto& sum) {
                    const Float prob = proba_ptr[idx];
                    sum += data_ptr[idx * inp_str + j] * data_ptr[idx * inp_str + k] * prob * (1 - prob);
                });
            });
            hess_deps.push_back(event);
        }
        auto event = q.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            const auto range = make_range_1d(n);
            auto sumReduction = sycl::reduction(hes_ptr + (j + 1), sycl::plus<>());
            cgh.parallel_for(range, sumReduction, [=](sycl::id<1> idx, auto& sum) {
                const Float prob = proba_ptr[idx];
                sum += data_ptr[idx * inp_str + j] * prob * (1 - prob);
            });
        });
        hess_deps.push_back(event);
    }
    auto event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        const auto range = make_range_1d(n);
        auto sumReduction = sycl::reduction(hes_ptr, sycl::plus<>());
        cgh.parallel_for(range, sumReduction, [=](sycl::id<1> idx, auto& sum) {
            const Float prob = proba_ptr[idx];
            sum += prob * (1 - prob);
        });
    });
    hess_deps.push_back(event);

    auto copy_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(hess_deps);
        const auto range = make_range_2d(p + 1, p + 1);
        cgh.parallel_for(range, [=](sycl::id<2> idx) {
            auto j = idx[0];
            auto k = idx[1];
            if (j < k) {
                hes_ptr[k * out_str + j] = hes_ptr[j * out_str + k];
            }
            if (j == k) {
                hes_ptr[j * out_str + j] += 2 * L2;
            }
        });
    });

    return copy_event;

}

template <typename Float>
sycl::event compute_hessian4(sycl::queue& q,
                            const ndview<Float, 1>& parameters,
                            const ndview<Float, 2>& data,
                            const ndview<std::int32_t, 1>& labels,
                            const ndview<Float, 1>& probabilities,
                            ndview<Float, 2>& out_hessian,
                            Float L1,
                            Float L2,
                            const event_vector& deps) {
    
    const std::int64_t n = data.get_dimension(0);
    const std::int64_t p = data.get_dimension(1);

    ONEDAL_ASSERT(out_hessian.get_dimension(0) == (p + 1));
    ONEDAL_ASSERT(out_hessian.get_dimension(1) == (p + 1));

    auto data_ptr = data.get_data();
    auto hes_ptr = out_hessian.get_mutable_data();
    auto proba_ptr = probabilities.get_mutable_data();

    const auto inp_str = data.get_leading_stride();
    const auto out_str = out_hessian.get_leading_stride();
    const std::int64_t block = 32;
    const auto max_wg = device_max_wg_size(q);
    const auto wg = std::min(p + 1, max_wg);

    const auto range =  make_multiple_nd_range_2d({(n + block - 1) / block, wg}, {1l, wg});

    // std::cout << "Local size: " << device_local_mem_size(q) << std::endl;

    auto hes_event = q.submit([&](sycl::handler& cgh){
        // nikita's optimized version
        std::int64_t req_sz = (p + 1) * (p + 1) * sizeof(Float) * 2;
        ONEDAL_ASSERT(req_sz < device_local_mem_size(q));
        using acc_t = sycl::local_accessor<Float>;
        acc_t tmp{(p + 1) * (p + 1), cgh};


    

        cgh.parallel_for(range, [=](sycl::nd_item<2> item) {
            const auto yid = item.get_global_id(0);
            const auto xid = item.get_global_id(1);
            const auto wg2 = wg;//item.get_local_range(1);
            for(std::int64_t i = 0; i <= p; ++i) {
                for(std::int64_t j = xid; j <= p; j += wg2) {
                    tmp[i * (p + 1) + j] = Float(0);
                }
            }
            for(std::int64_t k = 0; k < block; ++k) {
                const std::int64_t row = yid * block + k;
                const auto prob = proba_ptr[row] * (1 - proba_ptr[row]);
                if (row < n) {
                    for(std::int64_t i = 0; i <= p; ++i) {
                        // Float x1 = 1;
                        Float x1 = i > 0 ? row * inp_str + i - 1 : 1;
                        // if (i > 0) {
                        //    x1 = data_ptr[row * inp_str + i - 1];
                        //} 
                        for (std::int64_t j = xid; j <= i; j += wg2) {
                            //Float x2 = 1;
                            //if (j > 0) {
                            //    x2 = data_ptr[row * inp_str + j - 1];
                            //}
                            Float x2 = j > 0 ? data_ptr[row * inp_str + j - 1] : 1;
                            // const auto inp_j = inp_ptr[row * inp_str + j];
                            tmp[i * (p + 1) + j] += prob * x1 * x2;
                        }
                    }
                }
            }
            for (std::int64_t i = 0; i <= p; ++i) {
                for (std::int64_t j = xid; j <= i; j += wg2) {
                    const auto val = tmp[i * (p + 1) + j];
                    Float& out = hes_ptr[i * out_str + j];
                    sycl::atomic_ref<Float, sycl::memory_order::relaxed, sycl::memory_scope::device, 
                    sycl::access::address_space::ext_intel_global_device_space>(out).fetch_add(val);
                }
            }
        });
    });

    auto copy_event = q.submit([&](sycl::handler& cgh) {
        //cgh.depends_on(hess_deps);
        cgh.depends_on({hes_event});
        const auto range = make_range_2d(p + 1, p + 1);
        cgh.parallel_for(range, [=](sycl::id<2> idx) {
            auto j = idx[0];
            auto k = idx[1];
            if (j > k) {
                hes_ptr[k * out_str + j] = hes_ptr[j * out_str + k];
            }
            if (j == k) {
                hes_ptr[j * out_str + j] += 2 * L2;
            }
        });
    });

    return copy_event;

}


template <typename Float>
sycl::event compute_hessian(sycl::queue& q,
                            const ndview<Float, 1>& parameters,
                            const ndview<Float, 2>& data,
                            const ndview<std::int32_t, 1>& labels,
                            const ndview<Float, 1>& probabilities,
                            ndview<Float, 2>& out_hessian,
                            Float L1,
                            Float L2,
                            const event_vector& deps) {

    // 1 - my optimized version
    // 2 - basic nikita's version
    // 3 - my slow version
    // 4 - optimized nikita's version

    // out_hessian should be filled with zeros
    return compute_hessian1(q, parameters, data, labels, probabilities, out_hessian, L1, L2, deps);
}

#define INSTANTIATE(F)                                                               \
    template sycl::event compute_probabilities<F>(sycl::queue&,                        \
                                                const ndview<F, 1>&,                 \
                                                const ndview<F, 2>&,                 \
                                                ndview<F, 1>&,                       \
                                                const event_vector&);                \
    template sycl::event compute_logloss<F>(sycl::queue&,                            \
                                            const ndview<F, 1>&,                     \
                                            const ndview<F, 2>&,                     \
                                            const ndview<std::int32_t, 1>&,          \
                                            ndview<F, 1>&,                           \
                                            F,                                       \
                                            F,                                       \
                                            const event_vector&);                    \
    template sycl::event compute_logloss<F>(sycl::queue&,                            \
                                            const ndview<F, 1>&,                     \
                                            const ndview<F, 2>&,                     \
                                            const ndview<std::int32_t, 1>&,          \
                                            const ndview<F, 1>&,                     \
                                            ndview<F, 1>&,                           \
                                            F,                                       \
                                            F,                                       \
                                            const event_vector&);                    \
    template sycl::event compute_logloss_with_der<F>(sycl::queue&,                   \
                                                     const ndview<F, 1>&,            \
                                                     const ndview<F, 2>&,            \
                                                     const ndview<std::int32_t, 1>&, \
                                                     const ndview<F, 1>&,            \
                                                     ndview<F, 1>&,                  \
                                                     ndview<F, 1>&,                  \
                                                     F,                              \
                                                     F,                              \
                                                     const event_vector&);           \
    template sycl::event compute_derivative<F>(sycl::queue&,                    \
                                                const ndview<F, 1>&,            \
                                                const ndview<F, 2>&,            \
                                                const ndview<std::int32_t, 1>&, \
                                                const ndview<F, 1>&,            \
                                                ndview<F, 1>&,                  \
                                                F,                              \
                                                F,                              \
                                                const event_vector&);           \
    template sycl::event compute_hessian<F>(sycl::queue&,                            \
                                            const ndview<F, 1>&,                     \
                                            const ndview<F, 2>&,                     \
                                            const ndview<std::int32_t, 1>&,          \
                                            const ndview<F, 1>&,                     \
                                            ndview<F, 2>&,                           \
                                            F,                                       \
                                            F,                                       \
                                            const event_vector&);

INSTANTIATE(float);
INSTANTIATE(double);

} // namespace oneapi::dal::backend::primitives
