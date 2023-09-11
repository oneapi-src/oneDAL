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

#include "oneapi/dal/backend/primitives/objective_function/logloss.hpp"
#include "oneapi/dal/backend/primitives/blas/gemv.hpp"
#include "oneapi/dal/backend/primitives/element_wise.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
sycl::event compute_probabilities(sycl::queue& q,
                                  const ndview<Float, 1>& parameters,
                                  const ndview<Float, 2>& data,
                                  ndview<Float, 1>& probabilities,
                                  const bool fit_intercept,
                                  const event_vector& deps) {
    const std::int64_t n = data.get_dimension(0);
    const std::int64_t p = data.get_dimension(1);
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(parameters.has_data());
    ONEDAL_ASSERT(probabilities.has_mutable_data());
    ONEDAL_ASSERT(parameters.get_dimension(0) == p + 1);
    ONEDAL_ASSERT(probabilities.get_dimension(0) == n);

    auto fill_event = fill<Float>(q, probabilities, Float(1), {});
    using oneapi::dal::backend::operator+;

    auto param_arr = ndarray<Float, 1>::wrap(parameters.get_data(), 1);
    Float w0 = fit_intercept ? param_arr.slice(0, 1).to_host(q, deps).at(0) : 0; // Poor perfomance

    auto event = gemv(q,
                      data,
                      parameters.get_slice(1, parameters.get_dimension(0)),
                      probabilities,
                      Float(1),
                      w0,
                      { fill_event });
    auto* const prob_ptr = probabilities.get_mutable_data();

    const Float bottom = sizeof(Float) == 4 ? 1e-7 : 1e-15;
    const Float top = Float(1.0) - bottom;
    // Log Loss is undefined fot p = 0 and p = 1 so probabilities are clipped into [eps, 1 - eps]

    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(event);
        const auto range = make_range_1d(n);
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            prob_ptr[idx] = 1 / (1 + sycl::exp(-prob_ptr[idx]));
            if (prob_ptr[idx] < bottom) {
                prob_ptr[idx] = bottom;
            }
            if (prob_ptr[idx] > top) {
                prob_ptr[idx] = top;
            }
        });
    });
}

template <typename Float>
sycl::event compute_logloss(sycl::queue& q,
                            const ndview<Float, 1>& parameters,
                            const ndview<Float, 2>& data,
                            const ndview<std::int32_t, 1>& labels,
                            const ndview<Float, 1>& probabilities,
                            ndview<Float, 1>& out,
                            const Float L1,
                            const Float L2,
                            const bool fit_intercept,
                            const event_vector& deps) {
    const std::int64_t n = data.get_dimension(0);
    const std::int64_t p = data.get_dimension(1);
    ONEDAL_ASSERT(parameters.get_dimension(0) == p + 1);
    ONEDAL_ASSERT(labels.get_dimension(0) == n);
    ONEDAL_ASSERT(probabilities.get_dimension(0) == n);
    ONEDAL_ASSERT(labels.has_data());
    ONEDAL_ASSERT(parameters.has_data());
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(probabilities.has_data());

    const auto* const labels_ptr = labels.get_data();
    const auto* const prob_ptr = probabilities.get_data();

    auto* const out_ptr = out.get_mutable_data();

    auto loss_event = q.submit([&](sycl::handler& cgh) {
        const auto range = make_range_1d(n);
        using oneapi::dal::backend::operator+;
        using sycl::reduction;

        cgh.depends_on(deps);

        auto sum_reduction = reduction(out_ptr, sycl::plus<>());

        cgh.parallel_for(range, sum_reduction, [=](sycl::id<1> idx, auto& sum) {
            const Float prob = prob_ptr[idx];
            const std::int32_t label = labels_ptr[idx];
            sum += -label * sycl::log(prob) - (1 - label) * sycl::log(1 - prob);
        });
    });

    auto [out_reg, out_reg_e] = ndarray<Float, 1>::zeros(q, { 1 }, sycl::usm::alloc::device);
    auto* const reg_ptr = out_reg.get_mutable_data();
    const event_vector vector_out_reg = { out_reg_e };

    const auto* const param_ptr = parameters.get_data();

    if (L1 > 0 || L2 > 0) {
        auto reg_event = q.submit([&](sycl::handler& cgh) {
            cgh.depends_on(vector_out_reg);
            const auto range = make_range_1d(p);
            auto sum_reduction = sycl::reduction(reg_ptr, sycl::plus<>());
            cgh.parallel_for(range, sum_reduction, [=](sycl::id<1> idx, auto& sum) {
                const Float param = param_ptr[idx + 1];
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
    return loss_event;
}

template <typename Float>
sycl::event compute_logloss(sycl::queue& q,
                            const ndview<Float, 1>& parameters,
                            const ndview<Float, 2>& data,
                            const ndview<std::int32_t, 1>& labels,
                            ndview<Float, 1>& out,
                            const Float L1,
                            const Float L2,
                            const bool fit_intercept,
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
    auto prediction_event =
        compute_probabilities(q, parameters, data, probabilities, fit_intercept, deps);

    return compute_logloss(q,
                           parameters,
                           data,
                           labels,
                           probabilities,
                           out,
                           L1,
                           L2,
                           fit_intercept,
                           { prediction_event });
}

template <typename Float>
sycl::event compute_logloss_with_der(sycl::queue& q,
                                     const ndview<Float, 1>& parameters,
                                     const ndview<Float, 2>& data,
                                     const ndview<std::int32_t, 1>& labels,
                                     const ndview<Float, 1>& probabilities,
                                     ndview<Float, 1>& out,
                                     ndview<Float, 1>& out_derivative,
                                     const Float L1,
                                     const Float L2,
                                     const bool fit_intercept,
                                     const event_vector& deps) {
    // out, out_derivative should be filled with zeros

    const std::int64_t n = data.get_dimension(0);
    const std::int64_t p = data.get_dimension(1);

    ONEDAL_ASSERT(parameters.get_dimension(0) == p + 1);
    ONEDAL_ASSERT(labels.get_dimension(0) == n);
    ONEDAL_ASSERT(probabilities.get_dimension(0) == n);
    ONEDAL_ASSERT(out.get_count() == 1);
    ONEDAL_ASSERT(out_derivative.get_count() == p + 1);

    ONEDAL_ASSERT(labels.has_data());
    ONEDAL_ASSERT(parameters.has_data());
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(probabilities.has_data());
    ONEDAL_ASSERT(out.has_mutable_data());
    ONEDAL_ASSERT(out_derivative.has_mutable_data());

    // d loss_i / d pred_i
    auto derivative_object = ndarray<Float, 1>::empty(q, { n }, sycl::usm::alloc::device);

    auto* const der_obj_ptr = derivative_object.get_mutable_data();
    const auto* const proba_ptr = probabilities.get_data();
    const auto* const labels_ptr = labels.get_data();
    const auto* const param_ptr = parameters.get_data();
    auto* const out_ptr = out.get_mutable_data();
    auto* const out_derivative_ptr = out_derivative.get_mutable_data();

    auto loss_event = q.submit([&](sycl::handler& cgh) {
        using oneapi::dal::backend::operator+;
        using sycl::reduction;

        cgh.depends_on(deps);
        auto sum_reduction_logloss = reduction(out_ptr, sycl::plus<>());
        const auto wg_size = propose_wg_size(q);
        const auto range = make_multiple_nd_range_1d(n, wg_size);

        cgh.parallel_for(range, sum_reduction_logloss, [=](sycl::nd_item<1> id, auto& sum_logloss) {
            auto idx = id.get_group_linear_id() * wg_size + id.get_local_linear_id();
            if (idx >= std::size_t(n))
                return;
            const Float prob = proba_ptr[idx];
            const float label = labels_ptr[idx];
            sum_logloss += -label * sycl::log(prob) - (1 - label) * sycl::log(1 - prob);
            der_obj_ptr[idx] = prob - label;
        });
    });
    sycl::event derw0_event = sycl::event{};
    if (fit_intercept) {
        derw0_event = q.submit([&](sycl::handler& cgh) {
            using oneapi::dal::backend::operator+;
            using sycl::reduction;

            cgh.depends_on(deps + loss_event);
            auto sum_reduction_derivative_w0 = reduction(out_derivative_ptr, sycl::plus<>());
            const auto wg_size = propose_wg_size(q);
            const auto range = make_multiple_nd_range_1d(n, wg_size);

            cgh.parallel_for(range,
                             sum_reduction_derivative_w0,
                             [=](sycl::nd_item<1> id, auto& sum_dw0) {
                                 auto idx =
                                     id.get_group_linear_id() * wg_size + id.get_local_linear_id();
                                 if (idx >= std::size_t(n))
                                     return;
                                 sum_dw0 += der_obj_ptr[idx];
                             });
        });
    }
    auto out_der_suffix = out_derivative.get_slice(1, p + 1);

    auto der_event = gemv(q, data.t(), derivative_object, out_der_suffix, { loss_event });
    if (L1 == 0 && L2 == 0) {
        return der_event;
    }
    auto [reg_val, reg_val_e] = ndarray<Float, 1>::zeros(q, { 1 }, sycl::usm::alloc::device);

    const event_vector reg_deps = { reg_val_e, der_event };
    auto* const reg_ptr = reg_val.get_mutable_data();

    auto reg_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(reg_deps);
        const auto range = make_range_1d(p);
        auto sum_reduction = sycl::reduction(reg_ptr, sycl::plus<>());
        cgh.parallel_for(range, sum_reduction, [=](sycl::id<1> idx, auto& sum) {
            const Float param = param_ptr[idx + 1];
            sum += L1 * sycl::abs(param) + L2 * param * param;
            out_derivative_ptr[idx + 1] += L2 * 2 * param;
        });
    });

    auto final_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ reg_event, loss_event, derw0_event });
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
                               const Float L1,
                               const Float L2,
                               const bool fit_intercept,
                               const event_vector& deps) {
    // out_derivative should be filled with zeros

    const std::int64_t n = data.get_dimension(0);
    const std::int64_t p = data.get_dimension(1);

    ONEDAL_ASSERT(parameters.get_dimension(0) == p + 1);
    ONEDAL_ASSERT(labels.get_dimension(0) == n);
    ONEDAL_ASSERT(probabilities.get_dimension(0) == n);
    ONEDAL_ASSERT(out_derivative.get_count() == p + 1);

    ONEDAL_ASSERT(labels.has_data());
    ONEDAL_ASSERT(parameters.has_data());
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(probabilities.has_data());
    ONEDAL_ASSERT(out_derivative.has_mutable_data());

    // d loss_i / d pred_i
    auto derivative_object = ndarray<Float, 1>::empty(q, { n }, sycl::usm::alloc::device);

    auto* const der_obj_ptr = derivative_object.get_mutable_data();
    const auto* const proba_ptr = probabilities.get_data();
    const auto* const labels_ptr = labels.get_data();
    const auto* const param_ptr = parameters.get_data();
    auto* const out_derivative_ptr = out_derivative.get_mutable_data();

    auto loss_event = q.submit([&](sycl::handler& cgh) {
        using sycl::reduction;

        cgh.depends_on(deps);
        const auto wg_size = propose_wg_size(q);
        const auto range = make_multiple_nd_range_1d(n, wg_size);
        if (fit_intercept) {
            auto sum_reduction_derivative_w0 = reduction(out_derivative_ptr, sycl::plus<>());
            cgh.parallel_for(range,
                             sum_reduction_derivative_w0,
                             [=](sycl::nd_item<1> id, auto& sum_dw0) {
                                 auto idx =
                                     id.get_group_linear_id() * wg_size + id.get_local_linear_id();
                                 if (idx >= std::size_t(n))
                                     return;
                                 const Float prob = proba_ptr[idx];
                                 const Float label = labels_ptr[idx];
                                 der_obj_ptr[idx] = prob - label;
                                 sum_dw0 += der_obj_ptr[idx];
                             });
        }
        else {
            cgh.parallel_for(range, [=](sycl::nd_item<1> id) {
                auto idx = id.get_group_linear_id() * wg_size + id.get_local_linear_id();
                if (idx >= std::size_t(n))
                    return;
                const Float prob = proba_ptr[idx];
                const Float label = labels_ptr[idx];
                der_obj_ptr[idx] = prob - label;
            });
        }
    });

    auto out_der_suffix = out_derivative.get_slice(1, p + 1);

    auto der_event = gemv(q, data.t(), derivative_object, out_der_suffix, { loss_event });

    if (L1 == 0 && L2 == 0) {
        return der_event;
    }

    auto reg_event = q.submit([&](sycl::handler& cgh) {
        using oneapi::dal::backend::operator+;
        cgh.depends_on({ der_event });
        const auto range = make_range_1d(p);
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            const Float param = param_ptr[idx + 1];
            out_derivative_ptr[idx + 1] += L2 * 2 * param;
        });
    });

    return reg_event;
}

template <typename Float>
sycl::event compute_hessian(sycl::queue& q,
                            const ndview<Float, 1>& parameters,
                            const ndview<Float, 2>& data,
                            const ndview<std::int32_t, 1>& labels,
                            const ndview<Float, 1>& probabilities,
                            ndview<Float, 2>& out_hessian,
                            const Float L1,
                            const Float L2,
                            const bool fit_intercept,
                            const event_vector& deps) {
    const int64_t n = data.get_dimension(0);
    const int64_t p = data.get_dimension(1);

    ONEDAL_ASSERT(parameters.get_dimension(0) == p + 1);
    ONEDAL_ASSERT(labels.get_dimension(0) == n);
    ONEDAL_ASSERT(probabilities.get_dimension(0) == n);
    ONEDAL_ASSERT(out_hessian.get_dimension(0) == (p + 1));
    ONEDAL_ASSERT(out_hessian.get_dimension(1) == (p + 1));

    ONEDAL_ASSERT(labels.has_data());
    ONEDAL_ASSERT(parameters.has_data());
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(probabilities.has_data());
    ONEDAL_ASSERT(out_hessian.has_mutable_data());

    const auto* const data_ptr = data.get_data();
    auto* const hes_ptr = out_hessian.get_mutable_data();
    const auto* const proba_ptr = probabilities.get_data();

    const auto max_wg = device_max_wg_size(q);
    const auto wg = std::min(p + 1, max_wg);
    const auto inp_str = data.get_leading_stride();
    const auto out_str = out_hessian.get_leading_stride();

    constexpr std::int64_t block_size = 32;
    const std::int64_t num_blocks = (n + block_size - 1) / block_size;
    const auto range = make_multiple_nd_range_3d({ num_blocks, p + 1, wg }, { 1, 1, wg });

    auto hes_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::nd_item<3> item) {
            const std::int64_t obj_ind = item.get_global_id(0);
            const auto j = item.get_global_id(1);
            const auto param_ind_2 = item.get_global_id(2);
            Float val = 0;
            for (auto k = param_ind_2; k < j + 1; k += wg) {
                if (!fit_intercept && (j == 0 || k == 0)) {
                    continue;
                }
                val = 0;
                const std::int64_t last_ind = std::min((obj_ind + 1) * block_size, n);
                for (auto i = obj_ind * block_size; i < last_ind; ++i) {
                    const Float x1 = j > 0 ? data_ptr[i * inp_str + (j - 1)] : 1;
                    const Float x2 = k > 0 ? data_ptr[i * inp_str + (k - 1)] : 1;
                    const Float prob = proba_ptr[i] * (1 - proba_ptr[i]);
                    val += x1 * x2 * prob;
                }
                Float& out = hes_ptr[j * out_str + k];
                sycl::atomic_ref<Float,
                                 sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::ext_intel_global_device_space>(out)
                    .fetch_add(val);
            }
        });
    });

    auto make_symmetric = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ hes_event });
        const auto range = make_range_2d(p + 1, p + 1);
        cgh.parallel_for(range, [=](sycl::id<2> idx) {
            auto j = idx[0];
            auto k = idx[1];
            if (j > k) {
                hes_ptr[k * out_str + j] = hes_ptr[j * out_str + k];
            }
            else if (j == k && j > 0) {
                hes_ptr[j * out_str + j] += 2 * L2;
            }
        });
    });

    return make_symmetric;
}

template <typename Float>
sycl::event compute_raw_hessian(sycl::queue& q,
                                const ndview<Float, 1>& probabilities,
                                ndview<Float, 1>& out_hessian,
                                const event_vector& deps) {
    const std::int64_t n = probabilities.get_dimension(0);

    ONEDAL_ASSERT(out_hessian.get_dimension(0) == n);
    ONEDAL_ASSERT(probabilities.has_data());
    ONEDAL_ASSERT(out_hessian.has_mutable_data());

    const auto kernel = [=](const Float& val, Float*) -> Float {
        constexpr Float one(1);
        return val * (one - val);
    };

    return element_wise(q, kernel, probabilities, nullptr, out_hessian, deps);
}

template <typename Float>
logloss_hessian_product<Float>::logloss_hessian_product(sycl::queue& q,
                                                        const ndview<Float, 2>& data,
                                                        const Float L2,
                                                        const bool fit_intercept)
        : q_(q),
          data_(data),
          L2_{ L2 },
          fit_intercept_{ fit_intercept },
          n_{ data.get_dimension(0) },
          p_{ data.get_dimension(1) } {
    raw_hessian_ = ndarray<Float, 1>::empty(q_, { n_ });
    buffer_ = ndarray<Float, 1>::empty(q_, { n_ });
}

template <typename Float>
sycl::event logloss_hessian_product<Float>::set_raw_hessian(const ndview<Float, 1>& raw_hessian,
                                                            const event_vector& deps) {
    ONEDAL_ASSERT(raw_hessian.get_dimension(0) == n_);
    return copy(q_, raw_hessian_, raw_hessian, deps);
}

template <typename Float>
ndview<Float, 1>& logloss_hessian_product<Float>::get_raw_hessian() {
    return raw_hessian_;
}

template <typename Float>
sycl::event logloss_hessian_product<Float>::compute_with_fit_intercept(const ndview<Float, 1>& vec,
                                                                       ndview<Float, 1>& out,
                                                                       const event_vector& deps) {
    auto* const buffer_ptr = buffer_.get_mutable_data();
    const auto* const hess_ptr = raw_hessian_.get_data();
    auto* const out_ptr = out.get_mutable_data();
    ONEDAL_ASSERT(vec.get_dimension(0) == p_ + 1);
    ONEDAL_ASSERT(out.get_dimension(0) == p_ + 1);
    auto fill_buffer_event = fill<Float>(q_, buffer_, Float(1), deps);
    auto out_suf = out.get_slice(1, p_ + 1);
    auto out_bias = out.get_slice(0, 1);
    auto vec_suf = vec.get_slice(1, p_ + 1);

    sycl::event fill_out_event = fill<Float>(q_, out, Float(0), deps);

    Float v0 = vec.at_device(q_, 0, deps);
    sycl::event event_xv = gemv(q_, data_, vec_suf, buffer_, Float(1), v0, { fill_buffer_event });

    sycl::event event_dxv = q_.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ event_xv, fill_out_event });
        const auto range = make_range_1d(n_);
        auto sum_reduction = sycl::reduction(out_ptr, sycl::plus<>());
        cgh.parallel_for(range, sum_reduction, [=](sycl::id<1> idx, auto& sum_v0) {
            buffer_ptr[idx] = buffer_ptr[idx] * hess_ptr[idx];
            sum_v0 += buffer_ptr[idx];
        });
    });
    auto event_xtdxv =
        gemv(q_, data_.t(), buffer_, out_suf, Float(1), Float(0), { event_dxv, fill_out_event });

    const Float regularization_factor = L2_ * 2;

    const auto kernel_regularization = [=](const Float a, const Float param) {
        return a + param * regularization_factor;
    };

    auto add_regularization_event =
        element_wise(q_, kernel_regularization, out_suf, vec_suf, out_suf, { event_xtdxv });
    return add_regularization_event;
}

template <typename Float>
sycl::event logloss_hessian_product<Float>::compute_without_fit_intercept(
    const ndview<Float, 1>& vec,
    ndview<Float, 1>& out,
    const event_vector& deps) {
    ONEDAL_ASSERT(vec.get_dimension(0) == p_);
    ONEDAL_ASSERT(out.get_dimension(0) == p_);

    sycl::event fill_out_event = fill<Float>(q_, out, Float(0), deps);

    auto event_xv = gemv(q_, data_, vec, buffer_, Float(1), Float(0), deps);

    auto& buf_ndview = static_cast<ndview<Float, 1>&>(buffer_);
    auto& hess_ndview = static_cast<ndview<Float, 1>&>(raw_hessian_);
    constexpr sycl::multiplies<Float> kernel_mul{};
    auto event_dxv =
        element_wise(q_, kernel_mul, buf_ndview, hess_ndview, buf_ndview, { event_xv });

    auto event_xtdxv =
        gemv(q_, data_.t(), buffer_, out, Float(1), Float(0), { event_dxv, fill_out_event });

    const Float regularization_factor = L2_ * 2;

    const auto kernel_regularization = [=](const Float a, const Float param) {
        return a + param * regularization_factor;
    };

    auto add_regularization_event =
        element_wise(q_, kernel_regularization, out, vec, out, { event_xtdxv });

    return add_regularization_event;
}

template <typename Float>
sycl::event logloss_hessian_product<Float>::operator()(const ndview<Float, 1>& vec,
                                                       ndview<Float, 1>& out,
                                                       const event_vector& deps) {
    if (fit_intercept_) {
        return compute_with_fit_intercept(vec, out, deps);
    }
    else {
        return compute_without_fit_intercept(vec, out, deps);
    }
}

#define INSTANTIATE(F)                                                               \
    template sycl::event compute_probabilities<F>(sycl::queue&,                      \
                                                  const ndview<F, 1>&,               \
                                                  const ndview<F, 2>&,               \
                                                  ndview<F, 1>&,                     \
                                                  const bool,                        \
                                                  const event_vector&);              \
    template sycl::event compute_logloss<F>(sycl::queue&,                            \
                                            const ndview<F, 1>&,                     \
                                            const ndview<F, 2>&,                     \
                                            const ndview<std::int32_t, 1>&,          \
                                            ndview<F, 1>&,                           \
                                            const F,                                 \
                                            const F,                                 \
                                            const bool,                              \
                                            const event_vector&);                    \
    template sycl::event compute_logloss<F>(sycl::queue&,                            \
                                            const ndview<F, 1>&,                     \
                                            const ndview<F, 2>&,                     \
                                            const ndview<std::int32_t, 1>&,          \
                                            const ndview<F, 1>&,                     \
                                            ndview<F, 1>&,                           \
                                            const F,                                 \
                                            const F,                                 \
                                            const bool,                              \
                                            const event_vector&);                    \
    template sycl::event compute_logloss_with_der<F>(sycl::queue&,                   \
                                                     const ndview<F, 1>&,            \
                                                     const ndview<F, 2>&,            \
                                                     const ndview<std::int32_t, 1>&, \
                                                     const ndview<F, 1>&,            \
                                                     ndview<F, 1>&,                  \
                                                     ndview<F, 1>&,                  \
                                                     const F,                        \
                                                     const F,                        \
                                                     const bool,                     \
                                                     const event_vector&);           \
    template sycl::event compute_derivative<F>(sycl::queue&,                         \
                                               const ndview<F, 1>&,                  \
                                               const ndview<F, 2>&,                  \
                                               const ndview<std::int32_t, 1>&,       \
                                               const ndview<F, 1>&,                  \
                                               ndview<F, 1>&,                        \
                                               const F,                              \
                                               const F,                              \
                                               const bool,                           \
                                               const event_vector&);                 \
    template sycl::event compute_hessian<F>(sycl::queue&,                            \
                                            const ndview<F, 1>&,                     \
                                            const ndview<F, 2>&,                     \
                                            const ndview<std::int32_t, 1>&,          \
                                            const ndview<F, 1>&,                     \
                                            ndview<F, 2>&,                           \
                                            const F,                                 \
                                            const F,                                 \
                                            const bool,                              \
                                            const event_vector&);                    \
    template sycl::event compute_raw_hessian<F>(sycl::queue&,                        \
                                                const ndview<F, 1>&,                 \
                                                ndview<F, 1>&,                       \
                                                const event_vector&);                \
    template class logloss_hessian_product<F>;

INSTANTIATE(float);
INSTANTIATE(double);

} // namespace oneapi::dal::backend::primitives
