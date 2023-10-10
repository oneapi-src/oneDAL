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
#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::backend::primitives {

namespace pr = dal::backend::primitives;

template <typename Float>
sycl::event compute_probabilities(sycl::queue& q,
                                  const ndview<Float, 1>& parameters,
                                  const ndview<Float, 2>& data,
                                  ndview<Float, 1>& probabilities,
                                  bool fit_intercept,
                                  const event_vector& deps) {
    const std::int64_t n = data.get_dimension(0);
    const std::int64_t p = data.get_dimension(1);

    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(parameters.has_data());
    ONEDAL_ASSERT(probabilities.has_mutable_data());
    ONEDAL_ASSERT(parameters.get_dimension(0) == fit_intercept ? p + 1 : p);
    ONEDAL_ASSERT(probabilities.get_dimension(0) == n);

    auto fill_event = fill<Float>(q, probabilities, Float(1), {});
    using oneapi::dal::backend::operator+;

    Float w0 = fit_intercept ? parameters.get_slice(0, 1).at_device(q, 0l) : 0; // Poor perfomance
    ndview<Float, 1> param_suf = fit_intercept ? parameters.get_slice(1, p + 1) : parameters;

    auto event = gemv(q, data, param_suf, probabilities, Float(1), w0, { fill_event });
    auto* const prob_ptr = probabilities.get_mutable_data();

    const Float bottom = sizeof(Float) == 4 ? 1e-7 : 1e-15;
    const Float top = Float(1.0) - bottom;
    // Log Loss is undefined for p = 0 and p = 1 so probabilities are clipped into [eps, 1 - eps]

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
                            const ndview<std::int32_t, 1>& labels,
                            const ndview<Float, 1>& probabilities,
                            ndview<Float, 1>& out,
                            bool fit_intercept,
                            const event_vector& deps) {
    const std::int64_t n = labels.get_dimension(0);
    ONEDAL_ASSERT(probabilities.get_dimension(0) == n);
    ONEDAL_ASSERT(labels.has_data());
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
    return loss_event;
}

template <typename Float>
sycl::event compute_logloss_with_der(sycl::queue& q,
                                     const ndview<Float, 2>& data,
                                     const ndview<std::int32_t, 1>& labels,
                                     const ndview<Float, 1>& probabilities,
                                     ndview<Float, 1>& out,
                                     ndview<Float, 1>& out_derivative,
                                     bool fit_intercept,
                                     const event_vector& deps) {
    // out, out_derivative should be filled with zeros

    const std::int64_t n = data.get_dimension(0);
    const std::int64_t p = data.get_dimension(1);

    ONEDAL_ASSERT(labels.get_dimension(0) == n);
    ONEDAL_ASSERT(probabilities.get_dimension(0) == n);
    ONEDAL_ASSERT(out.get_dimension(0) == 1);
    ONEDAL_ASSERT(out_derivative.get_dimension(0) == fit_intercept ? p + 1 : p);

    ONEDAL_ASSERT(labels.has_data());
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(probabilities.has_data());
    ONEDAL_ASSERT(out.has_mutable_data());
    ONEDAL_ASSERT(out_derivative.has_mutable_data());

    // d loss_i / d pred_i
    auto derivative_object = ndarray<Float, 1>::empty(q, { n }, sycl::usm::alloc::device);

    auto* const der_obj_ptr = derivative_object.get_mutable_data();
    const auto* const proba_ptr = probabilities.get_data();
    const auto* const labels_ptr = labels.get_data();
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

    auto out_der_suffix = fit_intercept ? out_derivative.get_slice(1, p + 1) : out_derivative;

    return gemv(q, data.t(), derivative_object, out_der_suffix, { loss_event, derw0_event });
}

template <typename Float>
sycl::event compute_derivative(sycl::queue& q,
                               const ndview<Float, 2>& data,
                               const ndview<std::int32_t, 1>& labels,
                               const ndview<Float, 1>& probabilities,
                               ndview<Float, 1>& out_derivative,
                               bool fit_intercept,
                               const event_vector& deps) {
    // out_derivative should be filled with zeros

    const std::int64_t n = data.get_dimension(0);
    const std::int64_t p = data.get_dimension(1);

    ONEDAL_ASSERT(labels.get_dimension(0) == n);
    ONEDAL_ASSERT(probabilities.get_dimension(0) == n);
    ONEDAL_ASSERT(out_derivative.get_dimension(0) == fit_intercept ? p + 1 : p);

    ONEDAL_ASSERT(labels.has_data());
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(probabilities.has_data());
    ONEDAL_ASSERT(out_derivative.has_mutable_data());

    // d loss_i / d pred_i
    auto derivative_object = ndarray<Float, 1>::empty(q, { n }, sycl::usm::alloc::device);

    auto* const der_obj_ptr = derivative_object.get_mutable_data();
    const auto* const proba_ptr = probabilities.get_data();
    const auto* const labels_ptr = labels.get_data();
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

    auto out_der_suffix = fit_intercept ? out_derivative.get_slice(1, p + 1) : out_derivative;

    auto der_event = gemv(q, data.t(), derivative_object, out_der_suffix, { loss_event });

    return der_event;
}

template <typename Float>
sycl::event add_regularization_loss(sycl::queue& q,
                                    const ndview<Float, 1>& parameters,
                                    ndview<Float, 1>& out,
                                    Float L1,
                                    Float L2,
                                    bool fit_intercept,
                                    const event_vector& deps) {
    using dal::backend::operator+;
    auto [out_reg, out_reg_e] = ndarray<Float, 1>::zeros(q, { 1 }, sycl::usm::alloc::device);
    auto* const reg_ptr = out_reg.get_mutable_data();
    auto* const out_ptr = out.get_mutable_data();
    const auto* const param_ptr = parameters.get_data();
    auto new_deps = deps + out_reg_e;
    const std::int64_t p =
        fit_intercept ? parameters.get_dimension(0) - 1 : parameters.get_dimension(0);
    auto reg_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(new_deps);
        const auto range = make_range_1d(p);
        auto sum_reduction = sycl::reduction(reg_ptr, sycl::plus<>());
        const std::int64_t st_id = fit_intercept;
        cgh.parallel_for(range, sum_reduction, [=](sycl::id<1> idx, auto& sum) {
            const Float param = param_ptr[idx + st_id];
            sum += L1 * sycl::abs(param) + L2 * param * param;
        });
    });
    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ reg_event });
        cgh.single_task([=] {
            *out_ptr += *reg_ptr;
        });
    });
}

template <typename Float>
sycl::event add_regularization_gradient_loss(sycl::queue& q,
                                             const ndview<Float, 1>& parameters,
                                             ndview<Float, 1>& out,
                                             ndview<Float, 1>& out_derivative,
                                             Float L1,
                                             Float L2,
                                             bool fit_intercept,
                                             const event_vector& deps) {
    using dal::backend::operator+;
    auto [reg_val, reg_val_e] = ndarray<Float, 1>::zeros(q, { 1 }, sycl::usm::alloc::device);

    const std::int64_t p =
        fit_intercept ? parameters.get_dimension(0) - 1 : parameters.get_dimension(0);

    const auto* const param_ptr = parameters.get_data();
    auto* const reg_ptr = reg_val.get_mutable_data();
    auto* const out_ptr = out.get_mutable_data();
    auto* const grad_ptr = out_derivative.get_mutable_data();
    auto new_deps = deps + reg_val_e;
    auto reg_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(new_deps);
        const auto range = make_range_1d(p);
        auto sum_reduction = sycl::reduction(reg_ptr, sycl::plus<>());
        std::int64_t st_id = fit_intercept;
        cgh.parallel_for(range, sum_reduction, [=](sycl::id<1> idx, auto& sum) {
            const Float param = param_ptr[idx + st_id];
            sum += L1 * sycl::abs(param) + L2 * param * param;
            grad_ptr[idx + st_id] += L2 * 2 * param;
        });
    });

    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ reg_event });
        cgh.single_task([=] {
            *out_ptr += *reg_ptr;
        });
    });
}

template <typename Float>
sycl::event add_regularization_gradient(sycl::queue& q,
                                        const ndview<Float, 1>& parameters,
                                        ndview<Float, 1>& out_derivative,
                                        Float L1,
                                        Float L2,
                                        bool fit_intercept,
                                        const event_vector& deps) {
    auto* const grad_ptr = out_derivative.get_mutable_data();
    const auto* const param_ptr = parameters.get_data();
    const std::int64_t p =
        fit_intercept ? parameters.get_dimension(0) - 1 : parameters.get_dimension(0);
    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        const auto range = make_range_1d(p);
        std::int64_t st_id = fit_intercept;
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            const Float param = param_ptr[idx + st_id];
            grad_ptr[idx + st_id] += L2 * 2 * param;
        });
    });
}

template <typename Float>
sycl::event compute_hessian(sycl::queue& q,
                            const ndview<Float, 2>& data,
                            const ndview<std::int32_t, 1>& labels,
                            const ndview<Float, 1>& probabilities,
                            ndview<Float, 2>& out_hessian,
                            const Float L1,
                            const Float L2,
                            bool fit_intercept,
                            const event_vector& deps) {
    const int64_t n = data.get_dimension(0);
    const int64_t p = data.get_dimension(1);

    ONEDAL_ASSERT(labels.get_dimension(0) == n);
    ONEDAL_ASSERT(probabilities.get_dimension(0) == n);
    ONEDAL_ASSERT(out_hessian.get_dimension(0) == (p + 1));
    ONEDAL_ASSERT(out_hessian.get_dimension(1) == (p + 1));

    ONEDAL_ASSERT(labels.has_data());
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
    const auto kernel = [=](const Float val, Float) -> Float {
        constexpr Float one(1);
        return val * (one - val);
    };
    return element_wise(q, kernel, probabilities, Float(0), out_hessian, deps);
}

std::int64_t get_block_size(std::int64_t n, std::int64_t p) {
    constexpr std::int64_t max_alloc_size = 1 << 21;
    return p > max_alloc_size ? 512 : max_alloc_size / p;
}

template <typename Float>
LogLossHessianProduct<Float>::LogLossHessianProduct(sycl::queue& q,
                                                    const table& data,
                                                    Float L2,
                                                    bool fit_intercept,
                                                    std::int64_t bsz)
        : q_(q),
          data_(data),
          L2_(L2),
          fit_intercept_(fit_intercept),
          n_(data.get_row_count()),
          p_(data.get_column_count()),
          bsz_(bsz == -1 ? get_block_size(n_, p_) : bsz) {
    raw_hessian_ = ndarray<Float, 1>::empty(q_, { n_ }, sycl::usm::alloc::device);
    buffer_ = ndarray<Float, 1>::empty(q_, { n_ }, sycl::usm::alloc::device);
}

template <typename Float>
ndview<Float, 1>& LogLossHessianProduct<Float>::get_raw_hessian() {
    return raw_hessian_;
}

template <typename Float>
sycl::event LogLossHessianProduct<Float>::compute_with_fit_intercept(const ndview<Float, 1>& vec,
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

    // TODO: Add batch matrix-vector multiplication
    auto data_nd = table2ndarray<Float>(q_, data_, sycl::usm::alloc::device);

    sycl::event event_xv = gemv(q_, data_nd, vec_suf, buffer_, Float(1), v0, { fill_buffer_event });
    event_xv.wait_and_throw(); // Without this line gemv does not work correctly

    auto tmp_host = buffer_.to_host(q_);

    sycl::event event_dxv = q_.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ event_xv, fill_out_event });
        const auto range = make_range_1d(n_);
        auto sum_reduction = sycl::reduction(out_ptr, sycl::plus<>());
        cgh.parallel_for(range, sum_reduction, [=](sycl::id<1> idx, auto& sum_v0) {
            buffer_ptr[idx] = buffer_ptr[idx] * hess_ptr[idx];
            sum_v0 += buffer_ptr[idx];
        });
    });

    sycl::event event_xtdxv =
        gemv(q_, data_nd.t(), buffer_, out_suf, Float(1), Float(0), { event_dxv, fill_out_event });
    event_xtdxv.wait_and_throw(); // Without this line gemv does not work correctly

    const Float regularization_factor = L2_;

    const auto kernel_regularization = [=](const Float a, const Float param) {
        return a + param * regularization_factor;
    };

    auto add_regularization_event =
        element_wise(q_, kernel_regularization, out_suf, vec_suf, out_suf, { event_xtdxv });
    return add_regularization_event;
}

template <typename Float>
sycl::event LogLossHessianProduct<Float>::compute_without_fit_intercept(const ndview<Float, 1>& vec,
                                                                        ndview<Float, 1>& out,
                                                                        const event_vector& deps) {
    ONEDAL_ASSERT(vec.get_dimension(0) == p_);
    ONEDAL_ASSERT(out.get_dimension(0) == p_);

    sycl::event fill_out_event = fill<Float>(q_, out, Float(0), deps);

    // TODO: Add batch matrix-vector multiplication
    auto data_nd = table2ndarray<Float>(q_, data_, sycl::usm::alloc::device);

    sycl::event event_xv = gemv(q_, data_nd, vec, buffer_, Float(1), Float(0), deps);
    event_xv.wait_and_throw(); // Without this line gemv does not work correctly

    auto& buf_ndview = static_cast<ndview<Float, 1>&>(buffer_);
    auto& hess_ndview = static_cast<ndview<Float, 1>&>(raw_hessian_);
    constexpr sycl::multiplies<Float> kernel_mul{};
    auto event_dxv =
        element_wise(q_, kernel_mul, buf_ndview, hess_ndview, buf_ndview, { event_xv });

    sycl::event event_xtdxv =
        gemv(q_, data_nd.t(), buffer_, out, Float(1), Float(0), { event_dxv, fill_out_event });
    event_xtdxv.wait_and_throw(); // Without this line gemv does not work correctly

    const Float regularization_factor = L2_;

    const auto kernel_regularization = [=](const Float a, const Float param) {
        return a + param * regularization_factor;
    };

    auto add_regularization_event =
        element_wise(q_, kernel_regularization, out, vec, out, { event_xtdxv });

    return add_regularization_event;
}

template <typename Float>
sycl::event LogLossHessianProduct<Float>::operator()(const ndview<Float, 1>& vec,
                                                     ndview<Float, 1>& out,
                                                     const event_vector& deps) {
    if (fit_intercept_) {
        return compute_with_fit_intercept(vec, out, deps);
    }
    else {
        return compute_without_fit_intercept(vec, out, deps);
    }
}

template <typename Float>
LogLossFunction<Float>::LogLossFunction(sycl::queue q,
                                        const table& data,
                                        ndview<std::int32_t, 1>& labels,
                                        Float L2,
                                        bool fit_intercept,
                                        std::int64_t bsz)
        : q_(q),
          data_(data),
          labels_(labels),
          n_(data.get_row_count()),
          p_(data.get_column_count()),
          L2_(L2),
          fit_intercept_(fit_intercept),
          bsz_(bsz == -1 ? get_block_size(n_, p_) : bsz),
          hessp_(q, data, L2, fit_intercept, bsz_),
          dimension_(fit_intercept ? p_ + 1 : p_) {
    ONEDAL_ASSERT(labels.get_dimension(0) == n_);
    probabilities_ = ndarray<Float, 1>::empty(q_, { n_ }, sycl::usm::alloc::device);
    gradient_ = ndarray<Float, 1>::empty(q_, { dimension_ }, sycl::usm::alloc::device);
    buffer_ = ndarray<Float, 1>::empty(q_, { p_ + 2 }, sycl::usm::alloc::device);
}

template <typename Float>
event_vector LogLossFunction<Float>::update_x(const ndview<Float, 1>& x,
                                              bool need_hessp,
                                              const event_vector& deps) {
    using dal::backend::operator+;
    value_ = 0;
    auto fill_event = fill(q_, gradient_, Float(0), deps);
    const uniform_blocking blocking(n_, bsz_);

    event_vector last_iter_e = { fill_event };

    ndview<Float, 1> grad_ndview = gradient_;
    ndview<Float, 1> grad_batch = buffer_.slice(1, dimension_);
    ndview<Float, 1> loss_batch = buffer_.slice(0, 1);

    ndview<Float, 1> raw_hessian = hessp_.get_raw_hessian();

    for (std::int64_t b = 0; b < blocking.get_block_count(); ++b) {
        const auto first = blocking.get_block_start_index(b);
        const auto last = blocking.get_block_end_index(b);
        const std::int64_t cursize = last - first;

        const auto data_rows =
            row_accessor<const Float>(data_).pull(q_, { first, last }, sycl::usm::alloc::device);
        const auto data_batch = ndarray<Float, 2>::wrap(data_rows, { cursize, p_ });
        const auto labels_batch = labels_.get_slice(first, first + cursize);
        auto prob_batch = probabilities_.slice(first, cursize);
        sycl::event prob_e =
            compute_probabilities(q_, x, data_batch, prob_batch, fit_intercept_, last_iter_e);

        constexpr Float zero(0);

        auto fill_buffer_e = fill(q_, buffer_, zero, last_iter_e);

        sycl::event compute_e = compute_logloss_with_der(q_,
                                                         data_batch,
                                                         labels_batch,
                                                         prob_batch,
                                                         loss_batch,
                                                         grad_batch,
                                                         fit_intercept_,
                                                         { fill_buffer_e, prob_e });

        sycl::event update_grad_e =
            element_wise(q_, sycl::plus<>(), grad_ndview, grad_batch, grad_ndview, { compute_e });

        value_ += loss_batch.at_device(q_, 0, { compute_e });

        last_iter_e = { update_grad_e };

        if (need_hessp) {
            auto raw_hessian_batch = raw_hessian.get_slice(first, first + cursize);
            auto hess_e = compute_raw_hessian(q_, prob_batch, raw_hessian_batch, { prob_e });
            last_iter_e = last_iter_e + hess_e;
        }

        // TODO: Delete this wait_and_throw
        // ensure that while event is running in the background data is not overwritten
        wait_or_pass(last_iter_e).wait_and_throw();
    }

    if (L2_ > 0) {
        auto fill_loss_e = fill(q_, loss_batch, Float(0), { last_iter_e });
        auto loss_ptr = loss_batch.get_mutable_data();
        auto grad_ptr = gradient_.get_mutable_data();
        auto w_ptr = x.get_data();
        Float regularization_factor = L2_;

        auto regularization_e = q_.submit([&](sycl::handler& cgh) {
            cgh.depends_on(last_iter_e + fill_loss_e);
            const auto range = make_range_1d(p_);
            const std::int64_t st_id = fit_intercept_;
            auto sum_reduction = sycl::reduction(loss_ptr, sycl::plus<>());
            cgh.parallel_for(range, sum_reduction, [=](sycl::id<1> idx, auto& sum_v0) {
                const Float param = w_ptr[st_id + idx];
                grad_ptr[st_id + idx] += regularization_factor * param;
                sum_v0 += regularization_factor * param * param / 2;
            });
        });

        value_ += loss_batch.at_device(q_, 0, { regularization_e });

        last_iter_e = { regularization_e };
    }

    return last_iter_e;
}

template <typename Float>
Float LogLossFunction<Float>::get_value() {
    return value_;
}
template <typename Float>
ndview<Float, 1>& LogLossFunction<Float>::get_gradient() {
    return gradient_;
}

template <typename Float>
BaseMatrixOperator<Float>& LogLossFunction<Float>::get_hessian_product() {
    return hessp_;
}

#define INSTANTIATE(F)                                                               \
    template sycl::event compute_probabilities<F>(sycl::queue&,                      \
                                                  const ndview<F, 1>&,               \
                                                  const ndview<F, 2>&,               \
                                                  ndview<F, 1>&,                     \
                                                  bool,                              \
                                                  const event_vector&);              \
    template sycl::event compute_logloss<F>(sycl::queue&,                            \
                                            const ndview<std::int32_t, 1>&,          \
                                            const ndview<F, 1>&,                     \
                                            ndview<F, 1>&,                           \
                                            bool,                                    \
                                            const event_vector&);                    \
    template sycl::event compute_logloss_with_der<F>(sycl::queue&,                   \
                                                     const ndview<F, 2>&,            \
                                                     const ndview<std::int32_t, 1>&, \
                                                     const ndview<F, 1>&,            \
                                                     ndview<F, 1>&,                  \
                                                     ndview<F, 1>&,                  \
                                                     bool,                           \
                                                     const event_vector&);           \
    template sycl::event compute_derivative<F>(sycl::queue&,                         \
                                               const ndview<F, 2>&,                  \
                                               const ndview<std::int32_t, 1>&,       \
                                               const ndview<F, 1>&,                  \
                                               ndview<F, 1>&,                        \
                                               bool,                                 \
                                               const event_vector&);                 \
    template sycl::event add_regularization_loss<F>(sycl::queue&,                    \
                                                    const ndview<F, 1>&,             \
                                                    ndview<F, 1>&,                   \
                                                    F,                               \
                                                    F,                               \
                                                    bool,                            \
                                                    const event_vector&);            \
    template sycl::event add_regularization_gradient_loss<F>(sycl::queue&,           \
                                                             const ndview<F, 1>&,    \
                                                             ndview<F, 1>&,          \
                                                             ndview<F, 1>&,          \
                                                             F,                      \
                                                             F,                      \
                                                             bool,                   \
                                                             const event_vector&);   \
    template sycl::event add_regularization_gradient<F>(sycl::queue&,                \
                                                        const ndview<F, 1>&,         \
                                                        ndview<F, 1>&,               \
                                                        F,                           \
                                                        F,                           \
                                                        bool,                        \
                                                        const event_vector&);        \
    template sycl::event compute_hessian<F>(sycl::queue&,                            \
                                            const ndview<F, 2>&,                     \
                                            const ndview<std::int32_t, 1>&,          \
                                            const ndview<F, 1>&,                     \
                                            ndview<F, 2>&,                           \
                                            const F,                                 \
                                            const F,                                 \
                                            bool,                                    \
                                            const event_vector&);                    \
    template sycl::event compute_raw_hessian<F>(sycl::queue&,                        \
                                                const ndview<F, 1>&,                 \
                                                ndview<F, 1>&,                       \
                                                const event_vector&);                \
    template class LogLossHessianProduct<F>;                                         \
    template class LogLossFunction<F>;

INSTANTIATE(float);
INSTANTIATE(double);

} // namespace oneapi::dal::backend::primitives
