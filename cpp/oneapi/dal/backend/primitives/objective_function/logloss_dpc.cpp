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

namespace oneapi::dal::backend::primitives {

template <typename Float>
sycl::event compute_probabilities(sycl::queue& q,
                                  const ndview<Float, 1>& parameters,
                                  const ndview<Float, 2>& data,
                                  ndview<Float, 1>& probabilities,
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
    Float w0 = param_arr.slice(0, 1).to_host(q, deps).at(0); // Poor perfomance

    auto event = gemv(q,
                      data,
                      parameters.get_slice(1, parameters.get_dimension(0)),
                      probabilities,
                      Float(1),
                      w0,
                      { fill_event });
    auto* const prob_ptr = probabilities.get_mutable_data();

    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(event);
        const auto range = make_range_1d(n);
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            prob_ptr[idx] = 1 / (1 + sycl::exp(-prob_ptr[idx]));
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
                            Float L1,
                            Float L2,
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
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            const Float prob = prob_ptr[idx];
            const std::int32_t label = labels_ptr[idx];
            Float& out = *out_ptr;
            sycl::atomic_ref<Float,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::ext_intel_global_device_space>(out)
                .fetch_add(-label * sycl::log(prob) - (1 - label) * sycl::log(1 - prob));
        });
    });

    const auto* const param_ptr = parameters.get_data();

    if (L1 > 0 || L2 > 0) {
        auto reg_event = q.submit([&](sycl::handler& cgh) {
            cgh.depends_on({ loss_event });
            const auto range = make_range_1d(p);
            cgh.parallel_for(range, [=](sycl::id<1> idx) {
                const Float param = param_ptr[idx + 1];
                Float& out = *out_ptr;
                sycl::atomic_ref<Float,
                                 sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::ext_intel_global_device_space>(out)
                    .fetch_add(L1 * sycl::abs(param) + L2 * param * param);
            });
        });
        return reg_event;
    }
    return loss_event;
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

    return compute_logloss(q,
                           parameters,
                           data,
                           labels,
                           probabilities,
                           out,
                           L1,
                           L2,
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
                                     Float L1,
                                     Float L2,
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

        cgh.depends_on(deps);
        const auto wg_size = propose_wg_size(q);
        const auto range = make_multiple_nd_range_1d(n, wg_size);

        cgh.parallel_for(range, [=](sycl::nd_item<1> id) {
            auto idx = id.get_group_linear_id() * wg_size + id.get_local_linear_id();
            if (idx >= std::size_t(n))
                return;
            const Float prob = proba_ptr[idx];
            const float label = labels_ptr[idx];

            Float& out_logloss = *out_ptr;
            Float& out_der = *out_derivative_ptr;
            sycl::atomic_ref<Float,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::ext_intel_global_device_space>(
                out_logloss)
                .fetch_add(-label * sycl::log(prob) - (1 - label) * sycl::log(1 - prob));

            der_obj_ptr[idx] = prob - label;

            sycl::atomic_ref<Float,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::ext_intel_global_device_space>(out_der)
                .fetch_add(der_obj_ptr[idx]);
        });
    });

    auto out_der_suffix = out_derivative.get_slice(1, p + 1);

    auto der_event = gemv(q, data.t(), derivative_object, out_der_suffix, { loss_event });
    if (L1 == 0 && L2 == 0) {
        return der_event;
    }

    auto reg_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ loss_event, der_event });
        const auto range = make_range_1d(p);
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            const Float param = param_ptr[idx + 1];
            Float& out_logloss = *out_ptr;
            sycl::atomic_ref<Float,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::ext_intel_global_device_space>(
                out_logloss)
                .fetch_add(L1 * sycl::abs(param) + L2 * param * param);
            out_derivative_ptr[idx + 1] += L2 * 2 * param;
        });
    });

    return reg_event;
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
        cgh.depends_on(deps);
        const auto wg_size = propose_wg_size(q);
        const auto range = make_multiple_nd_range_1d(n, wg_size);

        cgh.parallel_for(range, [=](sycl::nd_item<1> id) {
            auto idx = id.get_group_linear_id() * wg_size + id.get_local_linear_id();
            if (idx >= std::size_t(n))
                return;
            const Float prob = proba_ptr[idx];
            const Float label = labels_ptr[idx];
            der_obj_ptr[idx] = prob - label;
            Float& out_der = *out_derivative_ptr;
            sycl::atomic_ref<Float,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::ext_intel_global_device_space>(out_der)
                .fetch_add(der_obj_ptr[idx]);
        });
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
                            Float L1,
                            Float L2,
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

#define INSTANTIATE(F)                                                               \
    template sycl::event compute_probabilities<F>(sycl::queue&,                      \
                                                  const ndview<F, 1>&,               \
                                                  const ndview<F, 2>&,               \
                                                  ndview<F, 1>&,                     \
                                                  const event_vector&);              \
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
    template sycl::event compute_derivative<F>(sycl::queue&,                         \
                                               const ndview<F, 1>&,                  \
                                               const ndview<F, 2>&,                  \
                                               const ndview<std::int32_t, 1>&,       \
                                               const ndview<F, 1>&,                  \
                                               ndview<F, 1>&,                        \
                                               F,                                    \
                                               F,                                    \
                                               const event_vector&);                 \
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
