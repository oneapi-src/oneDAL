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

#pragma once

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/algo/svm/backend/gpu/misc.hpp"

namespace oneapi::dal::svm::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace pr = dal::backend::primitives;

using sycl::ext::oneapi::maximum;
using sycl::ext::oneapi::minimum;

template <typename Data>
using local_accessor_rw_t =
    sycl::accessor<Data, 1, sycl::access::mode::read_write, sycl::access::target::local>;

template <typename Float>
struct enumerate_value {
    std::uint32_t index;
    Float value;
};

template <typename Float>
struct local_variables {
    Float delta_b_i;
    Float delta_b_j;
    Float local_diff;
    Float local_eps;
    enumerate_value<Float> max_val_ind;
};

template <typename Float>
inline void reduce_arg_max(sycl::nd_item<1> item,
                           const Float* objective_func,
                           enumerate_value<Float>& result) {
    auto wg = item.get_group();
    const std::uint32_t local_id = item.get_local_id(0);

    const std::uint32_t int_max = dal::detail::limits<std::uint32_t>::max();

    Float x = objective_func[local_id];
    std::uint32_t x_index = local_id;

    Float res_max = sycl::reduce_over_group(wg, x, maximum<Float>());

    std::uint32_t res_index =
        sycl::reduce_over_group(wg, res_max == x ? x_index : int_max, minimum<std::uint32_t>());

    if (local_id == 0) {
        result.value = res_max;
        result.index = res_index;
    }

    item.barrier(sycl::access::fence_space::local_space);
}

template <typename Float>
sycl::event solve_smo(sycl::queue& q,
                      const pr::ndview<Float, 2>& kernel_values,
                      const pr::ndview<std::uint32_t, 1>& ws_indices,
                      const pr::ndarray<Float, 1>& labels,
                      const pr::ndview<Float, 1>& grad,
                      const std::int64_t row_count,
                      const std::int64_t ws_count,
                      const std::int64_t max_inner_iter,
                      const Float C,
                      const Float eps,
                      const Float tau,
                      pr::ndview<Float, 1>& alpha,
                      pr::ndview<Float, 1>& delta_alpha,
                      pr::ndview<Float, 1>& grad_diff,
                      pr::ndview<std::uint32_t, 1>& inner_iter_count,
                      const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(solve_smo, q);
    ONEDAL_ASSERT(row_count > 0);
    ONEDAL_ASSERT(row_count <= dal::detail::limits<std::uint32_t>::max());
    ONEDAL_ASSERT(row_count == labels.get_dimension(0));
    ONEDAL_ASSERT(max_inner_iter > 0);
    ONEDAL_ASSERT(max_inner_iter <= dal::detail::limits<std::uint32_t>::max());
    ONEDAL_ASSERT(ws_count > 0);
    ONEDAL_ASSERT(ws_count <= dal::detail::limits<std::uint32_t>::max());
    ONEDAL_ASSERT(ws_indices.get_dimension(0) == ws_count);
    ONEDAL_ASSERT(delta_alpha.get_dimension(0) == ws_count);
    ONEDAL_ASSERT(labels.get_dimension(0) == grad.get_dimension(0));
    ONEDAL_ASSERT(alpha.get_dimension(0) == grad.get_dimension(0));

    const Float fp_min = dal::detail::limits<Float>::min();

    const Float* labels_ptr = labels.get_data();
    const Float* kernel_values_ptr = kernel_values.get_data();
    const std::uint32_t* ws_indices_ptr = ws_indices.get_data();
    const Float* grad_ptr = grad.get_data();
    Float* alpha_ptr = alpha.get_mutable_data();
    Float* delta_alpha_ptr = delta_alpha.get_mutable_data();
    Float* grad_diff_ptr = grad_diff.get_mutable_data();
    std::uint32_t* inner_iter_count_ptr = inner_iter_count.get_mutable_data();

    const sycl::nd_range<1> nd_range = dal::backend::make_multiple_nd_range_1d(ws_count, ws_count);

    auto solve_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        local_accessor_rw_t<Float> local_kernel_values(ws_count, cgh);
        local_accessor_rw_t<Float> objective_func(ws_count, cgh);
        local_accessor_rw_t<local_variables<Float>> local_vars(1, cgh);

        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            const std::uint32_t i = item.get_local_id(0);

            const std::uint32_t ws_index = ws_indices_ptr[i];

            Float grad_i = grad_ptr[ws_index];
            Float alpha_i = alpha_ptr[ws_index];
            const Float old_alpha_i = alpha_i;
            const Float labels_i = labels_ptr[ws_index];

            std::uint32_t b_i = 0;
            std::uint32_t b_j = 0;

            Float* local_kernel_values_ptr = local_kernel_values.get_pointer().get();
            Float* objective_func_ptr = objective_func.get_pointer().get();
            local_variables<Float>* local_vars_ptr = local_vars.get_pointer().get();

            local_kernel_values_ptr[i] = kernel_values_ptr[i * row_count + ws_index];
            item.barrier(sycl::access::fence_space::local_space);

            std::uint32_t inner_iter = 0;
            for (; inner_iter < max_inner_iter; ++inner_iter) {
                /* m(alpha) = min(grad[i]): i belongs to I_UP (alpha) */
                objective_func_ptr[i] =
                    is_upper_edge<Float>(labels_i, alpha_i, C) ? -grad_i : fp_min;

                /* Find i index of the working set (b_i) */
                reduce_arg_max(item,
                               objective_func_ptr,
                               local_vars_ptr[0].max_val_ind);
                b_i = local_vars_ptr[0].max_val_ind.index;
                const Float ma = -local_vars_ptr[0].max_val_ind.value;

                /* max_f(alpha) = max(grad[i]): i belongs to i_low (alpha)  */
                objective_func_ptr[i] =
                    is_lower_edge<Float>(labels_i, alpha_i, C) ? grad_i : fp_min;

                /* Find max gradient */
                reduce_arg_max(item,
                               objective_func_ptr,
                               local_vars_ptr[0].max_val_ind);

                if (i == 0) {
                    const Float max_f = local_vars_ptr[0].max_val_ind.value;

                    /* for condition check: m(alpha) >= max_f */
                    local_vars_ptr[0].local_diff = max_f - ma;
                    if (inner_iter == 0) {
                        local_vars_ptr[0].local_eps =
                            sycl::fmax(eps, local_vars_ptr[0].local_diff * Float(1e-1));
                        grad_diff_ptr[0] = local_vars_ptr[0].local_diff;
                    }
                }

                item.barrier(sycl::access::fence_space::local_space);
                if (local_vars_ptr[0].local_diff < local_vars_ptr[0].local_eps) {
                    break;
                }

                const Float kii = local_kernel_values_ptr[i];
                const Float k_bi_bi = local_kernel_values_ptr[b_i];
                const Float ki_bi = kernel_values_ptr[b_i * row_count + ws_index];

                if (is_lower_edge<Float>(labels_i, alpha_i, C) && ma < grad_i) {
                    /* M(alpha) = max((b^2/a) : i belongs to I_low(alpha) and ma < grad(alpha) */
                    const Float b = ma - grad_i;
                    const Float a = sycl::fmax(kii + k_bi_bi - Float(2.0) * ki_bi, tau);
                    const Float dt = b / a;

                    objective_func_ptr[i] = b * dt;
                }
                else {
                    objective_func_ptr[i] = fp_min;
                }

                /* Find j index of the working set (b_j) */
                reduce_arg_max(item,
                               objective_func_ptr,
                               local_vars_ptr[0].max_val_ind);
                b_j = local_vars_ptr[0].max_val_ind.index;

                const Float ki_bj = kernel_values_ptr[b_j * row_count + ws_index];

                /* Update alpha */
                if (i == b_i) {
                    local_vars_ptr[0].delta_b_i = labels_i > 0 ? C - alpha_i : alpha_i;
                }
                if (i == b_j) {
                    local_vars_ptr[0].delta_b_j = labels_i > 0 ? alpha_i : C - alpha_i;
                    const Float b = ma - grad_i;
                    const Float a = sycl::fmax(kii + k_bi_bi - Float(2.0) * ki_bi, tau);

                    const Float dt = -b / a;
                    local_vars_ptr[0].delta_b_j = sycl::fmin(local_vars_ptr[0].delta_b_j, dt);
                }

                item.barrier(sycl::access::fence_space::local_space);

                const Float delta =
                    sycl::fmin(local_vars_ptr[0].delta_b_i, local_vars_ptr[0].delta_b_j);
                alpha_i += i == b_i ? labels_i * delta : 0;
                alpha_i -= i == b_j ? labels_i * delta : 0;

                /* Update grad */
                grad_i = grad_i + delta * (ki_bi - ki_bj);
            }
            alpha_ptr[ws_index] = alpha_i;
            delta_alpha_ptr[i] = (alpha_i - old_alpha_i) * labels_i;
            if (i == 0) {
                inner_iter_count_ptr[0] = inner_iter;
            }
        });
    });
    return solve_event;
}

#endif

} // namespace oneapi::dal::svm::backend
