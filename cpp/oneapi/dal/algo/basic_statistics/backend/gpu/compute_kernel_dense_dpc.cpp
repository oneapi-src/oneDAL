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

#include "oneapi/dal/algo/basic_statistics/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/algo/basic_statistics/backend/gpu/compute_kernel_dense_impl.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/common.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::basic_statistics::backend {

using method_t = method::dense;
using task_t = task::compute;
using input_t = compute_input<task_t>;
using result_t = compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

template <typename Float>
static result_t compute(const bk::context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) {
    const auto res_op = desc.get_result_options();
    const auto res_min_max = result_options::min | result_options::max;
    const auto res_mean_varc = result_options::mean | result_options::variance;

    if ((res_op.test(res_min_max) && res_op.test(~res_min_max)) ||
        (res_op.test(res_mean_varc) && res_op.test(~res_mean_varc))) {
        return compute_kernel_dense_impl<Float, bs_mode_all>(ctx)(desc, input);
    }
    else if (res_op.test(res_min_max)) {
        return compute_kernel_dense_impl<Float, bs_mode_min_max>(ctx)(desc, input);
    }
    else if (res_op.test(res_mean_varc)) {
        return compute_kernel_dense_impl<Float, bs_mode_mean_variance>(ctx)(desc, input);
    }

    return compute_kernel_dense_impl<Float, bs_mode_all>(ctx)(desc, input);
}

template <typename Float>
struct compute_kernel_gpu<Float, method_t, task_t> {
    result_t operator()(const bk::context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute<Float>(ctx, desc, input);
    }
};

template struct compute_kernel_gpu<float, method_t, task_t>;
template struct compute_kernel_gpu<double, method_t, task_t>;

} // namespace oneapi::dal::basic_statistics::backend
