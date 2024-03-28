/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright contributors to the oneDAL project
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

#include "oneapi/dal/algo/decision_forest/detail/infer_ops.hpp"
#include "oneapi/dal/algo/decision_forest/parameters/cpu/infer_parameters.hpp"
#include "oneapi/dal/algo/decision_forest/backend/cpu/infer_kernel.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::decision_forest::detail {
namespace v1 {

template <typename Policy, typename Float, typename Task, typename Method>
struct infer_ops_dispatcher<Policy, Float, Task, Method> {
    infer_result<Task> operator()(const Policy& ctx,
                                  const descriptor_base<Task>& desc,
                                  const infer_parameters<Task>& params,
                                  const infer_input<Task>& input) const {
        return implementation(ctx, desc, params, input);
    }

    infer_parameters<Task> select_parameters(const Policy& ctx,
                                             const descriptor_base<Task>& desc,
                                             const infer_input<Task>& input) const {
        using kernel_dispatcher_t = dal::backend::kernel_dispatcher< //
            KERNEL_SINGLE_NODE_CPU(parameters::infer_parameters_cpu<Float, Method, Task>)>;
        return kernel_dispatcher_t{}(ctx, desc, input);
    }

    infer_result<Task> operator()(const Policy& ctx,
                                  const descriptor_base<Task>& desc,
                                  const infer_input<Task>& input) const {
        const auto params = select_parameters(ctx, desc, input);
        return implementation(ctx, desc, params, input);
    }

private:
    inline auto implementation(const Policy& ctx,
                               const descriptor_base<Task>& desc,
                               const infer_parameters<Task>& params,
                               const infer_input<Task>& input) const {
        using kernel_dispatcher_t = dal::backend::kernel_dispatcher< //
            KERNEL_SINGLE_NODE_CPU(backend::infer_kernel_cpu<Float, Method, Task>)>;
        return kernel_dispatcher_t{}(ctx, desc, params, input);
    }
};

#define INSTANTIATE(F, T, M)                                                               \
    template struct ONEDAL_EXPORT infer_ops_dispatcher<dal::detail::host_policy, F, T, M>; \
    template struct ONEDAL_EXPORT infer_ops_dispatcher<dal::detail::spmd_host_policy, F, T, M>;

INSTANTIATE(float, task::classification, method::by_default)
INSTANTIATE(float, task::regression, method::by_default)
INSTANTIATE(double, task::classification, method::by_default)
INSTANTIATE(double, task::regression, method::by_default)

} // namespace v1
} // namespace oneapi::dal::decision_forest::detail
