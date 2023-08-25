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

#include "oneapi/dal/algo/covariance/detail/partial_compute_ops.hpp"
#include "oneapi/dal/algo/covariance/backend/cpu/partial_compute_kernel.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::covariance::detail {
namespace v1 {

template <typename Policy, typename Float, typename Method, typename Task>
struct partial_compute_ops_dispatcher<Policy, Float, Method, Task> {
    partial_compute_result<Task> operator()(const Policy& policy,
                                            const descriptor_base<Task>& desc,
                                            const partial_compute_input<Task>& input) const {
        using kernel_dispatcher_t = dal::backend::kernel_dispatcher< //
            KERNEL_SINGLE_NODE_CPU(backend::partial_compute_kernel_cpu<Float, Method, Task>)>;
        return kernel_dispatcher_t()(policy, desc, input);
    }
};

#define INSTANTIATE(F, M, T) \
    template struct ONEDAL_EXPORT partial_compute_ops_dispatcher<dal::detail::host_policy, F, M, T>;

INSTANTIATE(float, method::dense, task::compute)
INSTANTIATE(double, method::dense, task::compute)

} // namespace v1
} // namespace oneapi::dal::covariance::detail
