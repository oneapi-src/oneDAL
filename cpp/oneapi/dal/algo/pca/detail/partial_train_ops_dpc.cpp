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

#include "oneapi/dal/algo/pca/detail/partial_train_ops.hpp"
#include "oneapi/dal/algo/pca/backend/cpu/partial_train_kernel.hpp"
#include "oneapi/dal/algo/pca/backend/gpu/partial_train_kernel.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::pca::detail {
namespace v1 {

template <typename Policy, typename Float, typename Method, typename Task>
struct partial_train_ops_dispatcher<Policy, Float, Method, Task> {
    partial_train_result<Task> operator()(const Policy& policy,
                                          const descriptor_base<Task>& desc,
                                          const partial_train_input<Task>& input) const {
        using kernel_dispatcher_t = dal::backend::kernel_dispatcher< //
            KERNEL_SINGLE_NODE_CPU(backend::partial_train_kernel_cpu<Float, Method, Task>),
            KERNEL_SINGLE_NODE_GPU(backend::partial_train_kernel_gpu<Float, Method, Task>)>;
        return kernel_dispatcher_t()(policy, desc, input);
    }
};

#define INSTANTIATE(F, M, T)      \
    template struct ONEDAL_EXPORT \
        partial_train_ops_dispatcher<dal::detail::data_parallel_policy, F, M, T>;

INSTANTIATE(float, method::cov, task::dim_reduction)
INSTANTIATE(double, method::cov, task::dim_reduction)

} // namespace v1
} // namespace oneapi::dal::pca::detail