/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/algo/svm/detail/train_ops.hpp"
#include "oneapi/dal/algo/svm/backend/cpu/train_kernel.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::svm::detail {

using dal::detail::host_policy;

template <typename Float, typename Method, typename Task>
struct train_ops_dispatcher<host_policy, Float, Method, Task> {
    train_result<Task> operator()(const host_policy& ctx,
                                  const descriptor_base<Task>& desc,
                                  const train_input<Task>& input) const {
        using kernel_dispatcher_t = dal::backend::kernel_dispatcher<KERNEL_SINGLE_NODE_CPU(
            backend::train_kernel_cpu<Float, Method, Task>)>;
        return kernel_dispatcher_t{}(ctx, desc, input);
    }
};

#define INSTANTIATE(F, M, T) \
    template struct ONEDAL_EXPORT train_ops_dispatcher<host_policy, F, M, T>;

INSTANTIATE(float, method::smo, task::classification)
INSTANTIATE(float, method::thunder, task::classification)
INSTANTIATE(float, method::thunder, task::nu_classification)
INSTANTIATE(double, method::smo, task::classification)
INSTANTIATE(double, method::thunder, task::classification)
INSTANTIATE(double, method::thunder, task::nu_classification)

INSTANTIATE(float, method::thunder, task::regression)
INSTANTIATE(float, method::thunder, task::nu_regression)
INSTANTIATE(double, method::thunder, task::regression)
INSTANTIATE(double, method::thunder, task::nu_regression)

} // namespace oneapi::dal::svm::detail
