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

template <typename Float, typename Task, typename Method, typename Kernel>
struct train_ops_dispatcher<default_execution_context, Float, Task, Method> {
    train_result operator()(const default_execution_context& ctx,
                            const descriptor_base& desc,
                            const train_input& input) const {
        using kernel_dispatcher_t =
            dal::backend::kernel_dispatcher<backend::train_kernel_cpu<Float, Task, Method, Kernel>>;
        return kernel_dispatcher_t()(ctx, desc, input);
    }
};

#define INSTANTIATE(F, T, M, K) \
    template struct train_ops_dispatcher<default_execution_context, F, T, M, K>;

#define INSTANTIATE_FOR_KERNEL(K)                                \
    INSTANTIATE(float, task::classification, method::smo, K)     \
    INSTANTIATE(float, task::classification, method::thunder, K) \
    INSTANTIATE(double, task::classification, method::smo, K)    \
    INSTANTIATE(double, task::classification, method::thunder, K)

INSTANTIATE_FOR_KERNEL(kernel_linear::descriptor_base)
INSTANTIATE_FOR_KERNEL(kernel_rbf::descriptor_base)

#define INSTANTIATE_FOR_FLOAT(F)                                                        \
    INSTANTIATE(F, task::classification, method::smo, kernel_linear::descriptor<F>)     \
    INSTANTIATE(F, task::classification, method::thunder, kernel_linear::descriptor<F>) \
    INSTANTIATE(F, task::classification, method::smo, kernel_rbf::descriptor<F>)        \
    INSTANTIATE(F, task::classification, method::thunder, kernel_rbf::descriptor<F>)

INSTANTIATE_FOR_KERNEL(float)
INSTANTIATE_FOR_KERNEL(double)

} // namespace oneapi::dal::svm::detail
