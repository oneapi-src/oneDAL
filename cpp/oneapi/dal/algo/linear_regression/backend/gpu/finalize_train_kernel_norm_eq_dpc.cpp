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

#include "oneapi/dal/algo/linear_regression/backend/gpu/finalize_train_kernel.hpp"
#include "oneapi/dal/algo/linear_regression/backend/gpu/finalize_train_kernel_norm_eq_impl.hpp"

#include "oneapi/dal/detail/common.hpp"

#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::linear_regression::backend {

namespace be = dal::backend;

template <typename Float, typename Task>
static train_result<Task> finalize_train(const be::context_gpu& ctx,
                                         const detail::descriptor_base<Task>& desc,
                                         const detail::train_parameters<Task>& params,
                                         const partial_train_result<Task>& input) {
    return finalize_train_kernel_norm_eq_impl<Float, Task>(ctx)(desc, params, input);
}

template <typename Float, typename Task>
struct finalize_train_kernel_gpu<Float, method::norm_eq, Task> {
    train_result<Task> operator()(const be::context_gpu& ctx,
                                  const detail::descriptor_base<Task>& desc,
                                  const detail::train_parameters<Task>& params,
                                  const partial_train_result<Task>& input) const {
        return finalize_train<Float, Task>(ctx, desc, params, input);
    }
};

template struct finalize_train_kernel_gpu<float, method::norm_eq, task::regression>;
template struct finalize_train_kernel_gpu<double, method::norm_eq, task::regression>;

} // namespace oneapi::dal::linear_regression::backend
