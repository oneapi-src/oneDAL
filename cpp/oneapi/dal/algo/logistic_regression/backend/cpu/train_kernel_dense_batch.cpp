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

#include "oneapi/dal/algo/logistic_regression/backend/cpu/train_kernel.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::logistic_regression::backend {

using dal::backend::context_cpu;

template <typename Float, typename Task>
struct train_kernel_cpu<Float, method::dense_batch, Task> {
    train_result<Task> operator()(const context_cpu& ctx,
                                  const detail::descriptor_base<Task>& desc,
                                  const detail::train_parameters<Task>& params,
                                  const train_input<Task>& input) const {
        throw unimplemented(
            dal::detail::error_messages::log_reg_dense_batch_method_is_not_implemented_for_cpu());
    }
};

template <typename Float, typename Task>
struct train_kernel_cpu<Float, method::sparse, Task> {
    train_result<Task> operator()(const context_cpu& ctx,
                                  const detail::descriptor_base<Task>& desc,
                                  const detail::train_parameters<Task>& params,
                                  const train_input<Task>& input) const {
        // CHANGE ERROR NAME !!!
        throw unimplemented(
            dal::detail::error_messages::log_reg_dense_batch_method_is_not_implemented_for_cpu());
    }
};

template struct train_kernel_cpu<float, method::dense_batch, task::classification>;
template struct train_kernel_cpu<double, method::dense_batch, task::classification>;

template struct train_kernel_cpu<float, method::sparse, task::classification>;
template struct train_kernel_cpu<double, method::sparse, task::classification>;

} // namespace oneapi::dal::logistic_regression::backend
