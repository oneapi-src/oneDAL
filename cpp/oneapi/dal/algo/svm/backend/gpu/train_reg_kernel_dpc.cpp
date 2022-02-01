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

#include "oneapi/dal/algo/svm/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::svm::backend {

template <typename Float, typename Method>
struct train_kernel_gpu<Float, Method, task::regression> {
    train_result<task::regression> operator()(const dal::backend::context_gpu& ctx,
                                              const detail::descriptor_base<task::regression>& desc,
                                              const train_input<task::regression>& input) const {
        throw unimplemented(
            dal::detail::error_messages::svm_regression_task_is_not_implemented_for_gpu());
    }
};

template <typename Float, typename Method>
struct train_kernel_gpu<Float, Method, task::nu_regression> {
    train_result<task::nu_regression> operator()(
        const dal::backend::context_gpu& ctx,
        const detail::descriptor_base<task::nu_regression>& desc,
        const train_input<task::nu_regression>& input) const {
        throw unimplemented(
            dal::detail::error_messages::svm_nu_regression_task_is_not_implemented_for_gpu());
    }
};

template struct train_kernel_gpu<float, method::thunder, task::regression>;
template struct train_kernel_gpu<double, method::thunder, task::regression>;
template struct train_kernel_gpu<float, method::thunder, task::nu_regression>;
template struct train_kernel_gpu<double, method::thunder, task::nu_regression>;

} // namespace oneapi::dal::svm::backend
