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

#include "oneapi/dal/algo/svm/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::svm::backend {

template <typename Float>
struct train_kernel_gpu<Float, method::smo, task::classification> {
    train_result<task::classification> operator()(
        const dal::backend::context_gpu& ctx,
        const detail::descriptor_base<task::classification>& params,
        const train_input<task::classification>& input) const {
        throw unimplemented(
            dal::detail::error_messages::svm_smo_method_is_not_implemented_for_gpu());
    }
};

template <typename Float>
struct train_kernel_gpu<Float, method::smo, task::nu_classification> {
    train_result<task::nu_classification> operator()(
        const dal::backend::context_gpu& ctx,
        const detail::descriptor_base<task::nu_classification>& params,
        const train_input<task::nu_classification>& input) const {
        throw unimplemented(
            dal::detail::error_messages::nu_svm_smo_method_is_not_implemented_for_gpu());
    }
};

template struct train_kernel_gpu<float, method::smo, task::classification>;
template struct train_kernel_gpu<double, method::smo, task::classification>;
template struct train_kernel_gpu<float, method::smo, task::nu_classification>;
template struct train_kernel_gpu<double, method::smo, task::nu_classification>;

} // namespace oneapi::dal::svm::backend
