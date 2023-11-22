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

#include "oneapi/dal/algo/pca/backend/gpu/train_kernel.hpp"

namespace oneapi::dal::pca::backend {

template <typename Float>
struct train_kernel_gpu<Float, method::svd, task::dim_reduction> {
    train_result<task::dim_reduction> operator()(
        const dal::backend::context_gpu& ctx,
        const detail::descriptor_base<task::dim_reduction>& desc,
        const detail::train_parameters<task::dim_reduction>& params,
        const train_input<task::dim_reduction>& input) const {
        throw unimplemented(
            dal::detail::error_messages::pca_svd_based_method_is_not_implemented_for_gpu());
    }
};

template struct train_kernel_gpu<float, method::svd, task::dim_reduction>;
template struct train_kernel_gpu<double, method::svd, task::dim_reduction>;

} // namespace oneapi::dal::pca::backend
