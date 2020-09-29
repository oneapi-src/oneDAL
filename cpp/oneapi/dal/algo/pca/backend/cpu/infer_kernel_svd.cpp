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

#include "oneapi/dal/algo/pca/backend/cpu/infer_kernel.hpp"
#include "oneapi/dal/algo/pca/backend/cpu/call_daal_pca_transform.hpp"

namespace oneapi::dal::pca::backend {

template <typename Float>
struct infer_kernel_cpu<Float, method::svd, task::dim_reduction> {
    infer_result<task::dim_reduction> operator()(
        const dal::backend::context_cpu& ctx,
        const descriptor_base<task::dim_reduction>& desc,
        const infer_input<task::dim_reduction>& input) const {
        return infer<Float>(ctx, desc, input);
    }
};

template struct infer_kernel_cpu<float, method::svd, task::dim_reduction>;
template struct infer_kernel_cpu<double, method::svd, task::dim_reduction>;

} // namespace oneapi::dal::pca::backend
