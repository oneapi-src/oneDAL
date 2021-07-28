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

#include "oneapi/dal/algo/decision_forest/backend/gpu/infer_kernel.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/infer_kernel_impl.hpp"

namespace oneapi::dal::decision_forest::backend {

using dal::backend::context_gpu;
using model_t = model<task::regression>;
using input_t = infer_input<task::regression>;
using result_t = infer_result<task::regression>;
using descriptor_t = detail::descriptor_base<task::regression>;

template <typename Float>
static result_t call_daal_kernel(const context_gpu& ctx,
                                 const descriptor_t& desc,
                                 const model_t& trained_model,
                                 const table& data) {
    auto& queue = ctx.get_queue();

    infer_kernel_impl<Float, std::int32_t, task::regression> infer_impl(queue);
    return infer_impl(desc, trained_model, data);
}

template <typename Float>
static result_t infer(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_model(), input.get_data());
}

template <typename Float>
struct infer_kernel_gpu<Float, method::by_default, task::regression> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return infer<Float>(ctx, desc, input);
    }
};

template struct infer_kernel_gpu<float, method::by_default, task::regression>;
template struct infer_kernel_gpu<double, method::by_default, task::regression>;

} // namespace oneapi::dal::decision_forest::backend
