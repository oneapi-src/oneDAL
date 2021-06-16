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

#include "oneapi/dal/algo/knn/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::knn::backend {

using dal::backend::context_gpu;
using descriptor_t = detail::descriptor_base<task::search>;

template <typename Float>
static train_result<task::search> call_daal_kernel(const context_gpu& ctx,
                                                   const descriptor_t& desc,
                                                   const table& data,
                                                   const table& labels) {
    throw unimplemented(dal::detail::error_messages::knn_search_task_is_not_implemented_for_gpu());
}

template <typename Float>
static train_result<task::search> train(const context_gpu& ctx,
                                        const descriptor_t& desc,
                                        const train_input<task::search>& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_data(), input.get_labels());
}

template <typename Float>
struct train_kernel_gpu<Float, method::brute_force, task::search> {
    train_result<task::search> operator()(const context_gpu& ctx,
                                          const descriptor_t& desc,
                                          const train_input<task::search>& input) const {
        return train<Float>(ctx, desc, input);
    }
};

template struct train_kernel_gpu<float, method::brute_force, task::search>;
template struct train_kernel_gpu<double, method::brute_force, task::search>;

} // namespace oneapi::dal::knn::backend
