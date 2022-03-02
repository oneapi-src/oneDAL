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

#include "oneapi/dal/algo/knn/backend/model_conversion.hpp"
#include "oneapi/dal/algo/knn/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/algo/knn/backend/distance_impl.hpp"
#include "oneapi/dal/algo/knn/backend/model_impl.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::knn::backend {

using dal::backend::context_gpu;

template <typename Task>
using descriptor_t = detail::descriptor_base<Task>;

template <typename Float, typename Task>
static train_result<Task> call_kernel(const context_gpu& ctx,
                                      const descriptor_t<Task>& desc,
                                      const table& data,
                                      const table& responses) {
    auto distance_impl = detail::get_distance_impl(desc);
    if (!distance_impl) {
        throw internal_error{ dal::detail::error_messages::unknown_distance_type() };
    }

    const auto model_impl = std::make_shared<brute_force_model_impl<Task>>(data, responses);
    return train_result<Task>().set_model(dal::detail::make_private<model<Task>>(model_impl));
}

template <typename Float, typename Task>
static train_result<Task> train(const context_gpu& ctx,
                                const descriptor_t<Task>& desc,
                                const train_input<Task>& input) {
    return call_kernel<Float, Task>(ctx, desc, input.get_data(), input.get_responses());
}

template <typename Float, typename Task>
struct train_kernel_gpu<Float, method::brute_force, Task> {
    train_result<Task> operator()(const context_gpu& ctx,
                                  const descriptor_t<Task>& desc,
                                  const train_input<Task>& input) const {
        return train<Float, Task>(ctx, desc, input);
    }
};

template struct train_kernel_gpu<float, method::brute_force, task::classification>;
template struct train_kernel_gpu<double, method::brute_force, task::classification>;
template struct train_kernel_gpu<float, method::brute_force, task::regression>;
template struct train_kernel_gpu<double, method::brute_force, task::regression>;
template struct train_kernel_gpu<float, method::brute_force, task::search>;
template struct train_kernel_gpu<double, method::brute_force, task::search>;

} // namespace oneapi::dal::knn::backend
