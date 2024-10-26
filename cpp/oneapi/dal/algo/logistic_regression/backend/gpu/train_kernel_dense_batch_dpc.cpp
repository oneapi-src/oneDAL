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

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/algo/logistic_regression/common.hpp"
#include "oneapi/dal/algo/logistic_regression/train_types.hpp"
#include "oneapi/dal/algo/logistic_regression/backend/model_impl.hpp"
#include "oneapi/dal/algo/logistic_regression/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/backend/primitives/objective_function.hpp"
#include "oneapi/dal/backend/primitives/optimizers.hpp"
#include "oneapi/dal/algo/logistic_regression/backend/optimizer_impl.hpp"
#include "oneapi/dal/algo/logistic_regression/backend/gpu/train_kernel_common.hpp"

namespace oneapi::dal::logistic_regression::backend {

using dal::backend::context_gpu;

namespace be = dal::backend;
namespace pr = be::primitives;

template <typename Float, typename Task>
static train_result<Task> train(const context_gpu& ctx,
                                const detail::descriptor_base<Task>& desc,
                                const detail::train_parameters<Task>& params,
                                const train_input<Task>& input) {
    // TODO: add check if the dataset can be moved to gpu
    // Move data to gpu
    const auto sample_count = input.get_data().get_row_count();
    const auto feature_count = input.get_data().get_column_count();
    auto queue = ctx.get_queue();
    pr::ndarray<Float, 2> data_nd =
        pr::table2ndarray<Float>(queue, input.get_data(), sycl::usm::alloc::device);
    table data_gpu = homogen_table::wrap(data_nd.flatten(queue, {}), sample_count, feature_count);
    return call_dal_kernel<Float, Task>(ctx, desc, params, data_gpu, input.get_responses());
}

template <typename Float, typename Task>
struct train_kernel_gpu<Float, method::dense_batch, Task> {
    train_result<Task> operator()(const context_gpu& ctx,
                                  const detail::descriptor_base<Task>& desc,
                                  const detail::train_parameters<Task>& params,
                                  const train_input<Task>& input) const {
        return train<Float, Task>(ctx, desc, params, input);
    }
};

template struct train_kernel_gpu<float, method::dense_batch, task::classification>;
template struct train_kernel_gpu<double, method::dense_batch, task::classification>;

} // namespace oneapi::dal::logistic_regression::backend
