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
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/algo/logistic_regression/common.hpp"
#include "oneapi/dal/algo/logistic_regression/train_types.hpp"
#include "oneapi/dal/algo/logistic_regression/backend/model_impl.hpp"
#include "oneapi/dal/algo/logistic_regression/backend/cpu/train_kernel.hpp"

namespace oneapi::dal::logistic_regression::backend {

using dal::backend::context_cpu;

template <typename Float, typename Task>
struct train_kernel_cpu<Float, method::newton_cg, Task> {
    train_result<Task> operator()(const context_cpu& ctx,
                                  const detail::descriptor_base<Task>& desc,
                                  const detail::train_parameters<Task>& params,
                                  const train_input<Task>& input) const {
        throw std::logic_error("Not implemented");
    }
};

template struct train_kernel_cpu<float, method::newton_cg, task::classification>;
template struct train_kernel_cpu<double, method::newton_cg, task::classification>;

} // namespace oneapi::dal::logistic_regression::backend
