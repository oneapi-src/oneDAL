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

#include "oneapi/dal/backend/interop/common.hpp"

#include "oneapi/dal/algo/logistic_regression/common.hpp"
#include "oneapi/dal/algo/logistic_regression/infer_types.hpp"
#include "oneapi/dal/algo/logistic_regression/backend/cpu/infer_kernel.hpp"

namespace oneapi::dal::logistic_regression::backend {

using dal::backend::context_cpu;

template <typename Float, typename Task>
struct infer_kernel_cpu<Float, method::dense_batch, Task> {
    infer_result<Task> operator()(const context_cpu& ctx,
                                  const detail::descriptor_base<Task>& desc,
                                  const infer_input<Task>& input) const {
        throw std::logic_error("Not implemented");
    }
};

template struct infer_kernel_cpu<float, method::dense_batch, task::binary_classification>;
template struct infer_kernel_cpu<double, method::dense_batch, task::binary_classification>;

} // namespace oneapi::dal::logistic_regression::backend
