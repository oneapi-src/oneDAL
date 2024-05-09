/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include "oneapi/dal/algo/logistic_regression/common.hpp"

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/algo/logistic_regression/train_types.hpp"

namespace oneapi::dal::logistic_regression::backend {

using dal::backend::context_gpu;

template <typename Float, typename Task>
train_result<Task> call_dal_kernel(const context_gpu& ctx,
                                   const detail::descriptor_base<Task>& desc,
                                   const detail::train_parameters<Task>& params,
                                   const table& data,
                                   const table& resp);

} // namespace oneapi::dal::logistic_regression::backend
