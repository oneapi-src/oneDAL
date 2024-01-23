/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "oneapi/dal/algo/decision_forest/parameters/cpu/infer_parameters.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::decision_forest::parameters {

using dal::backend::context_gpu;

// TODO: Is it correct to use method::by_default here?
template <typename Float, typename Task>
struct infer_parameters_gpu<Float, method::by_default, Task> {
    using params_t = detail::infer_parameters<Task>;
    params_t operator()(const context_gpu& ctx,
                        const detail::descriptor_base<Task>& desc,
                        const infer_input<Task>& input) const {
        return params_t{};
    }
};

template struct ONEDAL_EXPORT infer_parameters_gpu<float, method::dense, task::by_default>;
template struct ONEDAL_EXPORT infer_parameters_gpu<double, method::dense, task::by_default>;

} // namespace oneapi::dal::decision_forest::parameters
