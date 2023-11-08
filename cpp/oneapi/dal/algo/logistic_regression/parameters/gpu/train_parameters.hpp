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

#pragma once

#include "oneapi/dal/algo/logistic_regression/train_types.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::logistic_regression::parameters {

template <typename Float, typename Method, typename Task>
struct ONEDAL_EXPORT train_parameters_gpu {
    using params_t = detail::train_parameters<Task>;
    params_t operator()(const dal::backend::context_gpu& ctx,
                        const detail::descriptor_base<Task>& desc,
                        const train_input<Task>& input) const;
};

} // namespace oneapi::dal::logistic_regression::parameters
