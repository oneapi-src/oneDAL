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

#include "oneapi/dal/algo/objective_function/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::objective_function::detail {

result_option_id get_value_id() {
    return result_option_id{ result_option_id::make_by_index(0) };
}

result_option_id get_gradient_id() {
    return result_option_id{ result_option_id::make_by_index(1) };
}

result_option_id get_hessian_id() {
    return result_option_id{ result_option_id::make_by_index(2) };
}

result_option_id get_packed_gradient_id() {
    return result_option_id{ result_option_id::make_by_index(3) };
}

result_option_id get_packed_hessian_id() {
    return result_option_id{ result_option_id::make_by_index(4) };
}

template <typename Task>
result_option_id get_default_result_options() {
    return result_option_id{};
}

template <>
result_option_id get_default_result_options<task::logloss>() {
    return get_packed_hessian_id();
}

// namespace oneapi::dal::objective_function::detail
