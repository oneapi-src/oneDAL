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

#include "oneapi/dal/algo/objective_function/common.hpp"

namespace oneapi::dal::objective_function {

namespace detail {
namespace v1 {
template <typename Task>
class compute_input_impl;

template <typename Task>
class compute_result_impl;

} // namespace v1

using v1::compute_input_impl;
using v1::compute_result_impl;

} // namespace detail

namespace v1 {

/// @tparam Task Tag-type that specifies the type of the problem to solve. Can
///              be :expr:`task::logloss`.
template <typename Task = task::by_default>
class compute_input : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the given :literal:`data`
    /// property value
    compute_input(const table& data, const table& parameters, const table& responses);

    /// An $n \\times p$ table with the training data, where each row stores one
    /// feature vector.
    /// @remark default = table{}
    const table& get_data() const;
    const table& get_parameters() const;
    const table& get_responses() const;

    auto& set_data(const table& value) {
        set_data_impl(value);
        return *this;
    }

    auto& set_parameters(const table& value) {
        set_parameters_impl(value);
        return *this;
    }

    auto& set_responses(const table& value) {
        set_responses_impl(value);
        return *this;
    }

    auto& set_value_placeholder(homogen_table_builder& placeholder) {
        set_value_placeholder_impl(placeholder);
        return *this;
    }
    auto& set_gradient_placeholder(homogen_table_builder& placeholder) {
        set_gradient_placeholder_impl(placeholder);
        return *this;
    }
    auto& set_hessian_placeholder(homogen_table_builder& placeholder) {
        set_hessian_placeholder_impl(placeholder);
        return *this;
    }
    auto& set_packed_gradient_placeholder(homogen_table_builder& placeholder) {
        set_packed_gradient_placeholder_impl(placeholder);
        return *this;
    }
    auto& set_packed_hessian_placeholder(homogen_table_builder& placeholder) {
        set_packed_hessian_placeholder_impl(placeholder);
        return *this;
    }

protected:
    void set_data_impl(const table& value);
    void set_parameters_impl(const table& value);
    void set_responses_impl(const table& value);

    void set_value_placeholder_impl(homogen_table_builder& placeholder);
    void set_gradient_placeholder_impl(homogen_table_builder& placeholder);
    void set_hessian_placeholder_impl(homogen_table_builder& placeholder);
    void set_packed_gradient_placeholder_impl(homogen_table_builder& placeholder);
    void set_packed_hessian_placeholder_impl(homogen_table_builder& placeholder);


private:
    dal::detail::pimpl<detail::compute_input_impl<Task>> impl_;
};

/// @tparam Task Tag-type that specifies the type of the problem to solve. Can
///              be :expr:`task::logloss`.
template <typename Task = task::by_default>
class compute_result : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    compute_result();


    const table& get_value() const;
    const table& get_gradient() const;
    const table& get_hessian() const;
    const table& get_packed_gradient() const;
    const table& get_packed_hessian() const;

    
    const result_option_id& get_result_options() const;

    auto& set_result_options(const result_option_id& value) {
        set_result_options_impl(value);
        return *this;
    }

protected:
    void set_value_impl(const table&);
    void set_gradient_impl(const table&);
    void set_hessian_impl(const table&);
    void set_packed_gradient_impl(const table&);
    void set_packed_hessian_impl(const table&);
    void set_result_options_impl(const result_option_id&);

private:
    dal::detail::pimpl<detail::compute_result_impl<Task>> impl_;
};

} // namespace v1

using v1::compute_input;
using v1::compute_result;

} // namespace oneapi::dal::objective_function
