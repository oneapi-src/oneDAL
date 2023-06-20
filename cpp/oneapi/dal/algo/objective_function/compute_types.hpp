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
///              be :expr:`task::compute`.
template <typename Task = task::by_default>
class compute_input : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the given :literal:`data`,
    /// :literal:`parameters` and :literal:`responses` property valuea
    compute_input(const table& data, const table& parameters, const table& responses);

    /// An $n \\times p$ table with the training data, where each row stores one
    /// feature vector.
    /// @remark default = table{}
    const table& get_data() const;

    auto& set_data(const table& value) {
        set_data_impl(value);
        return *this;
    }

    /// An $p+1 \\times 1$ table with the model weights.
    /// @remark default = table{}
    const table& get_parameters() const;

    auto& set_parameters(const table& value) {
        set_parameters_impl(value);
        return *this;
    }

    /// An $n \\times 1$ table with the correct class labels.
    /// @remark default = table{}
    const table& get_responses() const;

    auto& set_responses(const table& value) {
        set_responses_impl(value);
        return *this;
    }

protected:
    void set_data_impl(const table& value);
    void set_parameters_impl(const table& value);
    void set_responses_impl(const table& value);

private:
    dal::detail::pimpl<detail::compute_input_impl<Task>> impl_;
};

/// @tparam Task Tag-type that specifies the type of the problem to solve. Can
///              be :expr:`task::compute`.
template <typename Task = task::by_default>
class compute_result : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    compute_result();

    /// The matrix of size $1 \\times 1$ with
    /// the objective function value.
    /// @remark default = table{}
    const table& get_value() const;

    auto& set_value(const table& value) {
        set_value_impl(value);
        return *this;
    }

    /// The matrix of size $p + 1 \\times 1$ with
    /// the objective function gradient.
    /// @remark default = table{}
    const table& get_gradient() const;

    auto& set_gradient(const table& value) {
        set_gradient_impl(value);
        return *this;
    }

    /// The matrix of size $p + 1 \\times p + 1$ with
    /// the objective function hessian.
    /// @remark default = table{}
    const table& get_hessian() const;

    auto& set_hessian(const table& value) {
        set_hessian_impl(value);
        return *this;
    }

    // TODO add packed_gradient and packed_hessian options

    /// Result options that indicates availability of the properties
    /// @remark default = default_result_options<Task>
    const result_option_id& get_result_options() const;

    auto& set_result_options(const result_option_id& value) {
        set_result_options_impl(value);
        return *this;
    }

protected:
    void set_value_impl(const table&);
    void set_gradient_impl(const table&);
    void set_hessian_impl(const table&);
    // TODO add packed_gradient and packed_hessian options
    void set_result_options_impl(const result_option_id&);

private:
    dal::detail::pimpl<detail::compute_result_impl<Task>> impl_;
};

} // namespace v1

using v1::compute_input;
using v1::compute_result;

} // namespace oneapi::dal::objective_function
