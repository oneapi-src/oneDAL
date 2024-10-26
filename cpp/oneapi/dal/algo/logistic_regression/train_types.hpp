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

#include "oneapi/dal/algo/logistic_regression/common.hpp"

namespace oneapi::dal::logistic_regression {

namespace detail {
namespace v1 {
template <typename Task>
class train_input_impl;

template <typename Task>
class train_result_impl;

template <typename Task>
struct train_parameters_impl;

template <typename Task = task::by_default>
class train_parameters : public base {
public:
    explicit train_parameters();
    train_parameters(train_parameters&&) = default;
    train_parameters(const train_parameters&) = default;

    std::int64_t get_cpu_macro_block() const;
    auto& set_cpu_macro_block(std::int64_t val) {
        set_cpu_macro_block_impl(val);
        return *this;
    }

    std::int64_t get_gpu_macro_block() const;
    auto& set_gpu_macro_block(std::int64_t val) {
        set_gpu_macro_block_impl(val);
        return *this;
    }

private:
    void set_cpu_macro_block_impl(std::int64_t val);
    void set_gpu_macro_block_impl(std::int64_t val);
    dal::detail::pimpl<train_parameters_impl<Task>> impl_;
};

} // namespace v1

using v1::train_parameters;
using v1::train_input_impl;
using v1::train_result_impl;

} // namespace detail

namespace v1 {

/// @tparam Task Tag-type that specifies type of the problem to solve. Can
///              be :expr:`task::classification`.
template <typename Task = task::by_default>
class train_input : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the given :literal:`data`
    /// and :literal:`responses` property values
    train_input(const table& data, const table& responses);

    /// The training set X
    /// @remark default = table{}
    const table& get_data() const;

    auto& set_data(const table& data) {
        set_data_impl(data);
        return *this;
    }

    /// Vector of responses y for the training set X
    /// @remark default = table{}
    const table& get_responses() const;

    auto& set_responses(const table& responses) {
        set_data_impl(responses);
        return *this;
    }

protected:
    void set_data_impl(const table& data);
    void set_responses_impl(const table& responses);

private:
    dal::detail::pimpl<detail::train_input_impl<Task>> impl_;
};

/// @tparam Task Tag-type that specifies type of the problem to solve. Can
///              be :expr:`task::classification`.
template <typename Task = task::by_default>
class train_result {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    train_result();

    /// The trained Logistic Regression model
    /// @remark default = model<Task>{}
    const model<Task>& get_model() const;

    auto& set_model(const model<Task>& value) {
        set_model_impl(value);
        return *this;
    }

    /// Table with Logistic Regression intercept
    const table& get_intercept() const;

    auto& set_intercept(const table& value) {
        set_intercept_impl(value);
        return *this;
    }

    /// Table of Logistic Regression coefficients
    const table& get_coefficients() const;

    auto& set_coefficients(const table& value) {
        set_coefficients_impl(value);
        return *this;
    }

    /// Actual number of optimizer iterations
    std::int64_t get_iterations_count() const;

    auto& set_iterations_count(std::int64_t value) {
        set_iterations_count_impl(value);
        return *this;
    }

    /// Number of optimizer subiterations
    std::int64_t get_inner_iterations_count() const;

    auto& set_inner_iterations_count(std::int64_t value) {
        set_inner_iterations_count_impl(value);
        return *this;
    }

    /// Table of Logistic Regression coefficients and intercept
    const table& get_packed_coefficients() const;

    auto& set_packed_coefficients(const table& value) {
        set_packed_coefficients_impl(value);
        return *this;
    }

    /// Result options that indicates availability of the properties
    const result_option_id& get_result_options() const;

    auto& set_result_options(const result_option_id& value) {
        set_result_options_impl(value);
        return *this;
    }

protected:
    void set_model_impl(const model<Task>&);

    void set_intercept_impl(const table&);
    void set_coefficients_impl(const table&);
    void set_packed_coefficients_impl(const table&);
    void set_iterations_count_impl(std::int64_t);
    void set_inner_iterations_count_impl(std::int64_t);

    void set_result_options_impl(const result_option_id&);

private:
    dal::detail::pimpl<detail::train_result_impl<Task>> impl_;
};

} // namespace v1

using v1::train_input;
using v1::train_result;

} // namespace oneapi::dal::logistic_regression
