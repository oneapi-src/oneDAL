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
class infer_input_impl;

template <typename Task>
class infer_result_impl;
} // namespace v1

using v1::infer_input_impl;
using v1::infer_result_impl;

} // namespace detail

namespace v1 {

/// @tparam Task Tag-type that specifies type of the problem to solve. Can
///              be :expr:`task::classification`.
template <typename Task = task::by_default>
class infer_input : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the given :literal:`data`
    /// and :literal:`model` property values
    infer_input(const table& data, const model<Task>& model);

    /// The dataset for inference $X'$
    /// @remark default = table{}
    const table& get_data() const;

    auto& set_data(const table& data) {
        set_data_impl(data);
        return *this;
    }

    /// The trained Logistic Regression model
    /// @remark default = model<Task>{}
    const model<Task>& get_model() const;

    auto& set_model(const model<Task>& m) {
        set_model_impl(m);
        return *this;
    }

protected:
    void set_data_impl(const table& data);
    void set_model_impl(const model<Task>& model);

private:
    dal::detail::pimpl<detail::infer_input_impl<Task>> impl_;
};

/// @tparam Task Tag-type that specifies type of the problem to solve. Can
///              be :expr:`task::classification`.
template <typename Task = task::by_default>
class infer_result {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    infer_result();

    /// The predicted responses
    /// @remark default = table{}
    const table& get_responses() const;

    /// The predicted probabilities
    /// @remark default = table{}
    const table& get_probabilities() const;

    auto& set_responses(const table& value) {
        set_responses_impl(value);
        return *this;
    }

    auto& set_probabilities(const table& value) {
        set_probabilities_impl(value);
        return *this;
    }

protected:
    void set_responses_impl(const table&);
    void set_probabilities_impl(const table&);

private:
    dal::detail::pimpl<detail::infer_result_impl<Task>> impl_;
};

} // namespace v1

using v1::infer_input;
using v1::infer_result;

} // namespace oneapi::dal::logistic_regression
