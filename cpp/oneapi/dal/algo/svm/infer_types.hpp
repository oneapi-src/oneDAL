/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/algo/svm/common.hpp"

namespace oneapi::dal::svm {

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

/// @tparam Task   Tag-type that specifies the type of the problem to solve.
/// Can be
///                :expr:`oneapi::dal::svm::task::classification`,
///                :expr:`oneapi::dal::svm::task::nu_classification`,
///                :expr:`oneapi::dal::svm::task::regression`, or
///                :expr:`oneapi::dal::svm::task::nu_regression`.
template <typename Task = task::by_default>
class infer_input : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the given :literal:`model`
    /// and :literal:`data` property values
    infer_input(const model<Task> &trained_model, const table &data);

    /// The trained SVM model
    /// @remark default = model<Task>{}
    const model<Task> &get_model() const;

    auto &set_model(const model<Task> &value) {
        set_model_impl(value);
        return *this;
    }

    /// The dataset for inference $X'$
    /// @remark default = table{}
    const table &get_data() const;

    auto &set_data(const table &value) {
        set_data_impl(value);
        return *this;
    }

protected:
    void set_model_impl(const model<Task> &value);
    void set_data_impl(const table &value);

private:
    dal::detail::pimpl<detail::infer_input_impl<Task>> impl_;
};

/// @tparam Task   Tag-type that specifies the type of the problem to solve.
/// Can be
///                :expr:`oneapi::dal::svm::task::classification`,
///                :expr:`oneapi::dal::svm::task::nu_classification`,
///                :expr:`oneapi::dal::svm::task::regression`, or
///                :expr:`oneapi::dal::svm::task::nu_regression`.
template <typename Task = task::by_default>
class infer_result : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    infer_result();

    /// The $n \\times 1$ table with the predicted labels
    /// @remark default = table{}
    [[deprecated]] const table &get_labels() const {
        return get_responses();
    }

    [[deprecated]] auto &set_labels(const table &value) {
        return set_responses(value);
    }

    /// The $n \\times 1$ table with the predicted responses
    /// @remark default = table{}
    const table &get_responses() const;

    auto &set_responses(const table &value) {
        set_responses_impl(value);
        return *this;
    }

    // TODO: Enable this function only for classification and nu_classification
    // problems.
    // This will break binary compatibility, so the change should happen in
    // 2022.1.
    //
    /// The $n \\times 1$ table with the predicted class.
    /// Used with :expr:`oneapi::dal::svm::task::classification`
    /// and :expr:`oneapi::dal::svm::task::nu_classification`.
    /// decision function for each observation
    /// @remark default = table{}
    const table &get_decision_function() const;

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    auto &set_decision_function(const table &value) {
        set_decision_function_impl(value);
        return *this;
    }

protected:
    void set_responses_impl(const table &);
    void set_decision_function_impl(const table &);

private:
    dal::detail::pimpl<detail::infer_result_impl<Task>> impl_;
};

} // namespace v1

using v1::infer_input;
using v1::infer_result;

} // namespace oneapi::dal::svm
