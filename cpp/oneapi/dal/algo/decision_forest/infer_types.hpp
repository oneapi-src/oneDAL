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

#include "oneapi/dal/algo/decision_forest/common.hpp"

namespace oneapi::dal::decision_forest {

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

template <typename Task = task::by_default>
class infer_input : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    infer_input(const model<Task>& trained_model, const table& data);

    const model<Task>& get_model() const;

    auto& set_model(const model<Task>& value) {
        set_model_impl(value);
        return *this;
    }

    const table& get_data() const;

    auto& set_data(const table& value) {
        set_data_impl(value);
        return *this;
    }

private:
    void set_model_impl(const model<Task>& value);
    void set_data_impl(const table& value);

    dal::detail::pimpl<detail::infer_input_impl<Task>> impl_;
};

template <typename Task = task::by_default>
class infer_result : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    infer_result();

    const table& get_labels() const;

    auto& set_labels(const table& value) {
        set_labels_impl(value);
        return *this;
    }

    /* classification specific methods */
    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    const table& get_probabilities() const {
        return get_probabilities_impl();
    }

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    auto& set_probabilities(const table& value) {
        set_probabilities_impl(value);
        return *this;
    }

private:
    void set_labels_impl(const table& value);
    const table& get_probabilities_impl() const;
    void set_probabilities_impl(const table& value);

    dal::detail::pimpl<detail::infer_result_impl<Task>> impl_;
};

} // namespace v1

using v1::infer_input;
using v1::infer_result;

} // namespace oneapi::dal::decision_forest
