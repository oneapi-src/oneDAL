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
template <typename Task = task::by_default>
class infer_input_impl;

template <typename Task = task::by_default>
class infer_result_impl;
} // namespace detail

template <typename Task = task::by_default>
class infer_input : public base {
public:
    using pimpl = dal::detail::pimpl<detail::infer_input_impl<Task>>;
    infer_input(const model<Task>& trained_model,
                const table& data,
                std::uint64_t results_to_compute);
    infer_input(
        const model<Task>& trained_model,
        const table& data,
        infer_result_to_compute results_to_compute = infer_result_to_compute::compute_class_labels);

    model<Task> get_model() const;
    table get_data() const;
    std::uint64_t get_results_to_compute() const;

    auto& set_model(const model<Task>& value) {
        set_model_impl(value);
        return *this;
    }
    auto& set_data(const table& value) {
        set_data_impl(value);
        return *this;
    }

    auto& set_results_to_compute(infer_result_to_compute value) {
        set_results_to_compute_impl(static_cast<std::uint64_t>(value));
        return *this;
    }
    auto& set_results_to_compute(std::uint64_t value) {
        set_results_to_compute_impl(value);
        return *this;
    }

private:
    void set_model_impl(const model<Task>& value);
    void set_data_impl(const table& value);
    void set_results_to_compute_impl(std::uint64_t value);

    pimpl impl_;
};

template <typename Task = task::by_default>
class infer_result : public base {
public:
    using pimpl = dal::detail::pimpl<detail::infer_result_impl<Task>>;
    template <typename T>
    using is_classification_t =
        std::enable_if_t<std::is_same_v<T, std::decay_t<task::classification>>>;
    infer_result();

    table get_labels() const;

    auto& set_labels(const table& value) {
        set_labels_impl(value);
        return *this;
    }

    /* classification specific methods */
    template <typename T = Task, typename = is_classification_t<T>>
    table get_probabilities() const {
        return get_probabilities_impl();
    }

    template <typename T = Task, typename = is_classification_t<T>>
    auto& set_probabilities(const table& value) {
        set_probabilities_impl(value);
        return *this;
    }

private:
    table get_probabilities_impl() const;
    void set_labels_impl(const table& value);
    void set_probabilities_impl(const table& value);

    pimpl impl_;
};

} // namespace oneapi::dal::decision_forest
