/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright contributors to the oneDAL project
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
#include "oneapi/dal/detail/parameters/system_parameters.hpp"

namespace oneapi::dal::decision_forest {

namespace detail {
namespace v1 {
template <typename Task>
class infer_input_impl;

template <typename Task>
class infer_result_impl;

template <typename Task>
struct infer_parameters_impl;

template <typename Task = task::by_default>
class infer_parameters : public dal::detail::system_parameters {
public:
    explicit infer_parameters();
    infer_parameters(infer_parameters&&) = default;
    infer_parameters(const infer_parameters&) = default;

    std::int64_t get_block_size() const;
    auto& set_block_size(std::int64_t val) {
        set_block_size_impl(val);
        return *this;
    }

    std::int64_t get_min_trees_for_threading() const;
    auto& set_min_trees_for_threading(std::int64_t val) {
        set_min_trees_for_threading_impl(val);
        return *this;
    }

    std::int64_t get_min_number_of_rows_for_vect_seq_compute() const;
    auto& set_min_number_of_rows_for_vect_seq_compute(std::int64_t val) {
        set_min_number_of_rows_for_vect_seq_compute_impl(val);
        return *this;
    }

    double get_scale_factor_for_vect_parallel_compute() const;
    auto& set_scale_factor_for_vect_parallel_compute(double val) {
        set_scale_factor_for_vect_parallel_compute_impl(val);
        return *this;
    }

private:
    void set_block_size_impl(std::int64_t val);
    void set_min_trees_for_threading_impl(std::int64_t val);
    void set_min_number_of_rows_for_vect_seq_compute_impl(std::int64_t val);
    void set_scale_factor_for_vect_parallel_compute_impl(double val);
    dal::detail::pimpl<infer_parameters_impl<Task>> impl_;
};

} // namespace v1

using v1::infer_input_impl;
using v1::infer_result_impl;
using v1::infer_parameters;

} // namespace detail

namespace v1 {

/// @tparam Task   Tag-type that specifies the type of the problem to solve. Can
///                be :expr:`task::classification` or :expr:`task::regression`.
template <typename Task = task::by_default>
class infer_input : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the given :literal:`model`
    /// and :literal:`data` property values
    infer_input(const model<Task>& trained_model, const table& data);

    /// The trained Decision Forest model
    /// @remark default = model<Task>{}
    const model<Task>& get_model() const;

    auto& set_model(const model<Task>& value) {
        set_model_impl(value);
        return *this;
    }

    /// The dataset for inference $X'$
    /// @remark default = table{}
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

/// @tparam Task   Tag-type that specifies the type of the problem to solve. Can
///                be :expr:`task::classification` or :expr:`task::regression`.
template <typename Task = task::by_default>
class infer_result : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    infer_result();

    /// The $n \\times 1$ table with the predicted labels
    /// @remark default = table{}
    [[deprecated]] const table& get_labels() const {
        return get_responses();
    }
    [[deprecated]] auto& set_labels(const table& value) {
        return set_responses(value);
    }

    /// The $n \\times 1$ table with the predicted responses
    /// @remark default = table{}
    const table& get_responses() const;

    auto& set_responses(const table& value) {
        set_responses_impl(value);
        return *this;
    }

    /// A $n \\times c$ table with the predicted class probabilities for each observation
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
    void set_responses_impl(const table& value);
    const table& get_probabilities_impl() const;
    void set_probabilities_impl(const table& value);

    dal::detail::pimpl<detail::infer_result_impl<Task>> impl_;
};

} // namespace v1

using v1::infer_input;
using v1::infer_result;

} // namespace oneapi::dal::decision_forest
