/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
class train_input_impl;

template <typename Task>
class train_result_impl;
} // namespace v1

using v1::train_input_impl;
using v1::train_result_impl;

} // namespace detail

namespace v1 {

/// @tparam Task   Tag-type that specifies the type of the problem to solve. Can
///                be :expr:`task::v1::classification`.
template <typename Task = task::by_default>
class train_input : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the given :literal:`data`,
    /// :literal:`labels` and :literal:`weights`
    train_input(const table& data, const table& labels, const table& weights = table{});

    /// The training set $X$
    /// @remark default = table{}
    const table& get_data() const;

    auto& set_data(const table& value) {
        set_data_impl(value);
        return *this;
    }

    /// The vector of labels $y$ for the training set $X$
    /// @remark default = table{}
    const table& get_labels() const;

    auto& set_labels(const table& value) {
        set_labels_impl(value);
        return *this;
    }

    /// The vector of weights $w$ for the training set $X$
    /// @remark default = table{}
    const table& get_weights() const;

    auto& set_weights(const table& value) {
        set_weights_impl(value);
        return *this;
    }

protected:
    void set_data_impl(const table& value);
    void set_labels_impl(const table& value);
    void set_weights_impl(const table& value);

private:
    dal::detail::pimpl<detail::train_input_impl<Task>> impl_;
};

/// @tparam Task Tag-type that specifies the type of the problem to solve. Can
///              be :expr:`task::v1::classification`.
template <typename Task = task::by_default>
class train_result : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    train_result();

    /// The number of support vectors
    /// @remark default = 0
    std::int64_t get_support_vector_count() const;

    /// The trained SVM model
    /// @remark default = model<Task>{}
    const model<Task>& get_model() const;

    auto& set_model(const model<Task>& value) {
        set_model_impl(value);
        return *this;
    }

    /// A $nsv \\times p$ table containing support vectors,
    /// where $nsv$ is the number of support vectors.
    /// @remark default = table{}
    const table& get_support_vectors() const;

    auto& set_support_vectors(const table& value) {
        set_support_vectors_impl(value);
        return *this;
    }

    /// A $nsv \\times 1$ table containing support indices
    /// @remark default = table{}
    const table& get_support_indices() const;

    auto& set_support_indices(const table& value) {
        set_support_indices_impl(value);
        return *this;
    }

    /// A $nsv \\times 1$ table containing coefficients of Lagrange multiplier
    /// @remark default = table{}
    const table& get_coeffs() const;

    auto& set_coeffs(const table& value) {
        set_coeffs_impl(value);
        return *this;
    }

    /// The bias
    /// @remark default = 0.0
    double get_bias() const;

    auto& set_bias(double value) {
        set_bias_impl(value);
        return *this;
    }

protected:
    void set_model_impl(const model<Task>&);
    void set_support_vectors_impl(const table&);
    void set_support_indices_impl(const table&);
    void set_coeffs_impl(const table&);
    void set_bias_impl(double);

private:
    dal::detail::pimpl<detail::train_result_impl<Task>> impl_;
};

} // namespace v1

using v1::train_input;
using v1::train_result;

} // namespace oneapi::dal::svm
