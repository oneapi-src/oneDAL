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
class train_input_impl;

template <typename Task>
class train_result_impl;
} // namespace v1

using v1::train_input_impl;
using v1::train_result_impl;

} // namespace detail

namespace v1 {

template <typename Task = task::by_default>
class train_input : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    train_input(const table& data, const table& labels, const table& weights = table{});

    const table& get_data() const;

    auto& set_data(const table& value) {
        set_data_impl(value);
        return *this;
    }

    const table& get_labels() const;

    auto& set_labels(const table& value) {
        set_labels_impl(value);
        return *this;
    }

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

template <typename Task = task::by_default>
class train_result : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    train_result();

    std::int64_t get_support_vector_count() const;

    const model<Task>& get_model() const;

    auto& set_model(const model<Task>& value) {
        set_model_impl(value);
        return *this;
    }

    const table& get_support_vectors() const;

    auto& set_support_vectors(const table& value) {
        set_support_vectors_impl(value);
        return *this;
    }

    const table& get_support_indices() const;

    auto& set_support_indices(const table& value) {
        set_support_indices_impl(value);
        return *this;
    }

    const table& get_coeffs() const;

    auto& set_coeffs(const table& value) {
        set_coeffs_impl(value);
        return *this;
    }

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
