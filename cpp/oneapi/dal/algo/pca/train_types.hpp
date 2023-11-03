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

#include "oneapi/dal/algo/pca/common.hpp"

namespace oneapi::dal::pca {

namespace detail {
namespace v1 {
template <typename Task>
class train_input_impl;

template <typename Task>
class train_result_impl;

template <typename Task>
class partial_train_result_impl;
} // namespace v1

using v1::train_input_impl;
using v1::train_result_impl;
using v1::partial_train_result_impl;

} // namespace detail

namespace v1 {

/// @tparam Task Tag-type that specifies type of the problem to solve. Can
///              be :expr:`task::dim_reduction`.
template <typename Task = task::by_default>
class train_input : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    train_input();
    /// Creates a new instance of the class with the given :literal:`data`
    /// property value
    train_input(const table& data);

    /// An $n \\times p$ table with the training data, where each row stores one
    /// feature vector.
    /// @remark default = table{}
    const table& get_data() const;

    auto& set_data(const table& data) {
        set_data_impl(data);
        return *this;
    }

protected:
    void set_data_impl(const table& data);

private:
    dal::detail::pimpl<detail::train_input_impl<Task>> impl_;
};

/// @tparam Task Tag-type that specifies type of the problem to solve. Can
///              be :expr:`task::dim_reduction`.
template <typename Task = task::by_default>
class train_result {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    train_result();

    /// An $r \\times p$ table with the eigenvectors. Each row contains one
    /// eigenvector.
    /// @remark default = table{}
    /// @invariant :expr:`eigenvectors == model.eigenvectors`
    const table& get_eigenvectors() const;

    /// The trained PCA model
    /// @remark default = model<Task>{}
    const model<Task>& get_model() const;

    auto& set_model(const model<Task>& value) {
        set_model_impl(value);
        return *this;
    }

    /// A $1 \\times r$ table that contains the eigenvalues for for the first
    /// :literal:`r` features.
    /// @remark default = table{}
    const table& get_eigenvalues() const;

    auto& set_eigenvalues(const table& value) {
        set_eigenvalues_impl(value);
        return *this;
    }

    /// A $1 \\times r$ table that contains the variances for the first :literal:`r`
    /// features.
    /// @remark default = table{}
    const table& get_variances() const;

    auto& set_variances(const table& value) {
        set_variances_impl(value);
        return *this;
    }

    /// A $1 \\times r$ table that contains the mean values for the first :literal:`r`
    /// features.
    /// @remark default = table{}
    const table& get_means() const;

    auto& set_means(const table& value) {
        set_means_impl(value);
        return *this;
    }
    /// Result options that indicates availability of the properties
    /// @remark default = default_result_options<Task>
    const result_option_id& get_result_options() const;

    auto& set_result_options(const result_option_id& value) {
        set_result_options_impl(value);
        return *this;
    }

protected:
    void set_model_impl(const model<Task>&);
    void set_eigenvalues_impl(const table&);
    void set_eigenvectors_impl(const table&);
    void set_variances_impl(const table&);
    void set_means_impl(const table&);
    void set_result_options_impl(const result_option_id&);

private:
    dal::detail::pimpl<detail::train_result_impl<Task>> impl_;
};

template <typename Task = task::by_default>
class partial_train_result : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    partial_train_result();

    /// The nobs value.
    /// @remark default = table{}
    const table& get_partial_n_rows() const;

    auto& set_partial_n_rows(const table& value) {
        set_partial_n_rows_impl(value);
        return *this;
    }

    /// The crossproduct matrix.
    /// @remark default = table{}
    const table& get_partial_crossproduct() const;

    auto& set_partial_crossproduct(const table& value) {
        set_partial_crossproduct_impl(value);
        return *this;
    }

    /// Sums.
    /// @remark default = table{}
    const table& get_partial_sum() const;

    auto& set_partial_sum(const table& value) {
        set_partial_sum_impl(value);
        return *this;
    }

    std::int64_t get_auxiliary_table_count() const;

    const table& get_auxiliary_table(const std::int64_t) const;

    auto& set_auxiliary_table(const table& value) {
        set_auxiliary_table_impl(value);
        return *this;
    }

protected:
    void set_partial_n_rows_impl(const table&);
    void set_partial_crossproduct_impl(const table&);
    void set_partial_sum_impl(const table&);
    void set_auxiliary_table_impl(const table&);

private:
    dal::detail::pimpl<detail::partial_train_result_impl<Task>> impl_;
};

template <typename Task = task::by_default>
class partial_train_input : protected train_input<Task> {
public:
    using task_t = Task;

    partial_train_input();

    partial_train_input(const table& data);

    partial_train_input(const partial_train_result<Task>& prev, const table& data);

    const table& get_data() const {
        return train_input<Task>::get_data();
    }

    auto& set_data(const table& value) {
        train_input<Task>::set_data(value);
        return *this;
    }

    const partial_train_result<Task>& get_prev() const {
        return prev_;
    }

    auto& set_prev(const partial_train_result<Task>& value) {
        prev_ = value;
        return *this;
    }

private:
    partial_train_result<Task> prev_;
};

} // namespace v1

using v1::train_input;
using v1::train_result;
using v1::partial_train_input;
using v1::partial_train_result;

} // namespace oneapi::dal::pca
