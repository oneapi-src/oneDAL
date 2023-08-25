/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/algo/covariance/common.hpp"

namespace oneapi::dal::covariance {

namespace detail {
namespace v1 {
template <typename Task>
class compute_input_impl;

template <typename Task>
class compute_result_impl;

template <typename Task>
class partial_compute_result_impl;
} // namespace v1

using v1::compute_input_impl;
using v1::compute_result_impl;
using v1::partial_compute_result_impl;

} // namespace detail

namespace v1 {

/// @tparam Task Tag-type that specifies the type of the problem to solve. Can
///              be :expr:`task::compute`.
template <typename Task = task::by_default>
class compute_input : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;
    compute_input();
    /// Creates a new instance of the class with the given :literal:`data`
    /// property value
    compute_input(const table& data);

    /// An $n \\times p$ table with the training data, where each row stores one
    /// feature vector.
    /// @remark default = table{}
    const table& get_data() const;

    auto& set_data(const table& value) {
        set_data_impl(value);
        return *this;
    }

protected:
    void set_data_impl(const table& value);

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

    /// The covariance matrix.
    /// @remark default = table{}
    const table& get_cov_matrix() const;

    auto& set_cov_matrix(const table& value) {
        set_cov_matrix_impl(value);
        return *this;
    }

    /// The correlation matrix.
    /// @remark default = table{}
    const table& get_cor_matrix() const;

    auto& set_cor_matrix(const table& value) {
        set_cor_matrix_impl(value);
        return *this;
    }

    /// Means.
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
    void set_cov_matrix_impl(const table&);
    void set_cor_matrix_impl(const table&);
    void set_means_impl(const table&);
    void set_result_options_impl(const result_option_id&);

private:
    dal::detail::pimpl<detail::compute_result_impl<Task>> impl_;
};

template <typename Task = task::by_default>
class partial_compute_result : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    partial_compute_result();

    /// The nobs value.
    /// @remark default = table{}
    const table& get_nobs() const;

    auto& set_nobs(const table& value) {
        set_nobs_impl(value);
        return *this;
    }

    /// The crossproduct matrix.
    /// @remark default = table{}
    const table& get_crossproduct() const;

    auto& set_crossproduct(const table& value) {
        set_crossproduct_impl(value);
        return *this;
    }

    /// Sums.
    /// @remark default = table{}
    const table& get_sums() const;

    auto& set_sums(const table& value) {
        set_sums_impl(value);
        return *this;
    }

protected:
    void set_nobs_impl(const table&);
    void set_crossproduct_impl(const table&);
    void set_sums_impl(const table&);

private:
    dal::detail::pimpl<detail::partial_compute_result_impl<Task>> impl_;
};

template <typename Task = task::by_default>
class partial_compute_input : protected compute_input<Task> {
public:
    using task_t = Task;

    partial_compute_input();

    partial_compute_input(const table& data);

    partial_compute_input(const partial_compute_result<Task>& prev, const table& data);

    const table& get_data() const {
        return compute_input<Task>::get_data();
    }

    auto& set_data(const table& value) {
        compute_input<Task>::set_data(value);
        return *this;
    }

    const partial_compute_result<Task>& get_prev() const {
        return prev_;
    }

    auto& set_prev(const partial_compute_result<Task>& value) {
        prev_ = value;
        return *this;
    }

private:
    partial_compute_result<Task> prev_;
};

} // namespace v1

using v1::compute_input;
using v1::compute_result;
using v1::partial_compute_input;
using v1::partial_compute_result;

} // namespace oneapi::dal::covariance
