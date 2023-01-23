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

#include "oneapi/dal/util/result_option_id.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/serialization.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/common.hpp"

namespace oneapi::dal::logloss_objective {

namespace task {
    struct logloss{};
    using by_default = logloss;
} // namespace task

namespace method {
    struct dense_batch{};
    using by_default = dense_batch;
} // namespace method

class result_option_id : public result_option_id_base {
public:
    constexpr result_option_id() = default;
    constexpr explicit result_option_id(const result_option_id_base& base)
            : result_option_id_base{ base } {}
};

namespace detail {

ONEDAL_EXPORT result_option_id get_value_id();
ONEDAL_EXPORT result_option_id get_gradient_id();
ONEDAL_EXPORT result_option_id get_hessian_id();
ONEDAL_EXPORT result_option_id get_packed_gradient_id();
ONEDAL_EXPORT result_option_id get_packed_hessian_id();

} // namespace detail

namespace result_options {
    const inline auto value = detail::get_value_id();
    const inline auto gradient = detail::get_gradient_id();
    const inline auto hessian = detail::get_hessian_id();
    const inline auto packed_gradient = detail::get_packed_gradient_id();
    const inline auto packed_hessian = detail::get_packed_hessian_id();
} // namespace result_options

namespace detail {

struct descriptor_tag {};

template <typename Task>
class descriptor_impl;

template <typename Float>
constexpr bool is_valid_float_v = dal::detail::is_one_of_v<Float, float, double>;

template <typename Method>
constexpr bool is_valid_method_v = dal::detail::is_one_of_v<Method, method::dense_batch>;

template <typename Task>
constexpr bool is_valid_task_v = dal::detail::is_one_of_v<Task, task::logloss>;

template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(is_valid_task_v<Task>);

public:
    using tag_t = descriptor_tag;
    using float_t = float;
    using method_t = method::by_default;
    using task_t = Task;

    descriptor_base();

    result_option_id get_result_options() const;
    double get_l1_regularization_coefficient() const;
    double get_l2_regularization_coefficient() const;

protected:
    void set_result_options_impl(const result_option_id& value);
    void set_l1_regularization_coefficient_impl(double l1_coef);
    void set_l2_regularization_coefficient_impl(double l2_coef);

private:
    dal::detail::pimpl<descriptor_impl<Task>> impl_;
};


template <typename Float = float,
          typename Method = method::by_default,
          typename Task = task::by_default>
class descriptor : public detail::descriptor_base<Task> {
    static_assert(detail::is_valid_float_v<Float>);
    static_assert(detail::is_valid_method_v<Method>);
    static_assert(detail::is_valid_task_v<Task>);

    using base_t = detail::descriptor_base<Task>;

public:
    using float_t = Float;
    using method_t = Method;
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    descriptor(double l1_regularization_coefficient = 0.0, double l2_regularization_coefficient = 0.0) {
        set_l1_regularization_coefficient(l1_regularization_coefficient);
        set_l2_regularization_coefficient(l2_regularization_coefficient);

    }

    double get_l1_regularization_coefficient() const {
        return base_t::get_l1_regularization_coefficient();
    }
    auto& set_l1_regularization_coefficient(double value) {
        base_t::set_l1_regularization_coefficient_impl(value);
        return *this;
    }
    
    auto& set_l2_regularization_coefficient(double value) {
        base_t::set_l2_regularization_coefficient_impl(value);
        return *this;
    }
    double get_l2_regularization_coefficient() const {
        return base_t::get_l2_regularization_coefficient();
    }

    /// Choose which results should be computed and returned.
    result_option_id get_result_options() const {
        return base_t::get_result_options();
    }

    auto& set_result_options(const result_option_id& value) {
        base_t::set_result_options_impl(value);
        return *this;
    }   
};

} // namespace detail

} // namespace oneapi::dal::objective_function
