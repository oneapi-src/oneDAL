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

#include "oneapi/dal/util/result_option_id.hpp"
#include "oneapi/dal/detail/serialization.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/common.hpp"

namespace oneapi::dal::linear_regression {

namespace task {
namespace v1 {
/// Tag-type that parameterizes entities used for solving
/// :capterm:`regression problem <regression>`.
struct regression {};

/// Alias tag-type for regression task.
using by_default = regression;
} // namespace v1

using v1::regression;
using v1::by_default;

} // namespace task

namespace method {
namespace v1 {
/// Tag-type that denotes :ref:`normal eqution <norm_eq>` computational method.
struct norm_eq {};

using by_default = norm_eq;
} // namespace v1

using v1::norm_eq;
using v1::by_default;

} // namespace method

/// Represents result option flag
/// Behaves like a regular :expr`enum`.
class result_option_id : public result_option_id_base {
public:
    constexpr result_option_id() = default;
    constexpr explicit result_option_id(const result_option_id_base& base)
            : result_option_id_base{ base } {}
};

namespace detail {

ONEDAL_EXPORT result_option_id get_intercept_id();
ONEDAL_EXPORT result_option_id get_coefficients_id();

} // namespace detail

/// Result options are used to define
/// what should algorithm return
namespace result_options {

/// Return the indices the intercept term in linear regression
const inline result_option_id intercept = detail::get_intercept_id();

/// Return the coefficients to use in linear regression
const inline result_option_id coefficients = detail::get_coefficients_id();

} // namespace result_options

namespace detail {
namespace v1 {

struct descriptor_tag {};

template <typename Task>
class descriptor_impl;

template <typename Task>
class model_impl;

template <typename Float>
constexpr bool is_valid_float_v = dal::detail::is_one_of_v<Float, float, double>;

template <typename Method>
constexpr bool is_valid_method_v = dal::detail::is_one_of_v<Method, method::norm_eq>;

template <typename Task>
constexpr bool is_valid_task_v = std::is_same_v<Task, task::regression>;

template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(is_valid_task_v<Task>);

public:
    using tag_t = descriptor_tag;

    descriptor_base();

    descriptor_base(bool compute_intercept);

    bool get_compute_intercept() const;
    double get_alpha() const;
    result_option_id get_result_options() const;

protected:
    void set_compute_intercept_impl(bool compute_intercept);
    void set_alpha_impl(double alpha);
    void set_result_options_impl(const result_option_id& value);

private:
    dal::detail::pimpl<descriptor_impl<Task>> impl_;
};

} // namespace v1

using v1::descriptor_tag;
using v1::descriptor_impl;
using v1::model_impl;
using v1::descriptor_base;

using v1::is_valid_float_v;
using v1::is_valid_method_v;
using v1::is_valid_task_v;

} // namespace detail

namespace v1 {

/// @tparam Float       The floating-point type that the algorithm uses for
///                     intermediate computations. Can be :expr:`float` or
///                     :expr:`double`.
/// @tparam Method      Tag-type that specifies an implementation of algorithm. Can
///                     be :expr:`method::norm_eq`.
/// @tparam Task        Tag-type that specifies type of the problem to solve. Can
///                     be :expr:`task::regression`.
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

    /// Creates a new instance of the class with the given :literal:`compute_intercept`
    explicit descriptor(bool compute_intercept) : base_t(compute_intercept) {}

    /// Creates a new instance of the class with default parameters
    explicit descriptor() : base_t(true) {}

    explicit descriptor(bool compute_intercept, double alpha) : base_t(compute_intercept) {
        set_alpha(alpha);
    }

    explicit descriptor(double alpha) : base_t(true) {
        set_alpha(alpha);
    }

    /// Defines should intercept be taken into consideration.
    bool get_compute_intercept() const {
        return base_t::get_compute_intercept();
    }

    auto& set_compute_intercept(bool compute_intercept) const {
        base_t::set_compute_intercept(compute_intercept);
        return *this;
    }

    /// Defines regularization term alpha used in Ridge Regression
    double get_alpha() const {
        return base_t::get_alpha();
    }

    auto& set_alpha(double value) {
        base_t::set_alpha_impl(value);
        return *this;
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

/// @tparam Task Tag-type that specifies type of the problem to solve.
template <typename Task = task::by_default>
class model : public base {
    static_assert(detail::is_valid_task_v<Task>);
    friend dal::detail::pimpl_accessor;
    friend dal::detail::serialization_accessor;

public:
    /// Creates a new instance of the class with the default property values.
    model();

    const table& get_packed_coefficients() const;
    [[deprecated]] const table& get_betas() const;
    model& set_packed_coefficients(const table& t);

private:
    void serialize(dal::detail::output_archive& ar) const;
    void deserialize(dal::detail::input_archive& ar);

    explicit model(const std::shared_ptr<detail::model_impl<Task>>& impl);
    dal::detail::pimpl<detail::model_impl<Task>> impl_;
};

} // namespace v1

using v1::descriptor;
using v1::model;

} // namespace oneapi::dal::linear_regression
