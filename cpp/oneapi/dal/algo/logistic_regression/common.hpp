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
#include "oneapi/dal/detail/serialization.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/common.hpp"

namespace oneapi::dal::logistic_regression {

namespace task {
namespace v1 {
/// Tag-type that parameterizes entities used for solving
/// :capterm:`regression problem <regression>`.
struct classification {};

/// Alias tag-type for regression task.
using by_default = classification;
} // namespace v1

using v1::classification;
using v1::by_default;

} // namespace task

namespace method {
namespace v1 {
/// Tag-type that denotes :ref:`normal eqution <norm_eq>` computational method.
struct newton_cg {};

using by_default = newton_cg;
} // namespace v1

using v1::newton_cg;
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

/// Return the indices the intercept term in logistic regression
const inline result_option_id intercept = detail::get_intercept_id();

/// Return the coefficients to use in logistic regression
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
constexpr bool is_valid_method_v = dal::detail::is_one_of_v<Method, method::newton_cg>;

template <typename Task>
constexpr bool is_valid_task_v = std::is_same_v<Task, task::classification>;

template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(is_valid_task_v<Task>);

public:
    using tag_t = descriptor_tag;

    descriptor_base();

    descriptor_base(bool compute_intercept, double l2_coef, std::int32_t max_iter, double tol);

    bool get_compute_intercept() const;
    //double get_l1_coef() const;
    double get_l2_coef() const;
    double get_tol() const;
    std::int32_t get_max_iter() const;
    //std::int64_t get_class_count() const;
    result_option_id get_result_options() const;

protected:
    void set_compute_intercept_impl(bool compute_intercept);
    //void set_l1_coef_impl(bool l1_coef);
    void set_l2_coef_impl(double l2_coef);
    void set_tol_impl(double tol);
    void set_max_iter_impl(std::int32_t max_iter);
    //void set_class_count_impl(std::int64_t class_count);
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
///                     be :expr:`method::newton_cg`.
/// @tparam Task        Tag-type that specifies type of the problem to solve. Can
///                     be :expr:`task::classification`.
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
    explicit descriptor(bool compute_intercept = true, double l2_coef = 0.0, std::int32_t maxiter = 100, double tol = 1.0e-4) : base_t(compute_intercept, l2_coef, maxiter, tol) {}

    /// Creates a new instance of the class with default parameters
    //explicit descriptor() : base_t(true) {}

    /// Defines should intercept be taken into consideration.
    bool get_compute_intercept() const {
        return base_t::get_compute_intercept();
    }

    // double get_l1_coef() const {
    //     return base_t::get_l1_coef();
    // }

    double get_l2_coef() const {
        return base_t::get_l2_coef();
    }

    double get_tol() const {
        return base_t::get_tol();
    }

    std::int32_t get_max_iter() const {
        return base_t::get_max_iter();
    }

    // double get_class_count() const {
    //     return base_t::get_class_count();
    // }

    auto& set_compute_intercept(bool compute_intercept) const {
        base_t::set_compute_intercept_impl(compute_intercept);
        return *this;
    }

    // auto& set_l1_coef(bool l1_coef) const {
    //     base_t::set_l1_coef_impl(l1_coef);
    //     return *this;
    // }

    auto& set_l2_coef(bool l2_coef) const {
        base_t::set_l2_coef_impl(l2_coef);
        return *this;
    }


    auto& set_tol(double tol) const {
        base_t::set_tol_impl(tol);
        return *this;
    }

    auto& set_max_iter(std::int32_t maxiter) const {
        base_t::set_max_iter_impl(maxiter);
        return *this;
    }

    // auto& set_class_count(std::int64_t class_count) const {
    //     base_t::set_class_count_impl(class_count);
    //     return *this;
    // }

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
    model& set_packed_coefficients(const table& t);

private:
    //void serialize(dal::detail::output_archive& ar) const;
    //void deserialize(dal::detail::input_archive& ar);

    explicit model(const std::shared_ptr<detail::model_impl<Task>>& impl);
    dal::detail::pimpl<detail::model_impl<Task>> impl_;
};

} // namespace v1

using v1::descriptor;
using v1::model;

} // namespace oneapi::dal::logistic_regression
