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

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/serialization.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/common.hpp"

namespace oneapi::dal::logloss_objective {

namespace task {

namespace v1 {

struct compute {};
using by_default = compute;

} // namespace v1

using v1::compute;
using v1::by_default;

} // namespace task

namespace method {
namespace v1 {
struct dense_batch {};
using by_default = dense_batch;

} // namespace v1

using v1::dense_batch;
using v1::by_default;
} // namespace method

namespace detail {

namespace v1 {

struct descriptor_tag {};

template <typename Task>
class descriptor_impl;

template <typename Float>
constexpr bool is_valid_float_v = dal::detail::is_one_of_v<Float, float, double>;

template <typename Method>
constexpr bool is_valid_method_v = dal::detail::is_one_of_v<Method, method::dense_batch>;

template <typename Task>
constexpr bool is_valid_task_v = dal::detail::is_one_of_v<Task, task::compute>;

template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(is_valid_task_v<Task>);

public:
    using tag_t = descriptor_tag;
    using float_t = float;
    using method_t = method::by_default;
    using task_t = Task;

    descriptor_base();

    double get_l1_regularization_coefficient() const;
    double get_l2_regularization_coefficient() const;
    bool get_intercept_flag() const;

protected:
    void set_l1_regularization_coefficient_impl(double l1_coef);
    void set_l2_regularization_coefficient_impl(double l2_coef);
    void set_intercept_flag_impl(bool fit_intercept);

private:
    dal::detail::pimpl<descriptor_impl<Task>> impl_;
};

} // namespace v1

using v1::descriptor_tag;
using v1::descriptor_impl;
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
///                     be :expr:`method::dense_batch`.
/// @tparam Task        Tag-type that specifies the type of the problem to solve. Can
///                     be :expr:`task::compute`.
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

    /// Creates a new instance of the class with the given :literal:`l1_regularization_coefficient`,
    /// :literal:`l2_regularization_coefficient` and :literal:`fit_intercept` property values.
    explicit descriptor(double l1_regularization_coefficient = 0.0,
                        double l2_regularization_coefficient = 0.0,
                        bool fit_intercept = true) {
        set_l1_regularization_coefficient(l1_regularization_coefficient);
        set_l2_regularization_coefficient(l2_regularization_coefficient);
        set_intercept_flag(fit_intercept);
    }

    /// The L1-regularization strength
    /// @invariant :expr:`l1_regularization_coefficient >= 0.0`
    double get_l1_regularization_coefficient() const {
        return base_t::get_l1_regularization_coefficient();
    }

    auto& set_l1_regularization_coefficient(double value) {
        base_t::set_l1_regularization_coefficient_impl(value);
        return *this;
    }

    /// The L2-regularization strength
    /// @invariant :expr:`l2_regularization_coefficient >= 0.0`
    double get_l2_regularization_coefficient() const {
        return base_t::get_l2_regularization_coefficient();
    }

    auto& set_l2_regularization_coefficient(double value) {
        base_t::set_l2_regularization_coefficient_impl(value);
        return *this;
    }

    /// The fit_intercept flag.
    bool get_intercept_flag() const {
        return base_t::get_intercept_flag();
    }

    auto& set_intercept_flag(bool fit_intercept) {
        base_t::set_intercept_flag_impl(fit_intercept);
        return *this;
    }
};

} // namespace v1

using v1::descriptor;

} // namespace oneapi::dal::logloss_objective
