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

#include "oneapi/dal/algo/logloss_objective/common.hpp"
#include "oneapi/dal/algo/objective_function/detail/objective.hpp"
#include "oneapi/dal/util/result_option_id.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/serialization.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/common.hpp"

namespace oneapi::dal::objective_function {

namespace task {
namespace v1 {

struct compute {};

/// Alias tag-type for compute task.
using by_default = compute;
} // namespace v1

using v1::compute;
using v1::by_default;

} // namespace task

namespace method {
namespace v1 {

struct dense {};
using by_default = dense;

} // namespace v1

using v1::dense;
using v1::by_default;

}


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
// ONEDAL_EXPORT result_option_id get_packed_gradient_id();
// ONEDAL_EXPORT result_option_id get_packed_hessian_id();

} // namespace detail

namespace result_options {
    const inline auto value = detail::get_value_id();
    const inline auto gradient = detail::get_gradient_id();
    const inline auto hessian = detail::get_hessian_id();
    // const inline auto packed_gradient = detail::get_packed_gradient_id();
    // const inline auto packed_hessian = detail::get_packed_hessian_id();
} // namespace result_options

namespace detail {

namespace v1 {

struct descriptor_tag {};

template <typename Task>
class descriptor_impl;

template <typename Float>
constexpr bool is_valid_float_v = dal::detail::is_one_of_v<Float, float, double>;

template <typename Method>
constexpr bool is_valid_method_v = dal::detail::is_one_of_v<Method, method::dense>;

template <typename Task>
constexpr bool is_valid_task_v = dal::detail::is_one_of_v<Task, task::compute>;

template <typename Objective>
constexpr bool is_valid_objective_v = dal::detail::is_one_of_v<Objective, logloss_objective::descriptor<float>, logloss_objective::descriptor<double>>;

template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(is_valid_task_v<Task>);
    friend detail::objective_accessor;

public:
    using tag_t = descriptor_tag;
    using float_t = float;
    using method_t = method::by_default;
    using task_t = Task;
    using objective_t = logloss_objective::descriptor<float_t>;

    descriptor_base();

    result_option_id get_result_options() const;

protected:
    explicit descriptor_base(const detail::objective_ptr& objective);

    void set_result_options_impl(const result_option_id& value);
    const detail::objective_ptr& get_objective_impl() const;
    void set_objective_impl(const detail::objective_ptr& objective);

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
using v1::is_valid_objective_v;

} // namespace detail

namespace v1 {


template <typename Float = float,
          typename Method = method::by_default,
          typename Task = task::by_default,
          typename Objective = logloss_objective::descriptor<Float>>
class descriptor : public detail::descriptor_base<Task> {
    static_assert(detail::is_valid_float_v<Float>);
    static_assert(detail::is_valid_method_v<Method>);
    static_assert(detail::is_valid_task_v<Task>);
    static_assert(detail::is_valid_objective_v<Objective>);
    using base_t = detail::descriptor_base<Task>;

public:
    using float_t = Float;
    using method_t = Method;
    using task_t = Task;
    using objective_t = Objective;


    explicit descriptor() : 
    base_t(std::make_shared<detail::objective<objective_t>>
    (objective_t{})) {
    }

    explicit descriptor(const objective_t& obj) {
        set_objective(obj);
    }


    /// Choose which results should be computed and returned.
    result_option_id get_result_options() const {
        return base_t::get_result_options();
    }

    auto& set_result_options(const result_option_id& value) {
        base_t::set_result_options_impl(value);
        return *this;
    }

    auto& set_objective(const objective_t& obj) {
        base_t::set_objective_impl(
            std::make_shared<detail::objective<objective_t>>(obj));
        return *this;
    }

    const objective_t& get_objective() {
        const auto obj = std::static_pointer_cast<objective_t>(base_t::get_objective_impl());
        return obj;
    } 
};



} // namespace v1

using v1::descriptor;

} // namespace oneapi::dal::objective_function
