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

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/util/result_option_id.hpp"

namespace oneapi::dal::dbscan {

namespace task {
namespace v1 {
/// Tag-type that parameterizes entities used for solving
/// :capterm:`clustering problem <clustering>`.
struct clustering {};

/// Alias tag-type for the clustering task.
using by_default = clustering;
} // namespace v1

using v1::clustering;
using v1::by_default;

} // namespace task

namespace method {
namespace v1 {
struct brute_force {};

using by_default = brute_force;
} // namespace v1

using v1::brute_force;
using v1::by_default;

} // namespace method

/// Represents result option flag
/// Behaves like a regular :expr`enum`.
class result_option_id : public result_option_id_base {
public:
    result_option_id() : result_option_id_base{} {}
    result_option_id(const result_option_id_base& base) : result_option_id_base{ base } {}
};

namespace detail {

ONEDAL_EXPORT result_option_id get_responses_id();
ONEDAL_EXPORT result_option_id get_core_observation_indices_id();
ONEDAL_EXPORT result_option_id get_core_observations_id();
ONEDAL_EXPORT result_option_id get_core_flags_id();

} // namespace detail

/// Result options are used to define
/// what should algorithm return
namespace result_options {

const inline result_option_id responses = detail::get_responses_id();
const inline result_option_id core_observation_indices = detail::get_core_observation_indices_id();
const inline result_option_id core_observations = detail::get_core_observations_id();
const inline result_option_id core_flags = detail::get_core_flags_id();

} // namespace result_options

namespace detail {
namespace v1 {
struct descriptor_tag {};

template <typename Task>
class descriptor_impl;

template <typename Float>
constexpr bool is_valid_float_v = dal::detail::is_one_of_v<Float, float, double>;

template <typename Method>
constexpr bool is_valid_method_v = dal::detail::is_one_of_v<Method, method::brute_force>;

template <typename Task>
constexpr bool is_valid_task_v = dal::detail::is_one_of_v<Task, task::clustering>;

template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(is_valid_task_v<Task>);

public:
    using tag_t = descriptor_tag;
    using method_t = method::by_default;
    using task_t = Task;

    descriptor_base();

    double get_epsilon() const;
    std::int64_t get_min_observations() const;
    bool get_mem_save_mode() const;
    result_option_id get_result_options() const;

protected:
    void set_min_observations_impl(std::int64_t);
    void set_epsilon_impl(double);
    void set_mem_save_mode_impl(bool);
    void set_result_options_impl(const result_option_id& value);

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

/// @tparam Float  The floating-point type that the algorithm uses for
///                intermediate computations. Can be :expr:`float` or
///                :expr:`double`.
/// @tparam Method Tag-type that specifies an implementation of algorithm. Can
///                be :expr:`method::brute_force`.
/// @tparam Task   Tag-type that specifies the type of the problem to solve. Can
///                be :expr:`task::clustering`.
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

    /// Creates a new instance of the class with the given :literal:`epsilon`, :literal:`min_observations`
    explicit descriptor(double epsilon, std::int64_t min_observations) {
        set_min_observations(min_observations);
        set_epsilon(epsilon);
    }

    /// The number of neighbors
    std::int64_t get_min_observations() const {
        return base_t::get_min_observations();
    }

    auto& set_min_observations(std::int64_t value) {
        base_t::set_min_observations_impl(value);
        return *this;
    }

    /// The distance `epsilon` for neighbor search
    /// @invariant :expr:`epsilon >= 0.0`
    double get_epsilon() const {
        return base_t::get_epsilon();
    }

    auto& set_epsilon(double value) {
        base_t::set_epsilon_impl(value);
        return *this;
    }

    /// The flag for memory saving mode
    bool get_mem_save_mode() const {
        return base_t::get_mem_save_mode();
    }

    auto& set_mem_save_mode(bool value) {
        base_t::set_mem_save_mode_impl(value);
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

} // namespace v1

using v1::descriptor;

} // namespace oneapi::dal::dbscan
