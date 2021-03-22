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

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::kmeans_init {

namespace task {
namespace v1 {
/// Tag-type that parameterizes entities used for obtaining the initial K-Means centroids.
struct init {};

/// Alias tag-type for the initialization task.
using by_default = init;
} // namespace v1

using v1::init;
using v1::by_default;

} // namespace task

namespace method {
namespace v1 {
/// Tag-type that denotes :ref:`dense <kmeans_init_c_math_dense>`
/// computational method.
struct dense {};
struct random_dense {};
struct plus_plus_dense {};
struct parallel_plus_dense {};
using by_default = dense;
} // namespace v1

using v1::dense;
using v1::random_dense;
using v1::plus_plus_dense;
using v1::parallel_plus_dense;
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
constexpr bool is_valid_method_v = dal::detail::is_one_of_v<Method,
                                                            method::dense,
                                                            method::random_dense,
                                                            method::plus_plus_dense,
                                                            method::parallel_plus_dense>;

template <typename Task>
constexpr bool is_valid_task_v = dal::detail::is_one_of_v<Task, task::init>;

template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(is_valid_task_v<Task>);

public:
    using tag_t = descriptor_tag;
    using float_t = float;
    using method_t = method::by_default;
    using task_t = Task;

    descriptor_base();

    std::int64_t get_cluster_count() const;

protected:
    void set_cluster_count_impl(std::int64_t);

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
/// @tparam Method Tag-type that specifies an implementation
///                of K-Means Initialization algorithm.
/// @tparam Task   Tag-type that specifies the type of the problem to solve. Can
///                be :expr:`task::init`.
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

    /// Creates a new instance of the class with the given :literal:`cluster_count`
    explicit descriptor(std::int64_t cluster_count = 2) {
        set_cluster_count(cluster_count);
    }

    /// The number of clusters k
    /// @invariant :expr:`cluster_count > 0`
    /// @remark default = 2
    std::int64_t get_cluster_count() const {
        return base_t::get_cluster_count();
    }

    auto& set_cluster_count(int64_t value) {
        base_t::set_cluster_count_impl(value);
        return *this;
    }
};

} // namespace v1

using v1::descriptor;

} // namespace oneapi::dal::kmeans_init
