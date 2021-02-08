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

namespace oneapi::dal::kmeans {

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
/// Tag-type that denotes `Lloyd's <kmeans_t_math_lloyd_>`_ computational
/// method.
struct lloyd_dense {};

/// Alias tag-type for `Lloyd's <kmeans_t_math_lloyd_>`_ computational
/// method.
using by_default = lloyd_dense;
} // namespace v1

using v1::lloyd_dense;
using v1::by_default;

} // namespace method

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
constexpr bool is_valid_method_v = dal::detail::is_one_of_v<Method, method::lloyd_dense>;

template <typename Task>
constexpr bool is_valid_task_v = dal::detail::is_one_of_v<Task, task::clustering>;

template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(is_valid_task_v<Task>);

public:
    using tag_t = descriptor_tag;
    using float_t = float;
    using method_t = method::by_default;
    using task_t = Task;

    descriptor_base();

    /// The number of clusters k
    /// @invariant :expr:`cluster_count > 0`
    /// @remark default = 2
    std::int64_t get_cluster_count() const;

    /// The maximum number of iterations :literal:`T`
    /// @invariant :expr:`max_iteration_count >= 0`
    /// @remark default = 100
    std::int64_t get_max_iteration_count() const;

    /// The threshold $\\varepsilon$ for the stop condition
    /// @invariant :expr:`accuracy_threshold >= 0.0`
    /// @remark default = 0.0
    double get_accuracy_threshold() const;

protected:
    void set_cluster_count_impl(std::int64_t);
    void set_max_iteration_count_impl(std::int64_t);
    void set_accuracy_threshold_impl(double);

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

/// @tparam Float  The floating-point type that the algorithm uses for
///                intermediate computations. Can be :expr:`float` or
///                :expr:`double`.
/// @tparam Method Tag-type that specifies an implementation of algorithm. Can
///                be :expr:`method::v1::lloyd_dense`.
/// @tparam Task   Tag-type that specifies the type of the problem to solve. Can
///                be :expr:`task::v1::clustering`.
template <typename Float = detail::descriptor_base<>::float_t,
          typename Method = detail::descriptor_base<>::method_t,
          typename Task = detail::descriptor_base<>::task_t>
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

    auto& set_cluster_count(int64_t value) {
        base_t::set_cluster_count_impl(value);
        return *this;
    }

    auto& set_max_iteration_count(int64_t value) {
        base_t::set_max_iteration_count_impl(value);
        return *this;
    }

    auto& set_accuracy_threshold(double value) {
        base_t::set_accuracy_threshold_impl(value);
        return *this;
    }
};

/// @tparam Task Tag-type that specifies type of the problem to solve. Can
///              be :expr:`task::v1::clustering`.
template <typename Task = task::by_default>
class model : public base {
    static_assert(detail::is_valid_task_v<Task>);
    friend dal::detail::pimpl_accessor;

public:
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    model();

    /// A $k \\times p$ table with the cluster centroids. Each row of the
    /// table stores one centroid.
    /// @remark default = table{}
    const table& get_centroids() const;

    auto& set_centroids(const table& value) {
        set_centroids_impl(value);
        return *this;
    }

    /// Number of clusters k in the trained model.
    /// @invariant :expr:`cluster_count == centroids.row_count`
    /// @remark default = 0
    std::int64_t get_cluster_count() const;

protected:
    void set_centroids_impl(const table&);

private:
    dal::detail::pimpl<detail::model_impl<Task>> impl_;
};

} // namespace v1

using v1::descriptor;
using v1::model;

} // namespace oneapi::dal::kmeans
