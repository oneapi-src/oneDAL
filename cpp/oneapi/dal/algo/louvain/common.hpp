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
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::preview::louvain {

namespace task {
struct vertex_partitioning {};
using by_default = vertex_partitioning;
} // namespace task

namespace method {
struct fast {};
using by_default = fast;
} // namespace method

namespace detail {
struct descriptor_tag {};

template <typename Task>
class descriptor_impl;

template <typename Method>
constexpr bool is_valid_method = dal::detail::is_one_of_v<Method, method::fast>;

template <typename Task>
constexpr bool is_valid_task = dal::detail::is_one_of_v<Task, task::vertex_partitioning>;

/// The base class for the Louvain algorithm descriptor
template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(is_valid_task<Task>);

public:
    using tag_t = descriptor_tag;
    using float_t = float;
    using method_t = method::by_default;
    using task_t = Task;

    descriptor_base();

    double get_accuracy_threshold() const;
    double get_resolution() const;
    std::int64_t get_max_iteration_count() const;

protected:
    void set_accuracy_threshold(double value);
    void set_resolution(double value);
    void set_max_iteration_count(std::int64_t value);

    dal::detail::pimpl<descriptor_impl<Task>> impl_;
};

} // namespace detail

/// Class for the Louvain algorithm descriptor
///
/// @tparam Float The data type of the result
/// @tparam Method The algorithm method
/// @tparam Task   The task to solve by the algorithm
/// @tparam Allocator   Custom allocator for all memory management inside the
/// algorithm
template <typename Float = float,
          typename Method = method::by_default,
          typename Task = task::by_default,
          typename Allocator = std::allocator<char>>
class descriptor : public detail::descriptor_base<Task> {
    static_assert(detail::is_valid_method<Method>);
    static_assert(detail::is_valid_task<Task>);

    using base_t = detail::descriptor_base<Task>;

public:
    using float_t = Float;
    using method_t = Method;
    using task_t = Task;
    using allocator_t = Allocator;

    explicit descriptor(const Allocator &allocator = std::allocator<char>()) {
        alloc_ = allocator;
    }

    /// Returns the threshold for the stop condition of the local moving
    /// phase of the algorithm
    ///
    /// @remark default = 0.0001
    double get_accuracy_threshold() const {
        return base_t::get_accuracy_threshold();
    }

    /// Sets the threshold for the stop condition of the local moving
    /// phase of the algorithm
    ///
    /// @param [in] accuracy_threshold  modularity threshold value
    /// @invariant :expr:`accuracy_threshold >= 0`
    /// @remark default = 0.0001
    auto &set_accuracy_threshold(double accuracy_threshold) {
        base_t::set_accuracy_threshold(accuracy_threshold);
        return *this;
    }

    /// Returns resolution parameter in the modularity formula
    ///
    /// @remark default = 1.0
    double get_resolution() const {
        return base_t::get_resolution();
    }

    /// Sets resolution parameter in the modularity formula
    ///
    /// @param [in] resolution  Resolution parameter in the modularity formula
    /// @invariant :expr:`resolution >= 0`
    /// @remark default = 1.0
    auto &set_resolution(double resolution) {
        base_t::set_resolution(resolution);
        return *this;
    }

    /// Returns the maximum number of iterations of the Louvain algorithm
    ///
    /// @remark default = 10
    std::int64_t get_max_iteration_count() const {
        return base_t::get_max_iteration_count();
    }

    /// Sets the maximum number of iterations of the Louvain algorithm
    ///
    /// @param [in] max_iteration_count  Maximum number of iterations of the
    ///                                  Louvain algorithm
    /// @invariant :expr:`max_iteration_count >= 0`
    /// @remark default = 10
    auto &set_max_iteration_count(std::int64_t max_iteration_count) {
        base_t::set_max_iteration_count(max_iteration_count);
        return *this;
    }

    Allocator get_allocator() const {
        return alloc_;
    }

private:
    Allocator alloc_;
};

namespace detail {

template <typename Graph>
constexpr bool is_valid_graph =
    dal::detail::is_one_of_v<Graph,
                             undirected_adjacency_vector_graph<vertex_user_value_type<Graph>,
                                                               edge_user_value_type<Graph>,
                                                               graph_user_value_type<Graph>,
                                                               vertex_type<Graph>,
                                                               graph_allocator<Graph>>>;

} // namespace detail
} // namespace oneapi::dal::preview::louvain
