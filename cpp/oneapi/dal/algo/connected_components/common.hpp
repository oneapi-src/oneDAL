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

namespace oneapi::dal::preview::connected_components {

namespace task {
/// Tag-type that parameterizes entities that are used for Connected Components algorithm.
struct vertex_partitioning {};

/// Alias tag-type for the vertex partitioning task.
using by_default = vertex_partitioning;
} // namespace task

namespace method {
/// Tag-type that denotes Afforest computational method.
struct afforest {};

/// Alias tag-type for Afforest computational method.
using by_default = afforest;
} // namespace method

namespace detail {
struct descriptor_tag {};

template <typename Task>
class descriptor_impl;

template <typename M>
using enable_if_afforest_t = std::enable_if_t<dal::detail::is_one_of_v<M, method::afforest>>;

template <typename Method>
constexpr bool is_valid_method = dal::detail::is_one_of_v<Method, method::afforest>;

template <typename Task>
constexpr bool is_valid_task = dal::detail::is_one_of_v<Task, task::vertex_partitioning>;

/// The base class for the Connected Components algorithm descriptor
template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(is_valid_task<Task>);

public:
    using tag_t = descriptor_tag;
    using float_t = float;
    using method_t = method::by_default;
    using task_t = Task;

    descriptor_base();

protected:
    dal::detail::pimpl<descriptor_impl<Task>> impl_;
};

} // namespace detail

/// Class for the Connected Components algorithm descriptor
///
/// @tparam Float       This parameter is not used for Connected Components algorithm.
/// @tparam Method      Tag-type that specifies the implementation of the algorithm. Can
///                     be :expr:`method::afforest`.
/// @tparam Task        Tag-type that specifies the type of the problem to solve. Can
///                     be :expr:`task::vertex_partitioning`.
/// @tparam Allocator   Custom allocator for all memory management inside the algorithm.
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

    explicit descriptor(const Allocator& allocator = std::allocator<char>()) {
        alloc_ = allocator;
    }

    /// Returns a copy of the allocator used in the algorithm for internal memory management.
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
} // namespace oneapi::dal::preview::connected_components
