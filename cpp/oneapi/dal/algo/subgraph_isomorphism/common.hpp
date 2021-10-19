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

namespace oneapi::dal::preview::subgraph_isomorphism {

namespace task {
struct compute {};
using by_default = compute;
} // namespace task

namespace method {
struct fast {};
using by_default = fast;
} // namespace method

enum class kind { induced, non_induced };

namespace detail {
struct descriptor_tag {};

template <typename Task>
class descriptor_impl;

template <typename Method>
constexpr bool is_valid_method = dal::detail::is_one_of_v<Method, method::fast>;

template <typename Task>
constexpr bool is_valid_task = dal::detail::is_one_of_v<Task, task::compute>;

/// The base class for the Subgraph Isomorphism algorithm descriptor
template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(is_valid_task<Task>);

public:
    using tag_t = descriptor_tag;
    using float_t = float;
    using method_t = method::by_default;
    using task_t = Task;

    /// Constructs the empty descriptor
    descriptor_base();

    /// Returns the kind of searched subgraph which is isomorphic to pattern graph
    auto get_kind() const -> subgraph_isomorphism::kind;

    /// Returns if semantic search is required
    auto get_semantic_match() const -> bool;

    /// Returns the maximum number of matches to search
    auto get_max_match_count() const -> std::int64_t;

protected:
    void set_kind(kind value);
    void set_semantic_match(bool semantic_match);
    void set_max_match_count(std::int64_t max_match_count);

    dal::detail::pimpl<descriptor_impl<Task>> impl_;
};

} // namespace detail

/// Class for the Subgraph Isomorphism algorithm descriptor
///
/// @tparam Float The data type of the result
/// @tparam Method The algorithm method
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

    explicit descriptor(Allocator allocator = std::allocator<char>()) {
        alloc_ = allocator;
    }

    /// Returns kind of subgraph isomorphism
    kind get_kind() const {
        return base_t::get_kind();
    }

    /// Sets the type of searched subgraph in Subgraph Isomorphism computation
    ///
    /// @param [in] value  The begin of the row of the graph block
    auto& set_kind(kind value) {
        base_t::set_kind(value);
        return *this;
    }

    /// Returns the flag if semantic search is requred in Subgraph Isomorphism computation
    bool get_semantic_match() const {
        return base_t::get_semantic_match();
    }

    /// Returns the flag if semantic search is requred in Subgraph Isomorphism computation
    ///
    /// @param [in] semantic_match The flag if semantic search is requred
    auto& set_semantic_match(bool semantic_match) {
        base_t::set_semantic_match(semantic_match);
        return *this;
    }

    /// Sets the maximum number of matchings to search in Subgraph Isomorphism computation
    std::int64_t get_max_match_count() const {
        return base_t::get_max_match_count();
    }

    /// Sets the maximum number of matchings to search in Subgraph Isomorphism computation
    ///
    /// @param [in] max_match_count  The maximum number of matchings
    auto& set_max_match_count(std::int64_t max_match_count) {
        base_t::set_max_match_count(max_match_count);
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
} // namespace oneapi::dal::preview::subgraph_isomorphism
