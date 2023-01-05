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

/// @file
/// Contains the definition of the input and output for Subgraph Isomorphism
/// algorithm

#pragma once

#include "oneapi/dal/algo/subgraph_isomorphism/common.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/detail/graph_matching_types.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism {

/// Class for the description of the input parameters of the Subgraph Isomorphism algorithm
///
/// @tparam Graph  The type of the input graph.
/// @tparam Task   Tag-type that specifies the type of the problem to solve.
///                Can be :expr:`task::compute`.
template <typename Graph, typename Task = task::compute>
class graph_matching_input : public base {
    static_assert(detail::is_valid_task<Task>);

public:
    using task_t = Task;
    static_assert(detail::is_valid_graph<Graph>,
                  "Only undirected_adjacency_vector_graph is supported.");
    /// Constructs the algorithm input initialized with the target and pattern graphs.
    ///
    /// @param [in] target_graph  The input target (bigger) graph
    /// @param [in] pattern_graph  The input pattern (smaller) graph
    graph_matching_input(const Graph& target_graph, const Graph& pattern_graph);

    /// Returns the constant reference to the input target graph
    const Graph& get_target_graph() const;

    /// Sets the target (bigger) graph to the input
    /// @param [in] target_graph  The input target (bigger) graph
    const auto& set_target_graph(const Graph& target_graph);

    /// Returns the constant reference to the input pattern graph
    const Graph& get_pattern_graph() const;

    /// Sets the pattern (smaller) graph to the input
    /// @param [in] pattern_graph  The input pattern (smaller) graph
    const auto& set_pattern_graph(const Graph& pattern_graph);

private:
    dal::detail::pimpl<detail::graph_matching_input_impl<Graph, Task>> impl_;
};

/// Class for the description of the result of the Subgraph Isomorphism algorithm
template <typename Task = task::by_default>
class graph_matching_result {
    static_assert(detail::is_valid_task<Task>);

public:
    using task_t = Task;
    /// Constructs the empty result
    graph_matching_result();

    /// Returns the table of size [match_count x pattern_vertex_count] with matchings
    /// of the pattern graph in the target graph. Each row of the table
    /// contain ids of vertices in target graph sorted by pattern vertex ids.
    /// I.e. j-th element of i-th row contain id of target graph vertex which
    /// was matched with j-th vertex of pattern graph in i-th match.
    const table& get_vertex_match() const {
        return get_vertex_match_impl();
    }

    /// The number pattern matches in the target graph.
    std::int64_t get_match_count() const {
        return get_match_count_impl();
    }

    /// Sets the table with matchings of the pattern graph in the target graph.
    auto& set_vertex_match(const table& value) {
        set_vertex_match_impl(value);
        return *this;
    }

    /// Sets the maximum number of pattern matches in the target graph.
    auto& set_match_count(std::int64_t value) {
        set_match_count_impl(value);
        return *this;
    }

private:
    const table& get_vertex_match_impl() const;
    std::int64_t get_match_count_impl() const;
    void set_vertex_match_impl(const table& value);
    void set_match_count_impl(std::int64_t value);
    dal::detail::pimpl<detail::graph_matching_result_impl> impl_;
};

template <typename Graph, typename Task>
graph_matching_input<Graph, Task>::graph_matching_input(const Graph& target_graph,
                                                        const Graph& pattern_graph)
        : impl_(new detail::graph_matching_input_impl<Graph, Task>(target_graph, pattern_graph)) {}

template <typename Graph, typename Task>
const Graph& graph_matching_input<Graph, Task>::get_target_graph() const {
    return impl_->target_graph;
}

template <typename Graph, typename Task>
const Graph& graph_matching_input<Graph, Task>::get_pattern_graph() const {
    return impl_->pattern_graph;
}

} // namespace oneapi::dal::preview::subgraph_isomorphism
