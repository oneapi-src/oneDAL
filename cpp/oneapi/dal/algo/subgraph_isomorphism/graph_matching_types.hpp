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

namespace oneapi::dal::preview {
namespace subgraph_isomorphism {

/// Class for the description of the input parameters of the Subgraph Isomorphism
/// algorithm
///
/// @tparam Graph  Type of the input graph
template <typename Graph>
class ONEDAL_EXPORT graph_matching_input {
public:
    static_assert(detail::is_valid_graph<Graph>,
                  "Only undirected_adjacency_vector_graph is supported.");
    /// Constructs the algorithm input initialized with the target and pattern graphs.
    ///
    /// @param [in] target_graph  The input target (big) graph
    /// @param [in] pattern_graph  The input patern (small) graph
    graph_matching_input(const Graph& target_graph, const Graph& patter_graph);

    /// Returns the constant reference to the input target graph
    const Graph& get_target_graph() const;

    /// Returns the constant reference to the input pattern graph
    const Graph& get_pattern_graph() const;

private:
    dal::detail::pimpl<detail::graph_matching_input_impl<Graph>> impl_;
};

/// Class for the description of the result of the Subgraph Isomorphism algorithm
class ONEDAL_EXPORT graph_matching_result {
public:
    /// Constructs the empty result
    graph_matching_result();

    /// Constructs the algorithm result initialized with the table of vertex matchings and
    /// the number of pattern matchings in target graph.
    ///
    /// @param [in]   vertex_match        The table of size [match_count x pattern_vertex_count] with
    ///                                   matchings of pattern graph in target graph. Each row of the table
    ///                                   contain ids of vertices in target graph sorted by pattern vertex ids.
    ///                                   I.e. j-th element of i-th row contain id of target graph vertex which
    ///                                   was matched with j-th vertex of pattern graph in i-th match.
    /// @param [in]   match_count         The number pattern matches in the target graph.
    graph_matching_result(const table& vertex_match, std::int64_t match_count);

    /// Returns the table of size [match_count x pattern_vertex_count] with matchings of pattern graph
    /// in target graph.
    table get_vertex_match() const;

    /// The number pattern matches in the target graph.
    std::int64_t get_match_count() const;

private:
    dal::detail::pimpl<detail::graph_matching_result_impl> impl_;
};

template <typename Graph>
graph_matching_input<Graph>::graph_matching_input(const Graph& target_graph,
                                                  const Graph& pattern_graph)
        : impl_(new detail::graph_matching_input_impl<Graph>(target_graph, pattern_graph)) {}

template <typename Graph>
const Graph& graph_matching_input<Graph>::get_target_graph() const {
    return impl_->target_graph;
}

template <typename Graph>
const Graph& graph_matching_input<Graph>::get_pattern_graph() const {
    return impl_->pattern_graph;
}

} // namespace subgraph_isomorphism
} // namespace oneapi::dal::preview
