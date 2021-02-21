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

/// @file
/// Contains the definition of the input and output for subgraph_isomorphism Similarity
/// algorithm

#pragma once

#include "oneapi/dal/algo/subgraph_isomorphism/common.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/detail/graph_matching_types.hpp"

namespace oneapi::dal::preview {
namespace subgraph_isomorphism {

/// Class for the description of the input parameters of the subgraph_isomorphism Similarity
/// algorithm
///
/// @tparam Graph  Type of the input graph
template <typename Graph>
class ONEDAL_EXPORT graph_matching_input {
public:
    static_assert(detail::is_valid_graph<Graph>,
                  "Only undirected_adjacency_vector_graph is supported.");
    /// Constructs the algorithm input initialized with the graph and the caching builder.
    ///
    /// @param [in]   graph  The input graph
    /// @param [in/out]  builder  The caching builder
    graph_matching_input(const Graph& target_graph, const Graph& patter_graph);

    /// Returns the constant reference to the input graph
    const Graph& get_target_graph() const;

    /// Returns the constant reference to the input graph
    const Graph& get_pattern_graph() const;

private:
    dal::detail::pimpl<detail::graph_matching_input_impl<Graph>> impl_;
};

/// Class for the description of the result of the subgraph_isomorphism Similarity algorithm
class ONEDAL_EXPORT graph_matching_result {
public:
    /// Constructs the empty result
    graph_matching_result();

    /// Constructs the algorithm result initialized with the table of vertex pairs,
    /// the table of the corresponding computed subgraph_isomorphism similarity coefficients, and
    /// the number of non-zero subgraph_isomorphism similarity coefficients in the block.
    ///
    /// @param [in]   vertex_pairs        The table of size [nonzero_coeff_count x 2] with
    ///                                   vertex pairs which have non-zero subgraph_isomorphism
    ///                                   similarity coefficients
    /// @param [in]   coeffs              The table of size [nonzero_coeff_count x 1] with
    ///                                   non-zero subgraph_isomorphism similarity coefficients
    /// @param [in]   nonzero_coeff_count The number of non-zero subgraph_isomorphism coefficients
    graph_matching_result(const table& vertex_pairs,
                          const table& coeffs,
                          std::int64_t nonzero_coeff_count);

    /// Returns the table of size [nonzero_coeff_count x 1] with non-zero subgraph_isomorphism
    /// similarity coefficients
    table get_vertex_match() const;

    /// The number of non-zero subgraph_isomorphism similarity coefficients in the block
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
