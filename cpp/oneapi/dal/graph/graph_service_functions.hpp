/* file: graph_service_functions.hpp */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
/// Contains the service functionality for the graph objects

#pragma once

#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/graph/detail/graph_service_functions_impl.hpp"
#include "oneapi/dal/graph/graph_common.hpp"

namespace oneapi::dal::preview {

/// Returns the number of vertices in the graph
///
/// @tparam Graph  Type of the graph
///
/// @param [in]   graph  Input graph object
///
/// @return The number of vertices in the graph
template <typename Graph>
constexpr auto get_vertex_count(const Graph &graph) noexcept -> vertex_size_type<Graph> {
    return detail::get_vertex_count_impl(graph);
}

/// Returns the number of edges in the graph
///
/// @tparam Graph  Type of the graph
///
/// @param [in]   graph  Input graph object
///
/// @return The number of edges in the graph
template <typename Graph>
constexpr auto get_edge_count(const Graph &graph) noexcept -> edge_size_type<Graph> {
    return detail::get_edge_count_impl(graph);
}

/// Returns the degree for the specified vertex
///
/// @tparam Graph  Type of the graph
///
/// @param [in]   graph  Input graph object
///
/// @param [in]   vertex Identifier of the vertex
///
/// @return The degree of the vertex
template <typename Graph>
constexpr auto get_vertex_degree(const Graph &graph, vertex_type<Graph> vertex)
    -> edge_size_type<Graph> {
    if (vertex < 0 || (vertex_size_type<Graph>)vertex >= detail::get_vertex_count_impl(graph)) {
        throw out_of_range(dal::detail::error_messages::
                               vertex_index_out_of_range_expect_from_zero_to_vertex_count());
    }
    return detail::get_vertex_degree_impl(graph, vertex);
}

/// Returns the range of the vertex neighbors for the specified vertex
///
/// @tparam Graph  Type of the graph
///
/// @param [in]   graph  Input graph object
///
/// @param [in]   vertex Identifier of the vertex
///
/// @return The range of the vertex neighbors
template <typename Graph>
constexpr auto get_vertex_neighbors(const Graph &graph, vertex_type<Graph> vertex)
    -> const_edge_range_type<Graph> {
    if (vertex < 0 || (vertex_size_type<Graph>)vertex >= detail::get_vertex_count_impl(graph)) {
        throw out_of_range(dal::detail::error_messages::
                               vertex_index_out_of_range_expect_from_zero_to_vertex_count());
    }
    return detail::get_vertex_neighbors_impl(graph, vertex);
}

} // namespace oneapi::dal::preview
