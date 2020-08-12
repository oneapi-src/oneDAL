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

#pragma once

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/data/detail/graph_container.hpp"
#include "oneapi/dal/data/graph_common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/exceptions.hpp"

/**
 * \brief Contains graph functionality preview as an experimental part of oneapi dal.
 */
namespace oneapi::dal::preview {

/// Get number of vertexes in a graph.
///
/// @tparam Graph  Type of graph
///
/// @param [in]   graph  Input graph
///
/// @return Result is vertex_size_type of Graph.
template <typename Graph>
constexpr auto get_vertex_count(const Graph& graph) noexcept -> vertex_size_type<Graph> {
    return detail::get_vertex_count_impl(graph);
}

template <typename Graph>
constexpr auto get_edge_count(const Graph& graph) noexcept -> edge_size_type<Graph> {
    return detail::get_edge_count_impl(graph);
}

/// Get degree of vertex in a graph.
///
/// @tparam Graph  Type of graph
///
/// @param [in]   graph  Input graph
///
/// @param [in]   vertex  vertex which degree to compute
///
/// @return Result is vertex_type of Graph.
template <typename Graph>
constexpr auto get_vertex_degree(const Graph& graph, const vertex_type<Graph>& vertex)
    -> vertex_edge_size_type<Graph> {
    if (vertex < 0 || (vertex_size_type<Graph>)vertex >= get_vertex_count_impl(graph)) {
        throw out_of_range("Vertex index should be in [0, vertex_count)");
    }
    return detail::get_vertex_degree_impl(graph, vertex);
}

template <typename Graph>
constexpr auto get_vertex_neighbors(const Graph& graph, const vertex_type<Graph>& vertex)
    -> const_vertex_edge_range_type<Graph> {
    if (vertex < 0 || (vertex_size_type<Graph>)vertex >= get_vertex_count_impl(graph)) {
        throw out_of_range("Vertex index should be in [0, vertex_count)");
    }
    return detail::get_vertex_neighbors_impl(graph, vertex);
}

} // namespace oneapi::dal::preview
