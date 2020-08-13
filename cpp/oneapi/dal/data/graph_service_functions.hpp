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

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/data/detail/graph_container.hpp"
#include "oneapi/dal/data/graph_common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::preview {

/// Returns the number of vertices in the graph
///
/// @tparam Graph  Type of the graph
///
/// @param [in]   graph  Input graph object
///
/// @return The number of vertices in the graph
template <typename G>
constexpr auto get_vertex_count(const G &g) noexcept -> vertex_size_type<G> {
    return get_vertex_count_impl(g);
}

/// Returns the number of edges in the graph
///
/// @tparam Graph  Type of the graph
///
/// @param [in]   graph  Input graph object
///
/// @return The number of edges in the graph
template <typename G>
constexpr auto get_edge_count(const G &g) noexcept -> edge_size_type<G> {
    return get_edge_count_impl(g);
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
template <typename G>
constexpr auto get_vertex_degree(const G &g, const vertex_type<G> &vertex)
    -> edge_size_type<G> {
    if (vertex < 0 || (vertex_size_type<G>)vertex >= get_vertex_count_impl(g)) {
        throw out_of_range("Vertex index should be in [0, vertex_count)");
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
template <typename G>
constexpr auto get_vertex_neighbors(const G &g, const vertex_type<G> &vertex)
    -> const_edge_range_type<G> {
    if (vertex < 0 || (vertex_size_type<G>)vertex >= get_vertex_count_impl(g)) {
        throw out_of_range("Vertex index should be in [0, vertex_count)");
    }
    return detail::get_vertex_neighbors_impl(graph, vertex);
}

} // namespace oneapi::dal::preview
