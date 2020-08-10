/* file: graph.hpp */
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

/*
//++
//  Graph types and service functionality
//--
*/

#pragma once

#include <cstdint>
#include <memory>

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/data/array.hpp"
#include "oneapi/dal/data/detail/graph_container.hpp"
#include "oneapi/dal/data/undirected_adjacency_array_graph.hpp"
#include "oneapi/dal/detail/common.hpp"

/**
 * \brief Contains graph functionality preview as an experimental part of oneapi dal.
 */
namespace oneapi::dal::preview {

template <typename G>
using vertex_user_value_type = typename G::vertex_user_value_type;

template <typename G>
using edge_user_value_type = typename G::edge_user_value_type;

template <typename G>
using vertex_type = typename G::vertex_type;

template <typename G>
using vertex_size_type = typename G::vertex_size_type;

template <typename G>
using edge_size_type = typename G::edge_size_type;

template <typename G>
using vertex_edge_size_type = typename G::vertex_edge_size_type;

template <typename G>
using edge_iterator_type = typename G::edge_iterator;

template <typename G>
using const_edge_iterator_type = typename G::const_edge_iterator;

template <typename G>
using vertex_edge_iterator_type = typename G::vertex_edge_iterator;

template <typename G>
using const_vertex_edge_iterator_type = typename G::const_vertex_edge_iterator;

template <typename G>
using edge_range_type = typename G::edge_range;

template <typename G>
using const_edge_range_type = typename G::const_edge_range;

template <typename G>
using vertex_edge_range_type = typename G::vertex_edge_range;

template <typename G>
using const_vertex_edge_range_type = typename G::const_vertex_edge_range;

/// Get number of vertexes in a graph.
///
/// @tparam G  Type of graph
///
/// @param [in]   g  graph
///
/// @return Result is vertex_size_type of G.
template <typename G>
constexpr auto get_vertex_count(const G &g) noexcept -> vertex_size_type<G> {
    return get_vertex_count_impl(g);
}

template <typename G>
constexpr auto get_edge_count(const G &g) noexcept -> edge_size_type<G> {
    return get_edge_count_impl(g);
}

/// Get degree of vertex in a graph.
///
/// @tparam G  Type of graph
///
/// @param [in]   g  graph
///
/// @param [in]   vertex  vertex which degree to compute
///
/// @return Result is vertex_type of G.
template <typename G>
constexpr auto get_vertex_degree(const G &g, const vertex_type<G> &vertex)
    -> vertex_edge_size_type<G> {
    return get_vertex_degree_impl(g, vertex);
}

template <typename G>
constexpr auto get_vertex_neighbors(const G &g, const vertex_type<G> &vertex)
    -> const_vertex_edge_range_type<G> {
    return get_vertex_neighbors_impl(g, vertex);
}

template <typename G>
constexpr auto get_vertex_value(const G &g, const vertex_type<G> &vertex)
    -> vertex_user_value_type<G> {
    return get_vertex_value(g, vertex);
}

template <typename IndexType = std::int32_t>
using edge_list = detail::graph_container<std::pair<IndexType, IndexType>>;

template <typename G>
void convert_to_csr(const edge_list<vertex_type<G>> &edges, G &graph) {
    convert_to_csr_impl(edges, graph);
}

} // namespace oneapi::dal::preview