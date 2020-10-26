/* file: graph_common.hpp */
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
/// Graph related common data type aliases

#pragma once

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/graph/detail/graph_container.hpp"

namespace oneapi::dal::preview {

template <typename Graph>
struct ONEDAL_EXPORT graph_traits {
    using graph_type = Graph;
    using allocator_type = typename Graph::allocator_type;

    // graph weight types
    using graph_user_value_type = typename Graph::graph_value;
    using const_graph_user_value_type = const graph_user_value_type;

    // vertex types
    using vertex_type = typename Graph::index_type;
    using const_vertex_type = const vertex_type;
    using vertex_iterator = typename Graph::vertex_iterator;
    using const_vertex_iterator = typename Graph::const_vertex_iterator;
    using vertex_size_type = typename Graph::vertex_size_type;

    using vertex_key_type = vertex_type;
    using const_vertex_key_type = const vertex_key_type;

    // vertex weight types
    using vertex_user_value_type = typename Graph::vertex_value;
    using const_vertex_user_value_type = const vertex_user_value_type;

    // edge types
    using edge_type = typename Graph::index_type;
    using edge_iterator = typename Graph::edge_iterator;
    using const_edge_iterator = typename Graph::const_edge_iterator;
    using edge_size_type = typename Graph::edge_size_type;

    // edge weight types
    using edge_user_value_type = typename Graph::edge_value;

    using edge_key_type = std::pair<vertex_key_type, vertex_key_type>;
    using edge_value_type = std::pair<edge_key_type, edge_user_value_type>;
    using edge_index = typename Graph::index_type;

    // ranges
    using edge_range = range<edge_iterator>;
    using const_edge_range = range<const_edge_iterator>;
};

/// Type of the graph properties
/// @tparam Graph Type of the graph
template <typename Graph>
using graph_user_value_type = typename graph_traits<Graph>::graph_user_value_type;

/// Type of the graph vertex properties
/// @tparam Graph Type of the graph
template <typename Graph>
using vertex_user_value_type = typename graph_traits<Graph>::vertex_user_value_type;

/// Type of the graph edge properties
/// @tparam Graph Type of the graph
template <typename Graph>
using edge_user_value_type = typename graph_traits<Graph>::edge_user_value_type;

/// Type of the graph vertex indices
/// @tparam Graph Type of the graph
template <typename Graph>
using vertex_type = typename graph_traits<Graph>::vertex_type;

/// Type of the graph vertex set size
/// @tparam Graph Type of the graph
template <typename Graph>
using vertex_size_type = typename graph_traits<Graph>::vertex_size_type;

/// Type of the graph edge set size
/// @tparam Graph Type of the graph
template <typename Graph>
using edge_size_type = typename graph_traits<Graph>::edge_size_type;

/// Type of the graph edge iterator
/// @tparam Graph Type of the graph
template <typename Graph>
using edge_iterator_type = typename graph_traits<Graph>::edge_iterator;

/// Type of the constant graph edge iterator
/// @tparam Graph Type of the graph
template <typename Graph>
using const_edge_iterator_type = typename graph_traits<Graph>::const_edge_iterator;

/// Type of the graph range of the edges
/// @tparam Graph Type of the graph
template <typename Graph>
using edge_range_type = typename graph_traits<Graph>::edge_range;

/// Type of the constant graph range of the edges
/// @tparam Graph Type of the graph
template <typename Graph>
using const_edge_range_type = typename graph_traits<Graph>::const_edge_range;

/// Type of the graph allocator
/// @tparam Graph Type of the graph
template <typename Graph>
using graph_allocator = typename graph_traits<Graph>::allocator_type;

/// Type of graph representation as an edge list
/// @tparam IndexType Type of the graph vertex indicies
template <typename IndexType = std::int32_t>
using edge_list = detail::graph_container<std::pair<IndexType, IndexType>>;

} // namespace oneapi::dal::preview
