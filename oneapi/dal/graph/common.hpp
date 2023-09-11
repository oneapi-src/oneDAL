/* file: common.hpp */
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
#include "oneapi/dal/graph/detail/common.hpp"
#include "oneapi/dal/graph/detail/container.hpp"

namespace oneapi::dal::preview {

template <typename Graph>
struct graph_traits {
    using graph_type = empty_value;
    using allocator_type = empty_value;

    // graph weight types
    using graph_user_value_type = empty_value;
    using const_graph_user_value_type = empty_value;

    // vertex types
    using vertex_type = empty_value;
    using vertex_iterator = empty_value;
    using const_vertex_iterator = empty_value;
    using vertex_size_type = empty_value;

    using vertex_key_type = empty_value;
    using const_vertex_key_type = empty_value;

    // vertex weight types
    using vertex_user_value_type = empty_value;

    // vertex edge types
    using vertex_edge_type = empty_value;
    using vertex_edge_iterator = empty_value;
    using const_vertex_edge_iterator = empty_value;
    using vertex_edge_size_type = empty_value;

    // edge types
    using edge_type = empty_value;
    using edge_iterator = empty_value;
    using const_edge_iterator = empty_value;
    using edge_size_type = empty_value;

    // edge weight types
    using edge_user_value_type = empty_value;

    using edge_key_type = empty_value;
    using edge_value_type = empty_value;
    using edge_index = empty_value;

    // ranges
    using vertex_edge_range = empty_value;
    using const_vertex_edge_range = empty_value;
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

/// Type of the graph vertex-edge size
/// @tparam Graph Type of the graph
template <typename Graph>
using vertex_edge_size_type = typename graph_traits<Graph>::vertex_edge_size_type;

/// Type of the graph vertex-edge size
/// @tparam Graph Type of the graph
template <typename Graph>
using vertex_outward_edge_size_type = typename graph_traits<Graph>::vertex_outward_edge_size_type;

/// Type of the graph vertex-edge iterator
/// @tparam Graph Type of the graph
template <typename Graph>
using vertex_edge_iterator_type = typename graph_traits<Graph>::vertex_edge_iterator;

/// Type of the constant graph vertex-edge iterator
/// @tparam Graph Type of the graph
template <typename Graph>
using const_vertex_edge_iterator_type = typename graph_traits<Graph>::const_vertex_edge_iterator;

/// Type of the graph range of the vertex-edges
/// @tparam Graph Type of the graph
template <typename Graph>
using vertex_edge_range_type = typename graph_traits<Graph>::vertex_edge_range;

/// Type of the constant graph range of the vertex-edges
/// @tparam Graph Type of the graph
template <typename Graph>
using const_vertex_edge_range_type = typename graph_traits<Graph>::const_vertex_edge_range;

/// Type of the constant graph range of the vertex-edges
/// @tparam Graph Type of the graph
template <typename Graph>
using const_vertex_outward_edge_range_type =
    typename graph_traits<Graph>::const_vertex_outward_edge_range;

/// Type of the graph allocator
/// @tparam Graph Type of the graph
template <typename Graph>
using graph_allocator = typename graph_traits<Graph>::allocator_type;

template <typename Graph>
constexpr bool is_directed = false;

} // namespace oneapi::dal::preview
