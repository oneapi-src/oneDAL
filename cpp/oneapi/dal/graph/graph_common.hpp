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

/// Type of the graph vertex properties
/// @tparam Graph Type of the graph
template <typename Graph>
using vertex_user_value_type = typename Graph::vertex_user_value_type;

/// Type of the graph edge properties
/// @tparam Graph Type of the graph
template <typename Graph>
using edge_user_value_type = typename Graph::edge_user_value_type;

/// Type of the graph vertex indices
/// @tparam Graph Type of the graph
template <typename Graph>
using vertex_type = typename Graph::vertex_type;

/// Type of the graph vertex set size
/// @tparam Graph Type of the graph
template <typename Graph>
using vertex_size_type = typename Graph::vertex_size_type;

/// Type of the graph edge set size
/// @tparam Graph Type of the graph
template <typename Graph>
using edge_size_type = typename Graph::edge_size_type;

/// Type of the graph edge iterator
/// @tparam Graph Type of the graph
template <typename Graph>
using edge_iterator_type = typename Graph::edge_iterator;

/// Type of the constant graph edge iterator
/// @tparam Graph Type of the graph
template <typename Graph>
using const_edge_iterator_type = typename Graph::const_edge_iterator;

/// Type of the graph range of the edges
/// @tparam Graph Type of the graph
template <typename Graph>
using edge_range_type = typename Graph::edge_range;

/// Type of the constant graph range of the edges
/// @tparam Graph Type of the graph
template <typename Graph>
using const_edge_range_type = typename Graph::const_edge_range;

/// Type of the graph allocator
/// @tparam Graph Type of the graph
template <typename Graph>
using graph_allocator = typename Graph::allocator_type;

/// Type of graph representation as an edge list
/// @tparam IndexType Type of the graph vertex indicies
template <typename IndexType = std::int32_t>
using edge_list = detail::graph_container<std::pair<IndexType, IndexType>>;

} // namespace oneapi::dal::preview
