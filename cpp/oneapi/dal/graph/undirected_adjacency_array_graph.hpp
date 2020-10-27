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
/// Contains the declaration of the undirected adjacency array graph

#pragma once

#include "oneapi/dal/graph/detail/graph_container.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_array_graph_impl.hpp"
#include "oneapi/dal/graph/graph_common.hpp"

namespace oneapi::dal::preview {

/// Class for a data management component responsible for representation of data
/// in the graph format. The class is designed to minimize storage requirements
/// and offer good performance characteristics. The graph is stored in a 0-based
/// CSR format with ordered vertex keys within each row. Self-loops and multi-edges
/// are not supported.
///
/// @tparam VertexValue  Type of vertex properties
/// @tparam EdgeValue    Type of edge properties
/// @tparam GraphValue   Type of graph properties
/// @tparam IndexType    Type of vertex indices
/// @tparam Allocator    Type of the custom allocator (currently not supported)
template <typename VertexValue = empty_value,
          typename EdgeValue = empty_value,
          typename GraphValue = empty_value,
          typename IndexType = std::int32_t,
          typename Allocator = std::allocator<char>>
class ONEDAL_EXPORT undirected_adjacency_array_graph {
public:
    using graph_type =
        undirected_adjacency_array_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>;
    using allocator_type = Allocator;

    // graph weight types
    using graph_user_value_type = GraphValue;
    using const_graph_user_value_type = const graph_user_value_type;

    // vertex types
    using vertex_type = IndexType;
    using const_vertex_type = const vertex_type;
    using vertex_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<vertex_type>;
    using vertex_set = detail::graph_container<vertex_type, vertex_allocator_type>;
    using vertex_iterator = typename vertex_set::iterator;
    using const_vertex_iterator = typename vertex_set::const_iterator;
    using vertex_size_type = typename vertex_set::size_type;

    using vertex_key_type = vertex_type;
    using const_vertex_key_type = const vertex_key_type;

    // vertex weight types
    using vertex_user_value_type = VertexValue;
    using const_vertex_user_value_type = const vertex_user_value_type;
    using vertex_user_value_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<vertex_user_value_type>;
    using vertex_user_value_set =
        detail::graph_container<vertex_user_value_type, vertex_user_value_allocator_type>;

    // edge types
    using edge_type = IndexType;
    using edge_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<edge_type>;
    using edge_set = detail::graph_container<edge_type, edge_allocator_type>;
    using edge_iterator = typename edge_set::iterator;
    using const_edge_iterator = typename edge_set::const_iterator;
    using edge_size_type = typename edge_set::size_type;

    // edge weight types
    using edge_user_value_type = EdgeValue;
    using edge_user_value_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<edge_user_value_type>;
    using edge_user_value_set =
        detail::graph_container<edge_user_value_type, edge_user_value_allocator_type>;

    using edge_key_type = std::pair<vertex_key_type, vertex_key_type>;
    using edge_value_type = std::pair<edge_key_type, edge_user_value_type>;
    using edge_index = IndexType;

    // ranges
    using edge_range = range<edge_iterator>;
    using const_edge_range = range<const_edge_iterator>;

    static_assert(std::is_integral_v<vertex_type> && std::is_signed_v<vertex_type> &&
                      sizeof(vertex_type) == 4,
                  "Use 32 bit signed integer for vertex index type");

    /// Constructs an empty undirected_adjacency_array_graph
    undirected_adjacency_array_graph();

    /// Constructs an empty undirected_adjacency_array_graph
    virtual ~undirected_adjacency_array_graph() = default;

    /// Move constructor for undirected_adjacency_array_graph
    undirected_adjacency_array_graph(undirected_adjacency_array_graph &&graph);

    /// Copy constructor for undirected_adjacency_array_graph
    undirected_adjacency_array_graph(const undirected_adjacency_array_graph &graph);

    /// Constructs an empty undirected_adjacency_array_graph with specified graph properties
    /// and allocator
    undirected_adjacency_array_graph(const graph_user_value_type &graph_user_value,
                                     allocator_type allocator = allocator_type()){};

    /// Constructs an empty undirected_adjacency_array_graph with move graph properties and
    /// allocator
    undirected_adjacency_array_graph(graph_user_value_type &&graph_user_value,
                                     allocator_type allocator = allocator_type()){};

    /// Copy operator for undirected_adjacency_array_graph
    undirected_adjacency_array_graph &operator=(const undirected_adjacency_array_graph &graph);

    /// Move operator for undirected_adjacency_array_graph
    undirected_adjacency_array_graph &operator=(undirected_adjacency_array_graph &&graph);
    using pimpl =
        oneapi::dal::detail::pimpl<detail::undirected_adjacency_array_graph_impl<VertexValue,
                                                                                 EdgeValue,
                                                                                 GraphValue,
                                                                                 IndexType,
                                                                                 Allocator>>;

private:
    pimpl impl_;
    friend pimpl &oneapi::dal::preview::detail::get_impl<graph_type>(graph_type &graph);

    friend const pimpl &oneapi::dal::preview::detail::get_impl<graph_type>(const graph_type &graph);
};
} // namespace oneapi::dal::preview
