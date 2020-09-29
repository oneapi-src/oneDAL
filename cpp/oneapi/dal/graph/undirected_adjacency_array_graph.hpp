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
class ONEAPI_DAL_EXPORT undirected_adjacency_array_graph {
public:
    using graph_type =
        undirected_adjacency_array_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>;
    using vertex_value = VertexValue;
    using edge_value = EdgeValue;
    using graph_value = GraphValue;
    using index_type = IndexType;
    using allocator_type = Allocator;

    static_assert(std::is_integral_v<index_type> && std::is_signed_v<index_type> &&
                      sizeof(index_type) == 4,
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
    undirected_adjacency_array_graph(const graph_user_value_type<graph_type> &graph_user_value,
                                     allocator_type allocator = allocator_type()){};

    /// Constructs an empty undirected_adjacency_array_graph with move graph properties and
    /// allocator
    undirected_adjacency_array_graph(graph_user_value_type<graph_type> &&graph_user_value,
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
