/* file: directed_adjacency_vector_graph.hpp */
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
/// Contains the declaration of the undirected adjacency array graph

#pragma once

#include "oneapi/dal/graph/common.hpp"
#include "oneapi/dal/graph/detail/container.hpp"
#include "oneapi/dal/graph/detail/directed_adjacency_vector_graph_impl.hpp"

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
class ONEDAL_EXPORT directed_adjacency_vector_graph {
public:
    using graph_type =
        directed_adjacency_vector_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>;

    static_assert(detail::is_valid_index_v<IndexType>, "Use int32_t for vertex index type");
    static_assert(detail::is_valid_edge_value_v<EdgeValue>,
                  "Use empty_value, double or int32_t for edge value type");

    /// Constructs an empty directed_adjacency_vector_graph
    directed_adjacency_vector_graph();

    ~directed_adjacency_vector_graph() = default;

    /// Move constructor for directed_adjacency_vector_graph
    directed_adjacency_vector_graph(directed_adjacency_vector_graph &&other) = default;

    /// Constructs an empty directed_adjacency_vector_graph with specified graph properties
    /// and allocator
    directed_adjacency_vector_graph(const GraphValue &value, Allocator allocator = Allocator()){};

    /// Constructs an empty directed_adjacency_vector_graph with move graph properties and
    /// allocator
    directed_adjacency_vector_graph(GraphValue &&value, Allocator allocator = Allocator()){};

    /// Move operator for directed_adjacency_vector_graph
    directed_adjacency_vector_graph &operator=(directed_adjacency_vector_graph &&other);

private:
    using pimpl = dal::detail::pimpl<typename graph_traits<graph_type>::impl_type>;

    pimpl impl_;

    friend dal::detail::pimpl_accessor;
};

template <typename VertexValue,
          typename EdgeValue,
          typename GraphValue,
          typename IndexType,
          typename Allocator>
struct graph_traits<
    directed_adjacency_vector_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>> {
    using graph_type =
        directed_adjacency_vector_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>;
    using impl_type = detail::directed_adjacency_vector_graph_impl<VertexValue,
                                                                   EdgeValue,
                                                                   GraphValue,
                                                                   IndexType,
                                                                   Allocator>;
    using allocator_type = Allocator;

    // graph weight types
    using graph_user_value_type = typename impl_type::graph_user_value_type;
    using const_graph_user_value_type = typename impl_type::const_graph_user_value_type;

    // vertex types
    using vertex_type = typename impl_type::vertex_type;
    using vertex_set = typename impl_type::vertex_set;
    using vertex_iterator = typename impl_type::vertex_iterator;
    using const_vertex_iterator = typename impl_type::const_vertex_iterator;
    using vertex_size_type = typename impl_type::vertex_size_type;

    using vertex_key_type = vertex_type;
    using const_vertex_key_type = const vertex_key_type;

    // vertex edge types
    using vertex_outward_edge_type = typename impl_type::vertex_outward_edge_type;
    using vertex_outward_edge_size_type = typename impl_type::vertex_outward_edge_size_type;
    using vertex_outward_edge_set = typename impl_type::vertex_outward_edge_set;
    using vertex_outward_edge_iterator = typename impl_type::vertex_outward_edge_iterator;
    using const_vertex_outward_edge_iterator =
        typename impl_type::const_vertex_outward_edge_iterator;

    // vertex weight types
    using vertex_user_value_type = typename impl_type::vertex_user_value_type;
    using vertex_user_value_allocator_type = typename impl_type::vertex_user_value_allocator_type;
    using vertex_user_value_set = typename impl_type::vertex_user_value_set;

    // edge types
    using edge_type = typename impl_type::edge_type;
    using edge_set = typename impl_type::edge_set;
    using edge_size_type = typename impl_type::edge_size_type;

    // edge weight types
    using edge_user_value_type = typename impl_type::edge_user_value_type;
    using edge_user_value_allocator_type = typename impl_type::edge_user_value_allocator_type;
    using edge_user_value_set = typename impl_type::edge_user_value_set;

    using edge_key_type = std::pair<vertex_key_type, vertex_key_type>;
    using edge_value_type = std::pair<edge_key_type, edge_user_value_type>;

    using edge_index = IndexType;

    // ranges
    using vertex_outward_edge_range = typename impl_type::vertex_outward_edge_range;
    using const_vertex_outward_edge_range = typename impl_type::const_vertex_outward_edge_range;
    using vertex_edge_range = typename impl_type::vertex_edge_range;
    using const_vertex_edge_range = typename impl_type::const_vertex_edge_range;
    using const_edge_value_range_type = typename impl_type::const_edge_value_range_type;
};

template <typename VertexValue,
          typename EdgeValue,
          typename GraphValue,
          typename IndexType,
          typename Allocator>
directed_adjacency_vector_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>::
    directed_adjacency_vector_graph()
        : impl_(new detail::directed_adjacency_vector_graph_impl<VertexValue,
                                                                 EdgeValue,
                                                                 GraphValue,
                                                                 IndexType,
                                                                 Allocator>) {}

template <typename VertexValue,
          typename EdgeValue,
          typename GraphValue,
          typename IndexType,
          typename Allocator>
directed_adjacency_vector_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>
    &directed_adjacency_vector_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>::
    operator=(directed_adjacency_vector_graph &&other) {
    if (&other != this) {
        swap(*this, other);
    }
    return *this;
}

template <typename VertexValue,
          typename EdgeValue,
          typename GraphValue,
          typename IndexType,
          typename Allocator>
constexpr bool is_directed<
    directed_adjacency_vector_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>> =
    true;

} // namespace oneapi::dal::preview
