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
#include "oneapi/dal/graph/detail/graph_service_functions_impl.hpp"
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

    static_assert(std::is_integral_v<IndexType> && std::is_signed_v<IndexType>,
                  "Use signed integer for vertex index type");

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
    undirected_adjacency_array_graph(const GraphValue &graph_user_value,
                                     Allocator allocator = Allocator()){};

    /// Constructs an empty undirected_adjacency_array_graph with move graph properties and
    /// allocator
    undirected_adjacency_array_graph(graph_user_value_type<graph_type> &&graph_user_value,
                                     Allocator allocator = Allocator()){};

    /// Copy operator for undirected_adjacency_array_graph
    undirected_adjacency_array_graph &operator=(const undirected_adjacency_array_graph &graph);

    /// Move operator for undirected_adjacency_array_graph
    undirected_adjacency_array_graph &operator=(undirected_adjacency_array_graph &&graph);

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
    undirected_adjacency_array_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>> {
    using graph_type =
        undirected_adjacency_array_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>;
    using impl_type = detail::undirected_adjacency_array_graph_impl<VertexValue,
                                                                    EdgeValue,
                                                                    GraphValue,
                                                                    IndexType,
                                                                    Allocator>;
    using allocator_type = Allocator;

    // graph weight types
    using graph_user_value_type = GraphValue;
    using const_graph_user_value_type = const graph_user_value_type;

    // vertex types
    using vertex_type = IndexType;
    using const_vertex_type = const vertex_type;
    using vertex_allocator_type = typename impl_type::vertex_allocator_type;
    using vertex_set = typename impl_type::vertex_set;
    using vertex_iterator = typename impl_type::vertex_iterator;
    using const_vertex_iterator = typename impl_type::const_vertex_iterator;
    using vertex_size_type = typename impl_type::vertex_size_type;

    using vertex_key_type = vertex_type;
    using const_vertex_key_type = const vertex_key_type;

    // vertex weight types
    using vertex_user_value_type = VertexValue;
    using const_vertex_user_value_type = const vertex_user_value_type;
    using vertex_user_value_allocator_type = typename impl_type::vertex_user_value_allocator_type;
    using vertex_user_value_set = typename impl_type::vertex_user_value_set;

    // edge types
    using edge_type = IndexType;
    using edge_allocator_type = typename impl_type::edge_allocator_type;
    using edge_set = typename impl_type::edge_set;
    using edge_iterator = typename impl_type::edge_iterator;
    using const_edge_iterator = typename impl_type::const_edge_iterator;
    using edge_size_type = typename impl_type::edge_size_type;

    // edge weight types
    using edge_user_value_type = EdgeValue;
    using edge_user_value_allocator_type = typename impl_type::edge_user_value_allocator_type;
    using edge_user_value_set = typename impl_type::edge_user_value_set;

    using edge_key_type = std::pair<vertex_key_type, vertex_key_type>;
    using edge_value_type = std::pair<edge_key_type, edge_user_value_type>;

    using edge_index = IndexType;

    // ranges
    using edge_range = range<edge_iterator>;
    using const_edge_range = range<const_edge_iterator>;
};

template <typename VertexValue,
          typename EdgeValue,
          typename GraphValue,
          typename IndexType,
          typename Allocator>
undirected_adjacency_array_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>::
    undirected_adjacency_array_graph()
        : impl_(new detail::undirected_adjacency_array_graph_impl<VertexValue,
                                                                  EdgeValue,
                                                                  GraphValue,
                                                                  IndexType,
                                                                  Allocator>) {}

template <typename VertexValue,
          typename EdgeValue,
          typename GraphValue,
          typename IndexType,
          typename Allocator>
undirected_adjacency_array_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>::
    undirected_adjacency_array_graph(const undirected_adjacency_array_graph &graph)
        : undirected_adjacency_array_graph() {
    const auto &layout = dal::detail::get_impl(graph).get_topology();

    impl_->get_topology()._vertex_count = layout._vertex_count;
    impl_->get_topology()._edge_count = layout._edge_count;

    impl_->get_topology()._vertex_neighbors = layout._vertex_neighbors;
    impl_->get_topology()._edge_offsets = layout._edge_offsets;
    impl_->get_topology()._degrees = layout._degrees;

    //this->impl_->_vertex_value = layout._vertex_value;
    //this->impl_->_edge_value = layout._edge_value;
}

template <typename VertexValue,
          typename EdgeValue,
          typename GraphValue,
          typename IndexType,
          typename Allocator>
undirected_adjacency_array_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>::
    undirected_adjacency_array_graph(undirected_adjacency_array_graph &&graph)
        : undirected_adjacency_array_graph() {
    auto &layout = dal::detail::get_impl(graph).get_topology();

    impl_->get_topology()._vertex_count = layout._vertex_count;
    layout._vertex_count = 0;

    impl_->get_topology()._edge_count = layout._edge_count;
    layout._edge_count = 0;

    impl_->get_topology()._vertex_neighbors = std::move(layout._vertex_neighbors);
    impl_->get_topology()._edge_offsets = std::move(layout._edge_offsets);
    impl_->get_topology()._degrees = std::move(layout._degrees);

    //this->impl_->_vertex_value = std::move(layout._vertex_value);
    //this->impl_->_edge_value = std::move(layout._edge_value);
}

template <typename VertexValue,
          typename EdgeValue,
          typename GraphValue,
          typename IndexType,
          typename Allocator>
undirected_adjacency_array_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>
    &undirected_adjacency_array_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>::
    operator=(const undirected_adjacency_array_graph &graph) {
    if (&graph != this) {
        const auto &layout = dal::detail::get_impl(graph).get_topology();

        impl_->get_topology()._vertex_count = layout._vertex_count;
        impl_->get_topology()._edge_count = layout._edge_count;

        impl_->get_topology()._vertex_neighbors = layout._vertex_neighbors;
        impl_->get_topology()._edge_offsets = layout._edge_offsets;
        impl_->get_topology()._degrees = layout._degrees;

        //this->impl_->_vertex_value = layout._vertex_value;
        //this->impl_->_edge_value = layout._edge_value;
    }
    return *this;
}

template <typename VertexValue,
          typename EdgeValue,
          typename GraphValue,
          typename IndexType,
          typename Allocator>
undirected_adjacency_array_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>
    &undirected_adjacency_array_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>::
    operator=(undirected_adjacency_array_graph &&graph) {
    if (&graph != this) {
        auto &layout = dal::detail::get_impl(graph).get_topology();

        impl_->get_topology()._vertex_count = layout._vertex_count;
        layout._vertex_count = 0;

        impl_->get_topology()._edge_count = layout._edge_count;
        layout._edge_count = 0;

        impl_->get_topology()._vertex_neighbors = std::move(layout._vertex_neighbors);
        impl_->get_topology()._edge_offsets = std::move(layout._edge_offsets);
        impl_->get_topology()._degrees = std::move(layout._degrees);

        //this->impl_->_vertex_value = std::move(layout._vertex_value);
        //this->impl_->_edge_value = std::move(layout._edge_value);
    }
    return *this;
}

namespace detail {

template <typename Graph>
ONEDAL_EXPORT auto get_vertex_count_impl(const Graph &graph) noexcept -> vertex_size_type<Graph> {
    const auto &layout = dal::detail::get_impl(graph).get_topology();
    return get_topology_vertex_count(layout);
}

template <typename Graph>
ONEDAL_EXPORT auto get_edge_count_impl(const Graph &graph) noexcept -> edge_size_type<Graph> {
    const auto &layout = dal::detail::get_impl(graph).get_topology();
    return get_topology_edge_count(layout);
    layout._edge_count;
}

template <typename Graph>
ONEDAL_EXPORT auto get_vertex_degree_impl(const Graph &graph,
                                          const vertex_type<Graph> &vertex) noexcept
    -> edge_size_type<Graph> {
    const auto &layout = dal::detail::get_impl(graph).get_topology();
    return get_topology_vertex_degree(layout, vertex);
}

template <typename Graph>
ONEDAL_EXPORT auto get_vertex_neighbors_impl(const Graph &graph,
                                             const vertex_type<Graph> &vertex) noexcept
    -> const_edge_range_type<Graph> {
    const auto &layout = dal::detail::get_impl(graph).get_topology();
    return get_topology_vertex_neighbors(layout, vertex);
}
} //namespace detail

} // namespace oneapi::dal::preview
