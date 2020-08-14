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

#include "oneapi/dal/data/undirected_adjacency_array_graph.hpp"
#include "daal/src/threading/threading.h"
#include "services/daal_atomic_int.h"

namespace oneapi::dal::preview {

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
    undirected_adjacency_array_graph(const undirected_adjacency_array_graph& graph)
        : undirected_adjacency_array_graph() {
    const auto& layout = detail::get_impl(graph);

    impl_->_vertex_count = layout->_vertex_count;
    impl_->_edge_count   = layout->_edge_count;

    impl_->_vertex_neighbors = layout->_vertex_neighbors;
    impl_->_edge_offsets     = layout->_edge_offsets;
    impl_->_degrees          = layout->_degrees;

    impl_->_vertex_value = layout->_vertex_value;
    impl_->_edge_value   = layout->_edge_value;
}

template <typename VertexValue,
          typename EdgeValue,
          typename GraphValue,
          typename IndexType,
          typename Allocator>
undirected_adjacency_array_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>::
    undirected_adjacency_array_graph(undirected_adjacency_array_graph&& graph)
        : undirected_adjacency_array_graph() {
    auto& layout = detail::get_impl(graph);

    impl_->_vertex_count  = layout->_vertex_count;
    layout->_vertex_count = 0;

    impl_->_edge_count  = layout->_edge_count;
    layout->_edge_count = 0;

    impl_->_vertex_neighbors = std::move(layout->_vertex_neighbors);
    impl_->_edge_offsets     = std::move(layout->_edge_offsets);
    impl_->_degrees          = std::move(layout->_degrees);

    impl_->_vertex_value = std::move(layout->_vertex_value);
    impl_->_edge_value   = std::move(layout->_edge_value);
}

template <typename VertexValue,
          typename EdgeValue,
          typename GraphValue,
          typename IndexType,
          typename Allocator>
undirected_adjacency_array_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>&
undirected_adjacency_array_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>::
operator=(const undirected_adjacency_array_graph& graph) {
    if (&graph != this) {
        const auto& layout = detail::get_impl(graph);

        impl_->_vertex_count = layout->_vertex_count;
        impl_->_edge_count   = layout->_edge_count;

        impl_->_vertex_neighbors = layout->_vertex_neighbors;
        impl_->_edge_offsets     = layout->_edge_offsets;
        impl_->_degrees          = layout->_degrees;

        impl_->_vertex_value = layout->_vertex_value;
        impl_->_edge_value   = layout->_edge_value;
    }
    return *this;
}

template <typename VertexValue,
          typename EdgeValue,
          typename GraphValue,
          typename IndexType,
          typename Allocator>
undirected_adjacency_array_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>&
undirected_adjacency_array_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>::
operator=(undirected_adjacency_array_graph&& graph) {
    if (&graph != this) {
        auto& layout = detail::get_impl(graph);

        impl_->_vertex_count  = layout->_vertex_count;
        layout->_vertex_count = 0;

        impl_->_edge_count  = layout->_edge_count;
        layout->_edge_count = 0;

        impl_->_vertex_neighbors = std::move(layout->_vertex_neighbors);
        impl_->_edge_offsets     = std::move(layout->_edge_offsets);
        impl_->_degrees          = std::move(layout->_degrees);

        impl_->_vertex_value = std::move(layout->_vertex_value);
        impl_->_edge_value   = std::move(layout->_edge_value);
    }
    return *this;
}

template class ONEAPI_DAL_EXPORT undirected_adjacency_array_graph<empty_value,
                                                                  empty_value,
                                                                  empty_value,
                                                                  std::int32_t,
                                                                  std::allocator<char>>;

using graph_default = undirected_adjacency_array_graph<empty_value,
                                                       empty_value,
                                                       empty_value,
                                                       std::int32_t,
                                                       std::allocator<char>>;

namespace detail {
template ONEAPI_DAL_EXPORT auto get_vertex_count_impl<graph_default>(
    const graph_default& g) noexcept -> vertex_size_type<graph_default>;

} // namespace detail
} // namespace oneapi::dal::preview
