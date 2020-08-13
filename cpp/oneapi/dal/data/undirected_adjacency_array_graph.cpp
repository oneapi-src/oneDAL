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
    undirected_adjacency_array_graph(undirected_adjacency_array_graph const &graph){
        impl_ = new detail::undirected_adjacency_array_graph_impl<VertexValue,
                                                                  EdgeValue,
                                                                  GraphValue,
                                                                  IndexType,
                                                                  Allocator>;

        const auto &layout = detail::get_impl(graph);

        impl_->_vertex_count = layout->_vertex_count;
        impl_->_edge_count = layout->_edge_count;

        impl_->_vertexes = layout->_vertexes;
        impl_->_edges = layout->_edges;

        impl_->_vertex_value = layout->_vertex_value;
        impl_->_edge_value = layout->_edge_value;
}

template <typename VertexValue,
          typename EdgeValue,
          typename GraphValue,
          typename IndexType,
          typename Allocator>
undirected_adjacency_array_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>::
    undirected_adjacency_array_graph(undirected_adjacency_array_graph &&graph){
        impl_ = new detail::undirected_adjacency_array_graph_impl<VertexValue,
                                                                  EdgeValue,
                                                                  GraphValue,
                                                                  IndexType,
                                                                  Allocator>;

        auto &layout = detail::get_impl(graph);

        impl_->_vertex_count = layout->_vertex_count;
        layout->_vertex_count = 0;

        impl_->_edge_count = layout->_edge_count;
        layout->_edge_count = 0;

        impl_->_vertexes = std::move(layout->_vertexes);
        impl_->_edges = std::move(layout->_edges);

        impl_->_vertex_value = std::move(layout->_vertex_value);
        impl_->_edge_value = std::move(layout->_edge_value);
}

template <typename VertexValue,
          typename EdgeValue,
          typename GraphValue,
          typename IndexType,
          typename Allocator>
undirected_adjacency_array_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>&
    undirected_adjacency_array_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>::operator=(undirected_adjacency_array_graph const &graph){
        if (&graph != this) {
            const auto &layout = detail::get_impl(graph);

            impl_->_vertex_count = layout->_vertex_count;
            impl_->_edge_count = layout->_edge_count;

            impl_->_vertexes = layout->_vertexes;
           impl_->_edges = layout->_edges;

            impl_->_vertex_value = layout->_vertex_value;
            impl_->_edge_value = layout->_edge_value;
        }
        return *this;
}

template <typename VertexValue,
          typename EdgeValue,
          typename GraphValue,
          typename IndexType,
          typename Allocator>
undirected_adjacency_array_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>&
    undirected_adjacency_array_graph<VertexValue, EdgeValue, GraphValue, IndexType, Allocator>::operator=(undirected_adjacency_array_graph &&graph){
        if (&graph != this) {
            auto &layout = detail::get_impl(graph);

            impl_->_vertex_count = layout->_vertex_count;
            layout->_vertex_count = 0;

            impl_->_edge_count = layout->_edge_count;
            layout->_edge_count = 0;

            impl_->_vertexes = std::move(layout->_vertexes);
            impl_->_edges = std::move(layout->_edges);

            impl_->_vertex_value = std::move(layout->_vertex_value);
            impl_->_edge_value = std::move(layout->_edge_value);
        }
        return *this;
}

template class ONEAPI_DAL_EXPORT undirected_adjacency_array_graph<empty_value,
                                                                  empty_value,
                                                                  empty_value,
                                                                  std::int32_t,
                                                                  std::allocator<empty_value>>;

using graph32 = undirected_adjacency_array_graph<empty_value,
                                                 empty_value,
                                                 empty_value,
                                                 std::int32_t,
                                                 std::allocator<empty_value>>;

template <typename G>
ONEAPI_DAL_EXPORT auto get_vertex_count_impl(const G &g) noexcept -> vertex_size_type<G> {
    const auto &layout = detail::get_impl(g);
    return layout->_vertex_count;
}

template ONEAPI_DAL_EXPORT auto get_vertex_count_impl<graph32>(const graph32 &g) noexcept
    -> vertex_size_type<graph32>;

template <typename G>
ONEAPI_DAL_EXPORT auto get_edge_count_impl(const G &g) noexcept -> edge_size_type<G> {
    const auto &layout = detail::get_impl(g);
    return layout->_edge_count;
}

template ONEAPI_DAL_EXPORT edge_size_type<graph32> get_edge_count_impl(const graph32 &g);

template <typename G>
ONEAPI_DAL_EXPORT auto get_vertex_degree_impl(const G &g, const vertex_type<G> &vertex) noexcept
    -> vertex_edge_size_type<G> {
    const auto &layout = detail::get_impl(g);
    return layout->_vertexes[vertex + 1] - layout->_vertexes[vertex];
}

template ONEAPI_DAL_EXPORT vertex_edge_size_type<graph32> get_vertex_degree_impl(
    const graph32 &g,
    const vertex_type<graph32> &vertex);

template <typename G>
ONEAPI_DAL_EXPORT auto get_vertex_neighbors_impl(const G &g, const vertex_type<G> &vertex) noexcept
    -> const_vertex_edge_range_type<G> {
    const auto &layout = detail::get_impl(g);
    const_vertex_edge_iterator_type<G> vertex_neighbors_begin =
        layout->_edges.begin() + layout->_vertexes[vertex];
    const_vertex_edge_iterator_type<G> vertex_neighbors_end =
        layout->_edges.begin() + layout->_vertexes[vertex + 1];
    return std::make_pair(vertex_neighbors_begin, vertex_neighbors_end);
}

template ONEAPI_DAL_EXPORT const_vertex_edge_range_type<graph32> get_vertex_neighbors_impl(
    const graph32 &g,
    const vertex_type<graph32> &vertex);

template <typename G>
ONEAPI_DAL_EXPORT void convert_to_csr_impl(const edge_list<vertex_type<G>> &edges, G &g) {
    auto layout    = detail::get_impl(g);
    using int_t    = typename G::vertex_size_type;
    using vertex_t = typename G::vertex_type;

    layout->_vertex_count = 0;
    layout->_edge_count   = 0;

    vertex_t max_id = 0;

    for (auto edge : edges) {
        max_id = std::max(max_id, std::max(edge.first, edge.second));
        layout->_edge_count += 1;
    }

    layout->_vertex_count = max_id + 1;
    int_t *degrees        = (int_t *)malloc(layout->_vertex_count * sizeof(int_t));
    for (int_t u = 0; u < layout->_vertex_count; ++u) {
        degrees[u] = 0;
    }

    for (auto edge : edges) {
        degrees[edge.first]++;
        degrees[edge.second]++;
    }

    layout->_vertexes.resize(layout->_vertex_count + 1);
    auto _rows              = layout->_vertexes.data();
    int_t total_sum_degrees = 0;
    _rows[0]                = total_sum_degrees;

    for (int_t i = 0; i < layout->_vertex_count; ++i) {
        total_sum_degrees += degrees[i];
        _rows[i + 1] = total_sum_degrees;
    }

    free(degrees);
    layout->_edges.resize(_rows[layout->_vertex_count] + 1);
    auto _cols = layout->_edges.data();
    auto offests(layout->_vertexes);

    for (auto edge : edges) {
        _cols[offests[edge.first]++]  = edge.second;
        _cols[offests[edge.second]++] = edge.first;
    }
}

template ONEAPI_DAL_EXPORT void convert_to_csr_impl(const edge_list<vertex_type<graph32>> &edges,
                                                    graph32 &g);

} // namespace oneapi::dal::preview
