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

template class undirected_adjacency_array_graph<empty_value,
                                                empty_value,
                                                empty_value,
                                                std::int64_t,
                                                std::allocator<empty_value>>;

template class undirected_adjacency_array_graph<empty_value,
                                                empty_value,
                                                empty_value,
                                                std::int32_t,
                                                std::allocator<empty_value>>;

using graph64 = undirected_adjacency_array_graph<empty_value,
                                                 empty_value,
                                                 empty_value,
                                                 std::int64_t,
                                                 std::allocator<empty_value>>;

using graph32 = undirected_adjacency_array_graph<empty_value,
                                                 empty_value,
                                                 empty_value,
                                                 std::int32_t,
                                                 std::allocator<empty_value>>;

template <typename G>
auto get_vertex_count_impl(const G &g) noexcept -> vertex_size_type<G> {
    const auto &layout = detail::get_impl(g);
    return layout->_vertex_count;
}
template auto get_vertex_count_impl<graph64>(const graph64 &g) noexcept
    -> vertex_size_type<graph64>;
template auto get_vertex_count_impl<graph32>(const graph32 &g) noexcept
    -> vertex_size_type<graph32>;

template <typename G>
auto get_edge_count_impl(const G &g) noexcept -> edge_size_type<G> {
    const auto &layout = detail::get_impl(g);
    return layout->_edge_count;
}

template edge_size_type<graph64> get_edge_count_impl(const graph64 &g);
template edge_size_type<graph32> get_edge_count_impl(const graph32 &g);

template <typename G>
auto get_vertex_degree_impl(const G &g, const vertex_type<G> &vertex) -> vertex_edge_size_type<G> {
    const auto &layout = detail::get_impl(g);
    return layout->_degrees[vertex];
}

template vertex_edge_size_type<graph64> get_vertex_degree_impl(const graph64 &g,
                                                               const vertex_type<graph64> &vertex);
template vertex_edge_size_type<graph32> get_vertex_degree_impl(const graph32 &g,
                                                               const vertex_type<graph32> &vertex);

template <typename G>
auto get_vertex_neighbors_impl(const G &g, const vertex_type<G> &vertex)
    -> const_vertex_edge_range_type<G> {
    const auto &layout = detail::get_impl(g);
    const_vertex_edge_iterator_type<G> vertex_neighbors_begin(
        &layout->_edge_offsets[layout->_vertex_neighbors[vertex]]);
    const_vertex_edge_iterator_type<G> vertex_neighbors_end(
        &layout->_edge_offsets[layout->_vertex_neighbors[vertex + 1]]);
    auto neighbors_range = std::make_pair(vertex_neighbors_begin, vertex_neighbors_end);
    return neighbors_range;
}

template const_vertex_edge_range_type<graph64> get_vertex_neighbors_impl(
    const graph64 &g,
    const vertex_type<graph64> &vertex);

template const_vertex_edge_range_type<graph32> get_vertex_neighbors_impl(
    const graph32 &g,
    const vertex_type<graph32> &vertex);

template <typename G>
auto get_vertex_value_impl(const G &g, const vertex_type<G> &vertex) -> vertex_user_value_type<G> {
    const auto &layout = detail::get_impl(g);
    return layout->_vertex_value[vertex];
}

template vertex_user_value_type<graph64> get_vertex_value_impl(const graph64 &g,
                                                               const vertex_type<graph64> &vertex);

template vertex_user_value_type<graph32> get_vertex_value_impl(const graph32 &g,
                                                               const vertex_type<graph32> &vertex);

template <typename G>
void convert_to_csr_impl(const edge_list<vertex_type<G>> &edges, G &g) {
    auto layout    = detail::get_impl(g);
    using int_t    = typename G::vertex_size_type;
    using vertex_t = typename G::vertex_type;
    using edge_t   = typename G::edge_type;

    layout->_vertex_count = 0;
    layout->_edge_count   = 0;

    vertex_t max_id = 0;

    for (auto edge : edges) {
        max_id = std::max(max_id, std::max(edge.first, edge.second));
        layout->_edge_count += 1;
    }

    layout->_vertex_count = max_id + 1;
    layout->_degrees      = array<vertex_t>::zeros(layout->_vertex_count);

    for (auto edge : edges) {
        layout->_degrees[edge.first]++;
        layout->_degrees[edge.second]++;
    }

    layout->_vertex_neighbors = array<vertex_t>::empty(layout->_vertex_count + 1);
    auto _rows                = layout->_vertex_neighbors.get_mutable_data();
    int_t total_sum_degrees   = 0;
    _rows[0]                  = total_sum_degrees;

    for (std::int64_t i = 0; i < layout->_vertex_count; ++i) {
        total_sum_degrees += layout->_degrees[i];
        _rows[i + 1] = total_sum_degrees;
    }

    layout->_edge_offsets = array<edge_t>::empty(_rows[layout->_vertex_count] + 1);
    auto _cols            = layout->_edge_offsets.get_mutable_data();
    auto offests(layout->_vertex_neighbors);

    for (auto edge : edges) {
        _cols[offests[edge.first]++]  = edge.second;
        _cols[offests[edge.second]++] = edge.first;
    }
}

template void convert_to_csr_impl(const edge_list<vertex_type<graph64>> &edges, graph64 &g);

template void convert_to_csr_impl(const edge_list<vertex_type<graph32>> &edges, graph32 &g);

} // namespace oneapi::dal::preview
