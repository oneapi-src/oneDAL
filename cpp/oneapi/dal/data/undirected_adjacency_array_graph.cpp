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
    const_vertex_edge_iterator_type<G> vertex_neighbors_begin =
        layout->_vertex_neighbors.begin() + layout->_edge_offsets[vertex];
    const_vertex_edge_iterator_type<G> vertex_neighbors_end =
        layout->_vertex_neighbors.begin() + layout->_edge_offsets[vertex + 1];
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

    if (edges.size() == 0) {
        layout->_vertex_count = 0;
        layout->_edge_count   = 0;
        return;
    }

    G g_unfiltred;
    auto layout_unfilt   = detail::get_impl(g_unfiltred);
    vertex_t max_node_id = edges[0].first;
    for (auto u : edges) {
        vertex_t max_id_in_edge = std::max(u.first, u.second);
        max_node_id             = std::max(max_node_id, max_id_in_edge);
    }

    layout_unfilt->_vertex_count = max_node_id + 1;

    daal::services::Atomic<int_t> *degrees_cv =
        new daal::services::Atomic<int_t>[layout_unfilt->_vertex_count];

    daal::threader_for(layout_unfilt->_vertex_count, layout_unfilt->_vertex_count, [&](int u) {
        degrees_cv[u].set(0);
    });

    daal::threader_for(edges.size(), edges.size(), [&](int u) {
        degrees_cv[edges[u].first].inc();
        degrees_cv[edges[u].second].inc();
    });

    daal::services::Atomic<int_t> *rows_cv =
        new daal::services::Atomic<int_t>[layout_unfilt->_vertex_count + 1];
    int_t total_sum_degrees = 0;
    rows_cv[0].set(total_sum_degrees);

    for (size_t i = 0; i < layout_unfilt->_vertex_count; ++i) {
        total_sum_degrees += degrees_cv[i].get();
        rows_cv[i + 1].set(total_sum_degrees);
    }
    delete[] degrees_cv;

    layout_unfilt->_vertex_neighbors.resize(rows_cv[layout_unfilt->_vertex_count].get());
    layout_unfilt->_edge_offsets.resize(layout_unfilt->_vertex_count + 1);
    auto _rows_un = layout_unfilt->_edge_offsets.data();
    auto _cols_un = layout_unfilt->_vertex_neighbors.data();

    daal::threader_for(layout_unfilt->_vertex_count + 1,
                       layout_unfilt->_vertex_count + 1,
                       [&](int n) {
                           _rows_un[n] = rows_cv[n].get();
                       });

    daal::threader_for(edges.size(), edges.size(), [&](int u) {
        _cols_un[rows_cv[edges[u].first].inc() - 1]  = edges[u].second;
        _cols_un[rows_cv[edges[u].second].inc() - 1] = edges[u].first;
    });
    delete[] rows_cv;

    //removing self-loops,  multiple edges from graph, and make neighbors in CSR sorted

    layout->_vertex_count = layout_unfilt->_vertex_count;

    layout->_degrees.resize(layout->_vertex_count);

    daal::threader_for(layout_unfilt->_vertex_count, layout_unfilt->_vertex_count, [&](int u) {
        std::sort(layout_unfilt->_vertex_neighbors.begin() + layout_unfilt->_edge_offsets[u],
                  layout_unfilt->_vertex_neighbors.begin() + layout_unfilt->_edge_offsets[u + 1]);
        auto neighs_u_new_end = std::unique(
            layout_unfilt->_vertex_neighbors.begin() + layout_unfilt->_edge_offsets[u],
            layout_unfilt->_vertex_neighbors.begin() + layout_unfilt->_edge_offsets[u + 1]);
        neighs_u_new_end =
            std::remove(layout_unfilt->_vertex_neighbors.begin() + layout_unfilt->_edge_offsets[u],
                        neighs_u_new_end,
                        u);
        layout->_degrees[u] = neighs_u_new_end - (layout_unfilt->_vertex_neighbors.begin() +
                                                  layout_unfilt->_edge_offsets[u]);
    });

    layout->_edge_offsets.resize(layout->_vertex_count + 1);

    total_sum_degrees        = 0;
    layout->_edge_offsets[0] = total_sum_degrees;

    for (size_t i = 0; i < layout->_vertex_count; ++i) {
        total_sum_degrees += layout->_degrees[i];
        layout->_edge_offsets[i + 1] = total_sum_degrees;
    }
    layout->_edge_count = layout->_edge_offsets[layout->_vertex_count] / 2;

    layout->_vertex_neighbors.resize(layout->_edge_offsets[layout->_vertex_count]);

    daal::threader_for(layout->_vertex_count, layout->_vertex_count, [&](int u) {
        auto neighs = get_vertex_neighbors_impl(g_unfiltred, u);
        for (int_t i = 0; i < get_vertex_degree_impl(g, u); i++) {
            *(layout->_vertex_neighbors.begin() + layout->_edge_offsets[u] + i) =
                *(neighs.first + i);
        }
    });

    return /*ok*/;
}

// template <typename G>
// void convert_to_csr_impl(const edge_list<vertex_type<G>> &edges, G &g) {
//     auto layout    = detail::get_impl(g);
//     using int_t    = typename G::vertex_size_type;
//     using vertex_t = typename G::vertex_type;

//     layout->_vertex_count = 0;
//     layout->_edge_count   = 0;

//     vertex_t max_id = 0;

//     for (auto edge : edges) {
//         max_id = std::max(max_id, std::max(edge.first, edge.second));
//         layout->_edge_count += 1;
//     }

//     layout->_vertex_count = max_id + 1;
//     layout->_degrees.resize(layout->_vertex_count);
//     std::fill(layout->_degrees.begin(), layout->_degrees.end(), 0);

//     for (auto edge : edges) {
//         layout->_degrees[edge.first]++;
//         layout->_degrees[edge.second]++;
//     }

//     layout->_edge_offsets.resize(layout->_vertex_count + 1);
//     auto _rows              = layout->_edge_offsets.data();
//     int_t total_sum_degrees = 0;
//     _rows[0]                = total_sum_degrees;

//     for (int_t i = 0; i < layout->_vertex_count; ++i) {
//         total_sum_degrees += layout->_degrees[i];
//         _rows[i + 1] = total_sum_degrees;
//     }

//     layout->_vertex_neighbors.resize(_rows[layout->_vertex_count] + 1);
//     auto _cols = layout->_vertex_neighbors.data();
//     auto offests(layout->_edge_offsets);

//     for (auto edge : edges) {
//         _cols[offests[edge.first]++]  = edge.second;
//         _cols[offests[edge.second]++] = edge.first;
//     }
// }

template void convert_to_csr_impl(const edge_list<vertex_type<graph64>> &edges, graph64 &g);

template void convert_to_csr_impl(const edge_list<vertex_type<graph32>> &edges, graph32 &g);

} // namespace oneapi::dal::preview