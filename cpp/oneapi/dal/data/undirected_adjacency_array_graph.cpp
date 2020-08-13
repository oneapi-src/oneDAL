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
    return layout->_degrees[vertex];
}

template ONEAPI_DAL_EXPORT vertex_edge_size_type<graph32> get_vertex_degree_impl(
    const graph32 &g,
    const vertex_type<graph32> &vertex);

template <typename G>
ONEAPI_DAL_EXPORT auto get_vertex_neighbors_impl(const G &g, const vertex_type<G> &vertex) noexcept
    -> const_vertex_edge_range_type<G> {
    const auto &layout = detail::get_impl(g);
    const_vertex_edge_iterator_type<G> vertex_neighbors_begin =
        layout->_vertex_neighbors.begin() + layout->_edge_offsets[vertex];
    const_vertex_edge_iterator_type<G> vertex_neighbors_end =
        layout->_vertex_neighbors.begin() + layout->_edge_offsets[vertex + 1];
    return std::make_pair(vertex_neighbors_begin, vertex_neighbors_end);
}

template ONEAPI_DAL_EXPORT const_vertex_edge_range_type<graph32> get_vertex_neighbors_impl(
    const graph32 &g,
    const vertex_type<graph32> &vertex);

template <typename G>
ONEAPI_DAL_EXPORT void convert_to_csr_impl(const edge_list<vertex_type<G>> &edges, G &g) {
    auto layout           = detail::get_impl(g);
    using int_t           = typename G::vertex_size_type;
    using vertex_t        = typename G::vertex_type;
    using vector_vertex_t = typename G::vertex_set;

    if (edges.size() == 0) {
        layout->_vertex_count = 0;
        layout->_edge_count   = 0;
        return;
    }

    G g_unfiltred;
    auto layout_unfilt = detail::get_impl(g_unfiltred);

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

    layout->_degrees = std::move(vector_vertex_t(layout->_vertex_count));

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

    layout->_edge_offsets.clear();
    layout->_edge_offsets.reserve(layout->_vertex_count + 1);

    total_sum_degrees = 0;
    layout->_edge_offsets.push_back(total_sum_degrees);

    for (size_t i = 0; i < layout->_vertex_count; ++i) {
        total_sum_degrees += layout->_degrees[i];
        layout->_edge_offsets.push_back(total_sum_degrees);
    }
    layout->_edge_count = layout->_edge_offsets[layout->_vertex_count] / 2;
    layout->_vertex_neighbors =
        std::move(vector_vertex_t(layout->_edge_offsets[layout->_vertex_count]));

    daal::threader_for(layout->_vertex_count, layout->_vertex_count, [&](int u) {
        auto neighs = get_vertex_neighbors_impl(g_unfiltred, u);
        for (int_t i = 0; i < get_vertex_degree_impl(g, u); i++) {
            *(layout->_vertex_neighbors.begin() + layout->_edge_offsets[u] + i) =
                *(neighs.first + i);
        }
    });

    return /*ok*/;
}

template ONEAPI_DAL_EXPORT void convert_to_csr_impl(const edge_list<vertex_type<graph32>> &edges,
                                                    graph32 &g);

} // namespace oneapi::dal::preview