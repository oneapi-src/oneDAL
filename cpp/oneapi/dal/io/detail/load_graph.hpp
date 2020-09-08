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

#pragma once

#include <algorithm>
#include <fstream>

#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/graph/detail/graph_container.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_array_graph_impl.hpp"
#include "oneapi/dal/graph/graph_common.hpp"
#include "oneapi/dal/graph/undirected_adjacency_array_graph.hpp"
#include "oneapi/dal/io/detail/load_graph_service.hpp"
#include "oneapi/dal/io/graph_csv_data_source.hpp"
#include "oneapi/dal/io/load_graph_descriptor.hpp"
#include "services/daal_atomic_int.h"
#include "services/daal_memory.h"

namespace oneapi::dal::preview::load_graph::detail {
edge_list<std::int32_t> load_edge_list(const std::string &name) {
    using int_t = std::int32_t;

    std::ifstream file(name);
    if (!file.is_open()) {
        throw invalid_argument("File not found");
    }
    edge_list<int_t> elist;

    char source_vertex[32], destination_vertex[32];
    while (file >> source_vertex >> destination_vertex) {
        auto edge = std::make_pair(daal_string_to_int(&source_vertex[0], 0),
                                   daal_string_to_int(&destination_vertex[0], 0));
        elist.push_back(edge);
    }

    file.close();
    return elist;
}

template <typename Graph>
void convert_to_csr_impl(const edge_list<vertex_type<Graph>> &edges, Graph &g) {
    auto layout = oneapi::dal::preview::detail::get_impl(g);
    using vertex_t = typename Graph::vertex_type;
    using vector_vertex_t = typename Graph::vertex_set;
    using vector_edge_t = typename Graph::edge_set;
    using allocator_t = typename Graph::allocator_type;
    using vertex_size_t = typename Graph::vertex_size_type;

    if (edges.size() == 0) {
        layout->_vertex_count = 0;
        layout->_edge_count = 0;
        throw invalid_argument("Empty edge list");
    }

    vertex_t max_node_id = edges[0].first;
    for (auto u : edges) {
        vertex_t max_id_in_edge = std::max(u.first, u.second);
        max_node_id = std::max(max_node_id, max_id_in_edge);
    }

    vertex_t _unf_vertex_count = max_node_id + 1;

    using atomic_t = typename daal::services::Atomic<vertex_t>;
    using allocator_atomic_t =
        typename std::allocator_traits<allocator_t>::template rebind_alloc<atomic_t>;

    auto *degrees_vec = new (std::nothrow)
        oneapi::dal::preview::detail::graph_container<atomic_t, allocator_atomic_t>(
            _unf_vertex_count);
    if (degrees_vec == nullptr) {
        throw bad_alloc();
    }
    atomic_t *degrees_cv = degrees_vec->data();
    if (degrees_cv == nullptr) {
        throw bad_alloc();
    }

    threader_for(_unf_vertex_count, _unf_vertex_count, [&](vertex_t u) {
        degrees_cv[u].set(0);
    });

    threader_for(edges.size(), edges.size(), [&](vertex_t u) {
        degrees_cv[edges[u].first].inc();
        degrees_cv[edges[u].second].inc();
    });

    auto *rows_vec = new (std::nothrow)
        oneapi::dal::preview::detail::graph_container<atomic_t, allocator_atomic_t>(
            _unf_vertex_count + 1);
    if (rows_vec == nullptr) {
        throw bad_alloc();
    }
    atomic_t *rows_cv = rows_vec->data();
    if (rows_cv == nullptr) {
        throw bad_alloc();
    }

    vertex_t total_sum_degrees = 0;
    rows_cv[0].set(total_sum_degrees);

    for (vertex_t i = 0; i < _unf_vertex_count; ++i) {
        total_sum_degrees += degrees_cv[i].get();
        rows_cv[i + 1].set(total_sum_degrees);
    }
    delete degrees_vec;

    vector_vertex_t _unf_vert_neighs_vec(rows_cv[_unf_vertex_count].get());
    vector_edge_t _unf_edge_offset_vec(_unf_vertex_count + 1);
    auto _unf_edge_offset_arr = _unf_edge_offset_vec.data();
    auto _unf_vert_neighs_arr = _unf_vert_neighs_vec.data();

    threader_for(_unf_vertex_count + 1, _unf_vertex_count + 1, [&](vertex_t n) {
        _unf_edge_offset_arr[n] = rows_cv[n].get();
    });

    threader_for(edges.size(), edges.size(), [&](vertex_t u) {
        _unf_vert_neighs_arr[rows_cv[edges[u].first].inc() - 1] = edges[u].second;
        _unf_vert_neighs_arr[rows_cv[edges[u].second].inc() - 1] = edges[u].first;
    });
    delete rows_vec;

    //removing self-loops,  multiple edges from graph, and make neighbors in CSR sorted

    layout->_vertex_count = _unf_vertex_count;

    layout->_degrees = std::move(vector_vertex_t(layout->_vertex_count));

    threader_for(_unf_vertex_count, _unf_vertex_count, [&](vertex_t u) {
        auto start_p = _unf_vert_neighs_vec.begin() + _unf_edge_offset_vec[u];
        auto end_p = _unf_vert_neighs_vec.begin() + _unf_edge_offset_vec[u + 1];
        std::sort(start_p, end_p);
        auto neighs_u_new_end = std::unique(start_p, end_p);
        neighs_u_new_end = std::remove(start_p, neighs_u_new_end, u);
        layout->_degrees[u] = (vertex_t)std::distance(start_p, neighs_u_new_end);
    });

    layout->_edge_offsets.clear();
    layout->_edge_offsets.reserve(layout->_vertex_count + 1);

    total_sum_degrees = 0;
    layout->_edge_offsets.push_back(total_sum_degrees);

    for (vertex_size_t i = 0; i < layout->_vertex_count; ++i) {
        total_sum_degrees += layout->_degrees[i];
        layout->_edge_offsets.push_back(total_sum_degrees);
    }
    layout->_edge_count = layout->_edge_offsets[layout->_vertex_count] / 2;

    layout->_vertex_neighbors =
        std::move(vector_vertex_t(layout->_edge_offsets[layout->_vertex_count]));

    auto vert_neighs = layout->_vertex_neighbors.data();
    auto edge_offs = layout->_edge_offsets.data();
    threader_for(layout->_vertex_count, layout->_vertex_count, [&](vertex_t u) {
        auto u_neighs = vert_neighs + edge_offs[u];
        auto _u_neighs_unf = _unf_vert_neighs_arr + _unf_edge_offset_arr[u];
        for (vertex_t i = 0; i < layout->_degrees[u]; i++) {
            u_neighs[i] = _u_neighs_unf[i];
        }
    });

    return;
} // namespace oneapi::dal::preview::load_graph::detail

template <typename Descriptor, typename DataSource>
output_type<Descriptor> load_impl(const Descriptor &desc, const DataSource &data_source) {
    output_type<Descriptor> graph;
    convert_to_csr_impl(load_edge_list(data_source.get_filename()), graph);
    return graph;
}
} // namespace oneapi::dal::preview::load_graph::detail
