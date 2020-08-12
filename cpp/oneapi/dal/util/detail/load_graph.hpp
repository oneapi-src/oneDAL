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

#include <fstream>

#include "oneapi/dal/data/graph_common.hpp"
#include "oneapi/dal/data/undirected_adjacency_array_graph.hpp"
#include "oneapi/dal/util/csv_data_source.hpp"
#include "oneapi/dal/util/load_graph_descriptor.hpp"

namespace oneapi::dal::preview::load_graph::detail {

edge_list<std::int32_t> load_edge_list(const std::string &name) {
    using int_t = std::int32_t;
    edge_list<int_t> elist;
    std::ifstream file(name);
    int_t source_vertex      = 0;
    int_t destination_vertex = 0;
    while (file >> source_vertex >> destination_vertex) {
        auto edge = std::make_pair(source_vertex, destination_vertex);
        elist.push_back(edge);
    }

    file.close();
    return elist;
}

template <typename Graph>
void convert_to_csr_impl(const edge_list<vertex_type<Graph>> &edges, Graph &g) {
    auto layout    = oneapi::dal::preview::detail::get_impl(g);
    using int_t    = typename Graph::vertex_size_type;
    using vertex_t = typename Graph::vertex_type;

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

template <typename Descriptor, typename DataSource>
output_type<Descriptor> load_impl(const Descriptor &desc, const DataSource &data_source);

template <>
output_type<descriptor<>> load_impl<descriptor<>, csv_data_source>(
    const descriptor<> &desc,
    const csv_data_source &data_source) {
    output_type<descriptor<>> graph;
    convert_to_csr_impl(load_edge_list(data_source.get_filename()), graph);
    return graph;
}
} // namespace oneapi::dal::preview::load_graph::detail
