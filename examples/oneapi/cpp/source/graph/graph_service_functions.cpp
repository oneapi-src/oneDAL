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

#include <iostream>

#include "example_util/utils.hpp"
#include "oneapi/dal/graph/service_functions.hpp"
#include "oneapi/dal/io/csv.hpp"

namespace dal = oneapi::dal;

using namespace dal;

int main(int argc, char **argv) {
    const auto filename = get_data_path("graph.csv");

    using graph_t = preview::undirected_adjacency_vector_graph<>;
    auto graph = read<graph_t>(csv::data_source{ filename });

    std::cout << "Number of vertices: " << preview::get_vertex_count(graph) << std::endl;
    std::cout << "Number of edges: " << preview::get_edge_count(graph) << std::endl;

    preview::vertex_type<graph_t> vertex_id = 0;
    std::cout << "Degree of " << vertex_id << ": " << preview::get_vertex_degree(graph, vertex_id)
              << std::endl;

    for (preview::vertex_size_type<graph_t> i = 0; i < preview::get_vertex_count(graph); ++i) {
        std::cout << "Neighbors of " << i << ": ";
        const auto neigh = preview::get_vertex_neighbors(graph, i);
        for (auto u = neigh.first; u != neigh.second; ++u) {
            std::cout << *u << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
