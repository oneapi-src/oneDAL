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

#include <iostream>

#include "example_util/utils.hpp"
#include "oneapi/dal/graph/graph_service_functions.hpp"
#include "oneapi/dal/graph/undirected_adjacency_array_graph.hpp"
#include "oneapi/dal/io/graph_csv_data_source.hpp"
#include "oneapi/dal/io/load_graph.hpp"

namespace onedal = oneapi::dal;

int main(int argc, char **argv) {
    const auto filename = get_data_path("graph.csv");

    const onedal::preview::graph_csv_data_source ds(filename);
    using my_graph_type = onedal::preview::undirected_adjacency_array_graph<>;
    const onedal::preview::load_graph::descriptor<onedal::preview::edge_list<int32_t>,
                                                  my_graph_type>
        desc;
    const auto my_graph = onedal::preview::load_graph::load(desc, ds);
    std::cout << "Number of vertices: " << onedal::preview::get_vertex_count(my_graph) << std::endl;
    std::cout << "Number of edges: " << onedal::preview::get_edge_count(my_graph) << std::endl;

    onedal::preview::vertex_type<my_graph_type> vertex_id = 0;
    std::cout << "Degree of " << vertex_id << ": "
              << onedal::preview::get_vertex_degree(my_graph, vertex_id) << std::endl;

    for (onedal::preview::vertex_size_type<my_graph_type> j = 0;
         j < onedal::preview::get_vertex_count(my_graph);
         ++j) {
        std::cout << "Neighbors of " << j << ": ";
        const auto neigh = onedal::preview::get_vertex_neighbors(my_graph, j);
        for (auto i = neigh.first; i != neigh.second; ++i) {
            std::cout << *i << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
