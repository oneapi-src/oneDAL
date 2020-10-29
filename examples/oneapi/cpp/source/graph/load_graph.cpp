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
    const std::string filename = get_data_path("graph.csv");

    const onedal::preview::graph_csv_data_source ds(filename);
    const onedal::preview::load_graph::descriptor<> d;
    const auto my_graph = onedal::preview::load_graph::load(d, ds);

    std::cout << "Graph is read from file: " << filename << std::endl;
    std::cout << "Number of vertices: " << onedal::preview::get_vertex_count(my_graph) << std::endl;
    std::cout << "Number of edges: " << onedal::preview::get_edge_count(my_graph) << std::endl;
    return 0;
}
