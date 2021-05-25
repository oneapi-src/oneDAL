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

#include <memory>

#include "example_util/utils.hpp"
#include "oneapi/dal/algo/shortest_paths.hpp"
#include "oneapi/dal/graph/directed_adjacency_vector_graph.hpp"
#include "oneapi/dal/io/graph_csv_data_source.hpp"
#include "oneapi/dal/io/load_graph.hpp"

namespace dal = oneapi::dal;
using namespace dal::preview::shortest_paths;

int main(int argc, char** argv) {
    const auto filename = get_data_path("weighted_edge_list.csv");

    //read the graph
    const dal::preview::graph_csv_data_source ds(filename);

    using vertex_type = int32_t;
    using weight_type = double;
    using my_graph_type = dal::preview::directed_adjacency_vector_graph<vertex_type, weight_type>;

    const dal::preview::load_graph::
        descriptor<dal::preview::weighted_edge_list<vertex_type, weight_type>, my_graph_type>
            d;
    const auto my_graph = dal::preview::load_graph::load(d, ds);

    std::allocator<char> alloc;
    // set algorithm parameters
    const auto shortest_paths_desc =
        descriptor<float, method::delta_stepping, task::one_to_all, std::allocator<char>>(
            0,
            0.85,
            optional_results::distances | optional_results::predecessors,
            alloc);
    // compute shortest paths
    const auto result_shortest_paths = dal::preview::traverse(shortest_paths_desc, my_graph);

    // extract the result
    std::cout << "Distances: " << result_shortest_paths.get_distances() << std::endl;
    std::cout << "Predecessors: " << result_shortest_paths.get_predecessors() << std::endl;

    return 0;
}
