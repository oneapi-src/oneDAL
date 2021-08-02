/*******************************************************************************
* Copyright 2021 Intel Corporation
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
#include "oneapi/dal/io/csv.hpp"

namespace dal = oneapi::dal;
using namespace dal::preview::shortest_paths;

int main(int argc, char** argv) {
    const auto filename = get_data_path("weighted_edge_list.csv");

    using vertex_type = int32_t;
    using weight_type = double;
    using graph_t = dal::preview::directed_adjacency_vector_graph<vertex_type, weight_type>;

    const auto graph = dal::read<graph_t>(dal::csv::data_source{ filename });

    std::allocator<char> alloc;
    // set algorithm parameters
    const auto shortest_paths_desc =
        descriptor<float, method::delta_stepping, task::one_to_all, std::allocator<char>>(
            0,
            0.85,
            optional_results::distances | optional_results::predecessors,
            alloc);
    // compute shortest paths
    const auto result_shortest_paths = dal::preview::traverse(shortest_paths_desc, graph);

    // extract the result
    std::cout << "Distances:" << std::endl << result_shortest_paths.get_distances() << std::endl;
    std::cout << "Predecessors:" << std::endl
              << result_shortest_paths.get_predecessors() << std::endl;

    return 0;
}
