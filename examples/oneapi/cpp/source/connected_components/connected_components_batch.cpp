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
#include "oneapi/dal/algo/connected_components.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/io/graph_csv_data_source.hpp"
#include "oneapi/dal/io/load_graph.hpp"

namespace dal = oneapi::dal;
using namespace dal::preview::connected_components;

int main(int argc, char** argv) {
    const auto filename = get_data_path("graph.csv");

    // read the graph
    const dal::preview::graph_csv_data_source ds(filename);
    const dal::preview::load_graph::descriptor<> d;
    const auto my_graph = dal::preview::load_graph::load(d, ds);
    std::allocator<char> alloc;

    // set algorithm parameters
    const auto cc_desc = descriptor<float, method::afforest, task::vertex_partitioning, std::allocator<char>>(alloc);

    try {
        // compute connected_components
        const auto result_connected_components = dal::preview::vertex_partitioning(cc_desc, my_graph);

        // extract the result
        std::cout << "Components' labels: " << result_connected_components.get_labels() << std::endl;
        std::cout << "Number of connected components: " << result_connected_components.get_component_count() << std::endl;
    }
    catch (dal::unimplemented& e) {
        std::cout << "  " << e.what() << std::endl;
    }
    return 0;
}
