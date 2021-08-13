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
#include "oneapi/dal/algo/louvain.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/io/graph_csv_data_source.hpp"
#include "oneapi/dal/io/load_graph.hpp"

namespace dal = oneapi::dal;
using namespace dal::preview::louvain;

int main(int argc, char** argv) {
    const auto filename = get_data_path("weighted_edge_list.csv");

    // read the graph
    const dal::preview::graph_csv_data_source ds(filename);

    using vertex_type = int32_t;
    using weight_type = double;
    using graph_type = dal::preview::undirected_adjacency_vector_graph<vertex_type, weight_type>;

    auto graph = dal::read<graph_t>(dal::csv::data_source{ filename },
                                    dal::preview::read_mode::weighted_edge_list);
    const auto graph = dal::preview::load_graph::load(d, ds);

    // set algorithm parameters
    const auto louvain_desc =
        descriptor<float, method::fast, task::vertex_partitioning, std::allocator<char>>()
            .set_resolution(1)
            .set_accuracy_threshold(0.0001)
            .set_max_iteration_count(3);
    // compute louvain
    try {
        const std::int64_t rows_count = 7;
        const std::int64_t cols_count = 1;
        const std::int64_t data[] = { 0, 1, 2, 3, 4, 5, 6 };
        const auto initial_labels = dal::homogen_table::wrap(data, rows_count, cols_count);

        const auto result = dal::preview::vertex_partitioning(louvain_desc, graph, initial_labels);

        std::cout << "Modularity: " << result.get_modularity() << std::endl;
        std::cout << "Number of communities: " << result.get_community_count() << std::endl;
        std::cout << "Get communities' labels: " << result.get_labels() << std::endl;
    }
    catch (dal::unimplemented& e) {
        std::cout << e.what() << std::endl;
    }
    return 0;
}
