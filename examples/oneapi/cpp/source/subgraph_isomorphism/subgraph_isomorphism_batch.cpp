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
#include <map>
#include <set>

#include "example_util/utils.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/io/graph_csv_data_source.hpp"
#include "oneapi/dal/io/load_graph.hpp"
#include "oneapi/dal/table/common.hpp"

namespace dal = oneapi::dal;

int main(int argc, char **argv) {
    auto target_filename = get_data_path("si_target_graph.csv");
    auto pattern_filename = get_data_path("si_pattern_graph.csv");

    const dal::preview::graph_csv_data_source ds_target(target_filename);
    const dal::preview::load_graph::descriptor<> d_target;
    const auto target_graph = dal::preview::load_graph::load(d_target, ds_target);

    const dal::preview::graph_csv_data_source ds_pattern(pattern_filename);
    const dal::preview::load_graph::descriptor<> d_pattern;
    const auto pattern_graph = dal::preview::load_graph::load(d_pattern, ds_pattern);

    std::allocator<char> alloc;

    // set algorithm parameters
    const auto subgraph_isomorphism_desc =
        dal::preview::subgraph_isomorphism::descriptor<>(alloc)
            .set_kind(dal::preview::subgraph_isomorphism::kind::non_induced)
            .set_semantic_match(false)
            .set_max_match_count(100);

    const auto result =
        dal::preview::graph_matching(subgraph_isomorphism_desc, target_graph, pattern_graph);

    // extract the result
    std::cout << "Number of matchings: " << result.get_match_count() << std::endl;
    std::cout << "Matchings:" << std::endl << result.get_vertex_match() << std::endl;

    return 0;
}
