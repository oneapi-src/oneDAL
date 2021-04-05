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

#include "example_util/output_helpers_graph.hpp"
#include "example_util/utils.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/io/graph_csv_data_source.hpp"
#include "oneapi/dal/io/load_graph.hpp"
#include "oneapi/dal/table/common.hpp"

namespace dal = oneapi::dal;

int main(int argc, char **argv) {
    // auto target_filename = get_data_path("si_target_graph.csv");
    // auto pattern_filename = get_data_path("si_pattern_graph.csv");
    auto target_filename = get_data_path(
        "/export/users/orazvens/si-non-induced/subgraph-isomorphism-prototype/data/PDBSv1/singles/103l.pdb.gff");
    auto pattern_filename = get_data_path(
        "/export/users/orazvens/si-non-induced/subgraph-isomorphism-prototype/data/PDBSv1/singles/103l.pdb.gff_queries/query32_1.gff");

    if (argc == 3) {
        target_filename = get_data_path(argv[1]);
        pattern_filename = get_data_path(argv[2]);
    }

    std::cout << "Search " << pattern_filename << " in " << target_filename << std::endl;

    // read the graph
    const dal::preview::graph_csv_data_source ds_target(target_filename); // n vertices
    const dal::preview::load_graph::descriptor<
        dal::preview::edge_list<>,
        dal::preview::undirected_adjacency_vector_graph<std::int32_t>>
        d_target;
    const auto target_graph = dal::preview::load_graph::load_gff(d_target, ds_target);

    const dal::preview::graph_csv_data_source ds_pattern(pattern_filename); // m vertices
    const dal::preview::load_graph::descriptor<
        dal::preview::edge_list<>,
        dal::preview::undirected_adjacency_vector_graph<std::int32_t>>
        d_pattern;
    const auto pattern_graph = dal::preview::load_graph::load_gff(d_pattern, ds_pattern);

    auto &vv_t = dal::detail::get_impl(target_graph).get_vertex_values();
    auto &vv_p = dal::detail::get_impl(pattern_graph).get_vertex_values();

    std::allocator<char> alloc;
    // set algorithm parameters
    const auto subgraph_isomorphism_desc =
        dal::preview::subgraph_isomorphism::descriptor<>(alloc)
            .set_kind(dal::preview::subgraph_isomorphism::kind::induced)
            .set_semantic_match(false)
            .set_max_match_count(100);

    // compute matchings
    const auto result =
        dal::preview::graph_matching(subgraph_isomorphism_desc, target_graph, pattern_graph);

    // extract the result
    const auto match_count = result.get_match_count();

    // print_table_int(result.get_vertex_match());
    print_table_int_sorted(result.get_vertex_match());
    // std::cout << "Matchings:\n" << result.get_vertex_match() << std::endl;
}
