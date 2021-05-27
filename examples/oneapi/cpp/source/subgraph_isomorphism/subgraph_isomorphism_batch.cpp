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
#include "oneapi/dal/graph/service_functions.hpp"

namespace dal = oneapi::dal;

inline dal::preview::edge_list<std::int32_t> load_vertex_labels_and_edge_list(
    const std::string &name,
    std::set<std::string> &labels_set,
    std::vector<std::string> &labels) {
    using int_t = std::int32_t;

    std::ifstream file(name);
    if (!file.is_open()) {
        throw dal::invalid_argument(dal::detail::error_messages::file_not_found());
    }
    std::string tmp;
    std::int32_t vertices_count, edges_count;

    file >> tmp; // read comment
    file >> vertices_count; // read number of values

    labels.reserve(vertices_count);
    for (int i = 0; i < vertices_count; i++) {
        file >> tmp; // read label
        labels.push_back(tmp);
        labels_set.insert(tmp);
    }

    file >> edges_count; // read number of edges
    dal::preview::edge_list<int_t> elist;
    elist.reserve(edges_count);
    std::int32_t source_vertex, destination_vertex;
    while (file >> source_vertex >> destination_vertex) {
        elist.push_back(std::make_pair(source_vertex, destination_vertex));
    }

    file.close();
    return elist;
}
template <typename Graph>
void add_lables(Graph &graph,
                const std::map<std::string, std::int32_t> &labels_map,
                const std::vector<std::string> &labels) {
    auto &graph_impl = oneapi::dal::detail::get_impl(graph);
    auto &vertex_allocator = graph_impl._vertex_allocator;
    auto &vv_p = graph_impl.get_vertex_values();

    auto vertex_count = dal::preview::get_vertex_count(graph);
    std::int32_t *labels_array =
        oneapi::dal::preview::detail::allocate(vertex_allocator, vertex_count);
    vv_p = dal::array<std::int32_t>::wrap(labels_array, vertex_count);
    for (int i = 0; i < vertex_count; i++) {
        labels_array[i] = labels_map.at(labels[i]);
    }
}
template <typename Graph>
void load_graph_gff(const std::string filename_target,
                    const std::string filename_pattern,
                    Graph &target,
                    Graph &pattern) {
    // read the graph
    std::set<std::string> mapping;
    std::vector<std::string> labels_p, labels_t;
    const dal::preview::graph_csv_data_source ds_target(filename_target),
        ds_pattern(filename_pattern); // n vertices

    {
        auto el_p = load_vertex_labels_and_edge_list(filename_pattern, mapping, labels_p);
        dal::preview::load_graph::detail::convert_to_csr_impl(el_p, pattern);
        auto el_t = load_vertex_labels_and_edge_list(filename_target, mapping, labels_t);
        dal::preview::load_graph::detail::convert_to_csr_impl(el_t, target);
    }

    std::map<std::string, std::int32_t> proper_map;
    std::int32_t index = 0;
    for (auto label : mapping) {
        proper_map.insert(std::make_pair(label, index));
        index++;
    }
    add_lables(target, proper_map, labels_t);
    add_lables(pattern, proper_map, labels_p);
}

int main(int argc, char **argv) {
    // auto target_filename = get_data_path("si_target_graph.csv");
    // auto pattern_filename = get_data_path("si_pattern_graph.csv");

    // const dal::preview::graph_csv_data_source ds_target(target_filename);
    // const dal::preview::load_graph::descriptor<> d_target;
    // const auto target_graph = dal::preview::load_graph::load(d_target, ds_target);

    // const dal::preview::graph_csv_data_source ds_pattern(pattern_filename);
    // const dal::preview::load_graph::descriptor<> d_pattern;
    // const auto pattern_graph = dal::preview::load_graph::load(d_pattern, ds_pattern);

    auto target_filename = get_data_path(
        "/nfs/inn/disks/nn-ssg_spd_numerics_users/maverbuk/daal_branches/si-proto/data/PDBSv1/singles/3dmk.pdb.gff");
    auto pattern_filename = get_data_path(
        "/nfs/inn/disks/nn-ssg_spd_numerics_users/maverbuk/daal_branches/si-proto/data/PDBSv1/singles/3dmk.pdb.gff_queries/query64_2.gff");

    if (argc == 3) {
        target_filename = get_data_path(argv[1]);
        pattern_filename = get_data_path(argv[2]);
    }

    typedef dal::preview::undirected_adjacency_vector_graph<std::int32_t> graph_t;
    graph_t target_graph, pattern_graph;
    load_graph_gff(target_filename, pattern_filename, target_graph, pattern_graph);

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
    // std::cout << "Matchings:" << std::endl << result.get_vertex_match() << std::endl;

    return 0;
}
