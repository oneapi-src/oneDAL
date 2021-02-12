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
#include "oneapi/dal/algo/triangle_counting.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/io/graph_csv_data_source.hpp"
#include "oneapi/dal/io/load_graph.hpp"
#include "oneapi/dal/table/common.hpp"

namespace dal = oneapi::dal;
using namespace dal::preview::triangle_counting;

int main(int argc, char **argv) {
    const auto filename = get_data_path(argv[1]);

    // read the graph
    const dal::preview::graph_csv_data_source ds(filename);
    const dal::preview::load_graph::descriptor<> d;
    const auto my_graph = dal::preview::load_graph::load(d, ds);

    // set algorithm parameters
    const auto tc_desc =
        descriptor<float, method::ordered_count, task::global>(kind::undirected_clique, relabel::no);

    // compute local triangles
    const auto result_vertex_ranking =
        dal::preview::vertex_ranking(tc_desc, my_graph);

    // extract the result
    const auto triangles = result_vertex_ranking.get_global_rank();

    std::cout << "Global triangles count: " << triangles << std::endl;
    /*auto arr = oneapi::dal::column_accessor<const std::int64_t>(triangles).pull(); 
    const auto x = arr.get_data(); 

    for(auto i = 0; i < get_vertex_count(my_graph); i++) { 
        std::cout << "Vertex " << i <<":/t" << x[i] << std::endl; 
    } */
}
