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
#include "oneapi/dal/algo/jaccard.hpp"
#include "oneapi/dal/graph/undirected_adjacency_array_graph.hpp"
#include "oneapi/dal/io/graph_csv_data_source.hpp"
#include "oneapi/dal/io/load_graph.hpp"
#include "oneapi/dal/table/common.hpp"

using namespace oneapi::dal;
using namespace oneapi::dal::preview;

int main(int argc, char **argv) {
  const std::string filename = get_data_path("graph.csv");
  // read the graph
  graph_csv_data_source ds(filename);
  load_graph::descriptor<> d;
  auto my_graph = load_graph::load(d, ds);

  // set blocks ranges
  auto row_range_begin = 0; auto row_range_end = 2;
  auto column_range_begin = 0; auto column_range_end = 3;

  // set algorithm parameters
  const auto jaccard_desc = jaccard::descriptor<>().set_block(
      {row_range_begin, row_range_end}, {column_range_begin, column_range_end});

  // create caching builder for jaccard result
  jaccard::caching_builder builder;

  // compute Jaccard similarity coefficients
  auto result_vertex_similarity = vertex_similarity(jaccard_desc, my_graph, builder);

  // extract the result
  auto jaccard_coeffs = result_vertex_similarity.get_coeffs();
  auto vertex_pairs = result_vertex_similarity.get_vertex_pairs();
  auto nonzero_coeff_count = result_vertex_similarity.get_nonzero_coeff_count();

  std::cout << "The number of nonzero Jaccard coeffs in the block: "
            << nonzero_coeff_count << std::endl;

  std::cout << "Vertex pairs: " << std::endl;
  print_vertex_similarity_result(vertex_pairs, nonzero_coeff_count);

  std::cout << "Jaccard values: " << std::endl;
  print_vertex_similarity_result(jaccard_coeffs, nonzero_coeff_count);
}
