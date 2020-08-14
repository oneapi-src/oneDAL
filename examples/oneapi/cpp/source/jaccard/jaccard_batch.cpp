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
#include "oneapi/dal/data/table.hpp"
#include "oneapi/dal/data/undirected_adjacency_array_graph.hpp"
#include "oneapi/dal/util/csv_data_source.hpp"
#include "oneapi/dal/util/load_graph.hpp"

const std::string filename("../data/graph.csv");

using namespace oneapi::dal;
using namespace oneapi::dal::preview;

int main(int argc, char **argv) {
  // read the graph
  csv_data_source ds(filename);
  load_graph::descriptor<> d;
  auto my_graph = load_graph::load(d, ds);

  // set blocks ranges
  auto row_range_begin = 0;
  auto row_range_end = 2;
  auto column_range_begin = 0;
  auto column_range_end = 3;

  // comupte the number of the vertex pairs in the block of the graph
  auto vertex_pairs_count = (row_range_end - row_range_begin) *
                            (column_range_end - column_range_begin);

  // compute the size of the result element for the algorithm
  auto vertex_pair_element_count = 2;   // 2 elements in the vertex pair
  auto jaccard_coeff_element_count = 1; // 1 Jaccard coeff for the vertex pair

  auto vertex_pair_size =
      vertex_pair_element_count * sizeof(int32_t); // size in bytes
  auto jaccard_coeff_size =
      jaccard_coeff_element_count * sizeof(float); // size in bytes

  // compute the maximal required memory for the result of the block processing
  // in bytes
  auto block_result_size =
      (vertex_pair_element_count + jaccard_coeff_element_count) *
      vertex_pair_size * jaccard_coeff_size * vertex_pairs_count;

  // allocate memory for the result of the block processing
  auto result_buffer_ptr =
      std::shared_ptr<byte_t>(new byte_t[block_result_size]);

  // set algorithm parameters
  const auto jaccard_desc_default = jaccard::descriptor<>().set_block(
      {row_range_begin, row_range_end}, {column_range_begin, column_range_end});

  // compute Jaccard similarity coefficients
  auto result_vertex_similarity =
      vertex_similarity(jaccard_desc_default, my_graph,
                        static_cast<void *>(result_buffer_ptr.get()));

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
