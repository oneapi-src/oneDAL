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

#include <chrono>
#include <iostream>

#include "example_util/utils.hpp"
#include "oneapi/dal/algo/jaccard.hpp"
#include "oneapi/dal/data/graph_service_functions.hpp"
#include "oneapi/dal/data/table.hpp"
#include "oneapi/dal/data/undirected_adjacency_array_graph.hpp"
#include "oneapi/dal/util/csv_data_source.hpp"
#include "oneapi/dal/util/load_graph.hpp"

#include "tbb/global_control.h"
#include "tbb/parallel_for.h"

using namespace oneapi::dal;
using namespace oneapi::dal::preview;
using namespace std;
using namespace std::chrono;

// input parameters - processed graph, size of block by row, size of block by
// column algorithm process only upper triangular matrix
template <class Graph>
void vertex_similarity_block_processing(const Graph &g, const int32_t row_size,
                                        const int32_t column_size) {
  // comupte the number of the vertex pairs in the block of the graph
  auto vertex_pairs_count = row_size * column_size;
  auto vertex_pair_element_count = 2;   // 2 elements in the vertex pair
  auto jaccard_coeff_element_count = 1; // 1 Jaccard coeff for the vertex pair

  auto vertex_pair_size =
      vertex_pair_element_count * sizeof(int32_t); // size in bytes
  auto jaccard_coeff_size =
      jaccard_coeff_element_count * sizeof(float); // size in bytes
  // compute the maximal required memory for the result of the block processing
  // in bytes
  auto block_result_size = (vertex_pair_element_count * vertex_pair_size +
                            jaccard_coeff_element_count * jaccard_coeff_size) *
                           vertex_pairs_count;
  std::vector<std::vector<byte_t>> processing_blocks(
      tbb::this_task_arena::max_concurrency(),
      std::vector<byte_t>(block_result_size)); // memory for all threads
  // start block precessing
  int32_t vertex_count = get_vertex_count(g);
  int32_t rows_count = vertex_count / row_size; // the number rows
  if (vertex_count % row_size) {
    rows_count++;
  }
  tbb::parallel_for(
      tbb::blocked_range<int>(0, rows_count),
      [&](const tbb::blocked_range<int> &r) {
        for (int i = r.begin(); i != r.end(); ++i) {
          // compute range for row
          int32_t block_begin_row = i * row_size;
          int32_t block_end_row = (i + 1) * row_size;
          int32_t begin_column =
              1 + block_begin_row; // start column ranges from diagonal
          int32_t columns_count =
              (vertex_count - begin_column) / column_size; // the number columns
          if ((vertex_count - begin_column) % column_size) {
            columns_count++;
          }
          tbb::parallel_for(
              tbb::blocked_range<int>(0, columns_count),
              [&](const tbb::blocked_range<int> &inner_r) {
                for (int j = inner_r.begin(); j != inner_r.end(); ++j) {
                  // compute range for column
                  int32_t block_begin_column = begin_column + j * column_size;
                  int32_t block_end_column =
                      begin_column + (j + 1) * column_size;
                  // set block ranges
                  const auto jaccard_desc_default =
                      jaccard::descriptor<>().set_block(
                          {block_begin_row,
                           std::min(block_end_row, vertex_count)},
                          {block_begin_column,
                           std::min(block_end_column, vertex_count)});
                  // compute coeffs
                  vertex_similarity(
                      jaccard_desc_default, g,
                      static_cast<void *>(
                          (processing_blocks
                               [tbb::this_task_arena::current_thread_index()])
                              .data()));
                }
              },
              tbb::simple_partitioner{});
        }
      },
      tbb::simple_partitioner{});
}

int main(int argc, char **argv) {
  // read the graph
  if (argc < 2)
    return 0;
  std::string filename = argv[1];
  csv_data_source ds(filename);
  load_graph::descriptor<> d;
  auto my_graph = load_graph::load(d, ds);

  int32_t row_size = 1024;
  int32_t column_size = 1024;
  int32_t tbb_threads_number = 1;

  tbb::global_control c(tbb::global_control::max_allowed_parallelism,
                        tbb_threads_number);

  vertex_similarity_block_processing(my_graph, row_size, column_size);

  return 0;
}