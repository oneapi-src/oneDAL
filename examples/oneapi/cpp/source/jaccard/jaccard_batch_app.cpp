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

#include "tbb/global_control.h"
#include "tbb/parallel_for.h"

#include "example_util/utils.hpp"
#include "oneapi/dal/algo/jaccard.hpp"
#include "oneapi/dal/graph/graph_service_functions.hpp"
#include "oneapi/dal/graph/undirected_adjacency_array_graph.hpp"
#include "oneapi/dal/io/csv_data_source.hpp"
#include "oneapi/dal/io/load_graph.hpp"
#include "oneapi/dal/table/homogen.hpp"

using namespace oneapi::dal;
using namespace oneapi::dal::preview;
using namespace std;
using namespace std::chrono;

/// Computes Jaccard similarity coefficients for the graph. The upper triangular
/// matrix is processed only as it is symmetic for undirected graph.
///
/// @param [in]   g  The input graph
/// @param [in]   row_size  The size of block by rows
/// @param [in]   column_size The size of block by columns
template <class Graph>
void vertex_similarity_block_processing(const Graph &g,
                                        const std::int32_t row_size,
                                        const std::int32_t column_size);

int main(int argc, char **argv) {
  // load the graph
  std::string filename = get_data_path("graph.csv");
  ;
  csv_data_source ds(filename);
  load_graph::descriptor<> d;
  auto my_graph = load_graph::load(d, ds);

  // set the block sizes for Jaccard similarity block processing
  int32_t row_size = 2;
  int32_t column_size = 5;

  // set the number of threads
  int32_t tbb_threads_number = 4;
  tbb::global_control c(tbb::global_control::max_allowed_parallelism,
                        tbb_threads_number);

  // compute Jaccard similarity coefficients for the graph
  vertex_similarity_block_processing(my_graph, row_size, column_size);

  return 0;
}

template <class Graph>
void vertex_similarity_block_processing(const Graph &g,
                                        const std::int32_t row_size,
                                        const std::int32_t column_size) {
  // compute the maximum required memory for the result of the block processing
  // in bytes
  auto max_block_size = compute_max_block_size(0, row_size, 0, column_size);

  // reserve memory for all threads
  std::vector<std::vector<byte_t>> processing_blocks(
      tbb::this_task_arena::max_concurrency(),
      std::vector<byte_t>(max_block_size));

  // compute the number of vertices in graph
  std::int32_t vertex_count = get_vertex_count(g);

  // compute the number of rows
  std::int32_t rows_count = vertex_count / row_size;
  if (vertex_count % row_size) {
    rows_count++;
  }

  // parallel processing by rows
  tbb::parallel_for(
      tbb::blocked_range<int>(0, rows_count),
      [&](const tbb::blocked_range<int> &r) {
        for (int i = r.begin(); i != r.end(); ++i) {
          // compute the range of rows
          int32_t block_begin_row = i * row_size;
          int32_t block_end_row = (i + 1) * row_size;

          // start column ranges from diagonal
          int32_t begin_column = 1 + block_begin_row;

          // compute the number of columns
          int32_t columns_count = (vertex_count - begin_column) / column_size;
          if ((vertex_count - begin_column) % column_size) {
            columns_count++;
          }

          // parallel processing by columns
          tbb::parallel_for(
              tbb::blocked_range<int>(0, columns_count),
              [&](const tbb::blocked_range<int> &inner_r) {
                for (int j = inner_r.begin(); j != inner_r.end(); ++j) {
                  // compute the range of columns
                  int32_t block_begin_column = begin_column + j * column_size;
                  int32_t block_end_column =
                      begin_column + (j + 1) * column_size;

                  // set block ranges for the vertex similarity algorithm
                  const auto jaccard_desc_default =
                      jaccard::descriptor<>().set_block(
                          {block_begin_row,
                           std::min(block_end_row, vertex_count)},
                          {block_begin_column,
                           std::min(block_end_column, vertex_count)});

                  // compute Jaccard coefficients for the block
                  vertex_similarity(
                      jaccard_desc_default, g,
                      static_cast<void *>(
                          (processing_blocks
                               [tbb::this_task_arena::current_thread_index()])
                              .data()));

                  // do application specific postprocessing of the result here
                }
              },
              tbb::simple_partitioner{});
        }
      },
      tbb::simple_partitioner{});
}
