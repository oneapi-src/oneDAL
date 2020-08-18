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

#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>

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

template <class Graph>
void jaccard_all_block(const Graph &g, int32_t block_i, int32_t block_j,
                       int number_of_threads) {
  int max_nnz_block = 0;
  int32_t num = get_vertex_count(g);
  // cout << num << endl;

  int32_t blocks = num / block_i;
  int32_t remain_els = num - blocks * block_i;
  int32_t delta = remain_els / blocks;
  int32_t tail = remain_els - blocks * delta;

  // EdgeID_t jac_size = 0;

  // int number_of_threads = tbb::this_task_arena::max_concurrency();

  std::vector<std::vector<byte_t>> processing_blocks(
      tbb::this_task_arena::max_concurrency(),
      std::vector<byte_t>(block_i * block_j * 3 * 2 * 4));
  // auto processing_blocks =
  //     std::shared_ptr<std::weak_ptr<byte_t>>(new
  //     std::weak_ptr<byte_t>[tbb::this_task_arena::max_concurrency()]);
  // auto processing_blocks_ptr = processing_blocks.get();

  // for (int32_t i = 0; i < tbb::this_task_arena::max_concurrency(); i++) {
  //   processing_blocks_ptr[i] = std::weak_ptr<byte_t>(new byte_t[block_i *
  //   block_j *3 * 2 * 4]);
  // }

  // cout << "tbb threads " << number_of_threads << endl;

  tbb::parallel_for(
      tbb::blocked_range<int>(0, blocks),
      [&](const tbb::blocked_range<int> &r) {
        for (int i = r.begin(); i != r.end(); ++i) {
          // for (int i = 0; i < blocks; ++i) {
          int block_size_i = block_i + delta;
          int begin_i = 0;
          int i_tail = 0;
          int32_t shift = i * block_size_i;
          if (i >= blocks - tail) {
            block_size_i = block_i + delta + 1;
            begin_i = (blocks - tail) * (block_i + delta);
            i_tail = blocks - tail;
            shift = (blocks - tail) * (block_i + delta) +
                    (i - (blocks - tail)) * block_size_i;
          }
          int32_t num_sh = (num - shift);
          int blocks_j = num_sh / block_j;
          int remain_els_j = num_sh - blocks_j * block_j;
          int delta_j = 0;
          if (blocks_j > 0) {
            delta_j = remain_els_j / blocks_j;
          } else {
            // Mutex.lock();
            // cout << begin_i + (i - i_tail) * block_size_i << " " << begin_i +
            // (i - i_tail + 1) * block_size_i << " : " <<
            //   begin_i + (i - i_tail) * block_size_i << " " << (int32_t)num
            //   << endl;

            const auto jaccard_desc_default = jaccard::descriptor<>().set_block(
                {begin_i + (i - i_tail) * block_size_i,
                 begin_i + (i - i_tail + 1) * block_size_i},
                {begin_i + (i - i_tail) * block_size_i, num});

            auto result_vertex_similarity = vertex_similarity(
                jaccard_desc_default, g,
                static_cast<void *>(
                    (processing_blocks
                         [tbb::this_task_arena::current_thread_index()])
                        .data()));

            // auto jaccard_coeffs = result_vertex_similarity.get_coeffs();
            // auto vertex_pairs = result_vertex_similarity.get_vertex_pairs();
            // auto nonzero_coeff_count =
            // result_vertex_similarity.get_nonzero_coeff_count();
            // print_revert_vertex_similarity_result(vertex_pairs,
            // jaccard_coeffs, nonzero_coeff_count);
          }

          int tail_j = remain_els_j - blocks_j * delta_j;
          tbb::parallel_for(
              tbb::blocked_range<int>(0, blocks_j),
              [&](const tbb::blocked_range<int> &inner_r) {
                for (int j = inner_r.begin(); j != inner_r.end(); ++j) {
                  // for (int j = 0; j < blocks_j ; ++j) {
                  int block_size_j = block_j + delta_j;
                  int begin_j = shift;
                  int j_tail = 0;
                  int block_size_j_end = block_j + delta_j;
                  if (j >= blocks_j - tail_j) {
                    block_size_j = block_j + delta_j + 1;
                    begin_j = (blocks_j - tail_j) * (block_j + delta_j) + shift;
                    j_tail = blocks_j - tail_j;
                    block_size_j_end = block_size_j;
                  }
                  // // Mutex.lock();
                  // cout << begin_i + (i - i_tail) * block_size_i << " " <<
                  // begin_i + (i - i_tail + 1) * block_size_i << " : " <<
                  //       begin_j + (j - j_tail) * block_size_j << " " <<
                  //       begin_j + (j - j_tail) * block_size_j +
                  //       block_size_j_end << endl;
                  const auto jaccard_desc_default =
                      jaccard::descriptor<>().set_block(
                          {begin_i + (i - i_tail) * block_size_i,
                           begin_i + (i - i_tail + 1) * block_size_i},
                          {begin_j + (j - j_tail) * block_size_j,
                           begin_j + (j - j_tail) * block_size_j +
                               block_size_j_end});

                  auto result_vertex_similarity = vertex_similarity(
                      jaccard_desc_default, g,
                      static_cast<void *>(
                          (processing_blocks
                               [tbb::this_task_arena::current_thread_index()])
                              .data()));

                  // auto jaccard_coeffs =
                  // result_vertex_similarity.get_coeffs(); auto vertex_pairs =
                  // result_vertex_similarity.get_vertex_pairs(); auto
                  // nonzero_coeff_count =
                  // result_vertex_similarity.get_nonzero_coeff_count();
                  // print_revert_vertex_similarity_result(vertex_pairs,
                  // jaccard_coeffs, nonzero_coeff_count);
                }
              },
              tbb::simple_partitioner{});
        }
      },
      tbb::simple_partitioner{});
}

template <class Graph>
void jaccard_all_block_v(const Graph &g, int32_t block_i, int32_t block_j,
                         int number_of_threads);

int main(int argc, char **argv) {
  // read the graph
  if (argc < 2)
    return 0;
  std::string filename = argv[1];
  csv_data_source ds(filename);
  load_graph::descriptor<> d;
  auto my_graph = load_graph::load(d, ds);
  //  cout << "edges number " << get_edge_count( my_graph) << endl;

  int num_trials_custom = 10;
  int num_trials_lib = 1;
  int verify = 0;

  int32_t block_size_x = 1;
  int32_t block_size_y = 1024;

  int32_t tbb_threads_number = 1;

  // const auto jaccard_desc_default = jaccard::descriptor<>().set_block(
  //   {0, 1},
  //   {0, 20});

  // std::vector<byte_t> jaccard_tmp(2048 *3 * 2 * 4);
  // auto result_vertex_similarity =
  //   vertex_similarity(jaccard_desc_default, my_graph,
  //           static_cast<void *>(jaccard_tmp.data()));

  //   auto jaccard_coeffs = result_vertex_similarity.get_coeffs();
  //   auto vertex_pairs = result_vertex_similarity.get_vertex_pairs();
  //   auto nonzero_coeff_count =
  //   result_vertex_similarity.get_nonzero_coeff_count();
  //   print_revert_vertex_similarity_result(vertex_pairs, jaccard_coeffs,
  //   nonzero_coeff_count);
  // std::cout << "Vertex pairs: " << std::endl;
  // print_vertex_similarity_result(vertex_pairs, nonzero_coeff_count);

  // //std::cout << "Jaccard values: " << std::endl;
  // print_vertex_similarity_result(jaccard_coeffs, nonzero_coeff_count);

  if (argc > 6) {
    block_size_x = stoi(argv[2]);
    block_size_y = stoi(argv[3]);
    num_trials_custom = stoi(argv[4]);
    tbb_threads_number = stoi(argv[5]);
    verify = stoi(argv[6]);
  }
  tbb::global_control c(tbb::global_control::max_allowed_parallelism,
                        tbb_threads_number);
  // cout << "jaccard_non_existent with custom block_size = " << block_size_x
  // <<"x" <<block_size_y << endl;
  vector<double> time;
  double median;

  for (int i = 0; i < num_trials_custom; i++) {
    auto start = high_resolution_clock::now();
    jaccard_all_block(
        my_graph, block_size_x, block_size_y,
        c.active_value(tbb::global_control::max_allowed_parallelism));
    auto stop = high_resolution_clock::now();

    time.push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(stop - start)
            .count());
    cout << i << " iter: " << time.back() << endl;
  }

  // if (verify) {
  //     jaccard_all_block(my_graph, block_size_x, block_size_y,
  //         c.active_value(tbb::global_control::max_allowed_parallelism));
  // }

  return 0;
}

class jaccard_pair {
public:
  int first;
  int second;
  float coefficient;

  jaccard_pair() {
    first = 0;
    second = 0;
    coefficient = 0;
  };
  jaccard_pair(int _first, int _second, float _coefficient) {
    first = _first;
    second = _second;
    coefficient = _coefficient;
  };
  //~jaccard_pair();
};

#include <fstream>

template <class Graph>
void jaccard_all_block_v(const Graph &my_graph, int32_t block_i,
                         int32_t block_j, int number_of_threads) {
  int max_nnz_block = 0;
  int32_t num = get_vertex_count(my_graph);
  // cout << num << endl;

  int32_t blocks = num / block_i;
  int32_t remain_els = num - blocks * block_i;
  int32_t delta = remain_els / blocks;
  int32_t tail = remain_els - blocks * delta;

  cout << "tbb threads " << number_of_threads << endl;
  ///*
  size_t num_non_exists = (num * num - num) / 2 - get_edge_count(my_graph);
  int64_t jaccard_size = 0;
  int32_t ratio_all_coefs_with_nnz_coeffs = 7;
  std::vector<jaccard_pair> jaccard_custom(num_non_exists /
                                           ratio_all_coefs_with_nnz_coeffs);
  int jaccard_custom_count = 0;

  std::mutex Mutex;

  cout << "tbb threads " << number_of_threads << endl;
  // EdgeID_t jac_size = 0;

  // int number_of_threads = tbb::this_task_arena::max_concurrency();

  std::vector<std::vector<byte_t>> processing_blocks(
      tbb::this_task_arena::max_concurrency(),
      std::vector<byte_t>(block_i * block_j * 3 * 2 * 4));
  // auto processing_blocks =
  //     std::shared_ptr<std::weak_ptr<byte_t>>(new
  //     std::weak_ptr<byte_t>[tbb::this_task_arena::max_concurrency()]);
  // auto processing_blocks_ptr = processing_blocks.get();

  // for (int32_t i = 0; i < tbb::this_task_arena::max_concurrency(); i++) {
  //   processing_blocks_ptr[i] = std::weak_ptr<byte_t>(new byte_t[block_i *
  //   block_j *3 * 2 * 4]);
  // }

  cout << "tbb threads " << number_of_threads << endl;

  tbb::parallel_for(
      tbb::blocked_range<int>(0, blocks),
      [&](const tbb::blocked_range<int> &r) {
        for (int i = r.begin(); i != r.end(); ++i) {
          // for (int i = 0; i < blocks; ++i) {
          int block_size_i = block_i + delta;
          int begin_i = 0;
          int i_tail = 0;
          int32_t shift = i * block_size_i;
          if (i >= blocks - tail) {
            block_size_i = block_i + delta + 1;
            begin_i = (blocks - tail) * (block_i + delta);
            i_tail = blocks - tail;
            shift = (blocks - tail) * (block_i + delta) +
                    (i - (blocks - tail)) * block_size_i;
          }
          int32_t num_sh = (num - shift);
          int blocks_j = num_sh / block_j;
          int remain_els_j = num_sh - blocks_j * block_j;
          int delta_j = 0;
          if (blocks_j > 0) {
            delta_j = remain_els_j / blocks_j;
          } else {
            // Mutex.lock();
            // cout << begin_i + (i - i_tail) * block_size_i << " " << begin_i +
            // (i - i_tail + 1) * block_size_i << " : " <<
            //   begin_i + (i - i_tail) * block_size_i << " " << (int32_t)num
            //   << endl;

            const auto jaccard_desc_default = jaccard::descriptor<>().set_block(
                {begin_i + (i - i_tail) * block_size_i,
                 begin_i + (i - i_tail + 1) * block_size_i},
                {begin_i + (i - i_tail) * block_size_i, num});

            auto result_vertex_similarity = vertex_similarity(
                jaccard_desc_default, my_graph,
                static_cast<void *>(
                    (processing_blocks
                         [tbb::this_task_arena::current_thread_index()])
                        .data()));

            Mutex.lock();
            auto jaccard_coeffs = result_vertex_similarity.get_coeffs();
            auto vertex_pairs = result_vertex_similarity.get_vertex_pairs();
            auto nonzero_coeff_count =
                result_vertex_similarity.get_nonzero_coeff_count();
            // print_revert_vertex_similarity_result(jaccard_coeffs,
            // vertex_pairs, nonzero_coeff_count);

            auto arr1 =
                oneapi::dal::row_accessor<const float>(vertex_pairs).pull();
            auto arr2 =
                oneapi::dal::row_accessor<const float>(jaccard_coeffs).pull();

            const auto x1 = arr1.get_data();
            const auto x2 = arr2.get_data();

            for (std::int64_t i = 0; i < nonzero_coeff_count; i++) {
              if (x1[i] < x1[2 + i]) {
                jaccard_custom[jaccard_custom_count].first = x1[i];
                jaccard_custom[jaccard_custom_count].second =
                    x1[vertex_pairs.get_column_count() + i];
                jaccard_custom[jaccard_custom_count].coefficient = x2[i];
                jaccard_custom_count++;
              }
            }
            Mutex.unlock();
          }

          int tail_j = remain_els_j - blocks_j * delta_j;
          tbb::parallel_for(
              tbb::blocked_range<int>(0, blocks_j),
              [&](const tbb::blocked_range<int> &inner_r) {
                for (int j = inner_r.begin(); j != inner_r.end(); ++j) {
                  // for (int j = 0; j < blocks_j ; ++j) {
                  int block_size_j = block_j + delta_j;
                  int begin_j = shift;
                  int j_tail = 0;
                  int block_size_j_end = block_j + delta_j;
                  if (j >= blocks_j - tail_j) {
                    block_size_j = block_j + delta_j + 1;
                    begin_j = (blocks_j - tail_j) * (block_j + delta_j) + shift;
                    j_tail = blocks_j - tail_j;
                    block_size_j_end = block_size_j;
                  }
                  // // Mutex.lock();
                  // cout << begin_i + (i - i_tail) * block_size_i << " " <<
                  // begin_i + (i - i_tail + 1) * block_size_i << " : " <<
                  //       begin_j + (j - j_tail) * block_size_j << " " <<
                  //       begin_j + (j - j_tail) * block_size_j +
                  //       block_size_j_end << endl;
                  const auto jaccard_desc_default =
                      jaccard::descriptor<>().set_block(
                          {begin_i + (i - i_tail) * block_size_i,
                           begin_i + (i - i_tail + 1) * block_size_i},
                          {begin_j + (j - j_tail) * block_size_j,
                           begin_j + (j - j_tail) * block_size_j +
                               block_size_j_end});

                  auto result_vertex_similarity = vertex_similarity(
                      jaccard_desc_default, my_graph,
                      static_cast<void *>(
                          (processing_blocks
                               [tbb::this_task_arena::current_thread_index()])
                              .data()));

                  Mutex.lock();
                  auto jaccard_coeffs = result_vertex_similarity.get_coeffs();
                  auto vertex_pairs =
                      result_vertex_similarity.get_vertex_pairs();
                  auto nonzero_coeff_count =
                      result_vertex_similarity.get_nonzero_coeff_count();
                  // print_revert_vertex_similarity_result(jaccard_coeffs,
                  // vertex_pairs, nonzero_coeff_count);

                  auto arr1 =
                      oneapi::dal::row_accessor<const float>(vertex_pairs)
                          .pull();
                  auto arr2 =
                      oneapi::dal::row_accessor<const float>(jaccard_coeffs)
                          .pull();

                  const auto x1 = arr1.get_data();
                  const auto x2 = arr2.get_data();

                  for (std::int64_t i = 0; i < nonzero_coeff_count; i++) {
                    if (x1[i] < x1[2 + i]) {
                      jaccard_custom[jaccard_custom_count].first = x1[i];
                      jaccard_custom[jaccard_custom_count].second =
                          x1[vertex_pairs.get_column_count() + i];
                      jaccard_custom[jaccard_custom_count].coefficient = x2[i];
                      jaccard_custom_count++;
                    }
                  }
                  Mutex.unlock();
                }
              },
              tbb::simple_partitioner{});
        }
      },
      tbb::simple_partitioner{});

  jaccard_custom.resize(jaccard_custom_count);

  sort(jaccard_custom.begin(), jaccard_custom.end(),
       [](const auto &lhs, const auto &rhs) {
         if (lhs.first < rhs.first) {
           return true;
         } else if (lhs.first == rhs.first) {
           if (lhs.second < rhs.second) {
             return true;
           } else
             return false;
         } else {
           return false;
         }
       });

  vector<jaccard_pair> jaccard_lib(0);
  ifstream myfile;
  myfile.open("/nfs/inn/proj/numerics1/Users/akumin/oneDAL/results_Enron.txt");
  if (myfile.is_open()) {
    int size_jaccard = 0;
    myfile >> size_jaccard;
    jaccard_lib.resize(size_jaccard);
    for (int i = 0; i < jaccard_lib.size(); i++) {
      myfile >> jaccard_lib[i].first >> jaccard_lib[i].second >>
          jaccard_lib[i].coefficient;
    }
    myfile.close();
  }

  cout << " jaccard lib: " << jaccard_lib.size() << endl;
  cout << " jaccard custom: " << jaccard_custom.size() << endl;

  // auto layout = my_graph.get_impl();
  bool is_correct = true;
  float eps = 0.000001;
  for (int i = 0; i < jaccard_lib.size(); i++) {
    if (!((jaccard_custom[i].coefficient < jaccard_lib[i].coefficient + eps) &&
          (jaccard_custom[i].coefficient > jaccard_lib[i].coefficient - eps))) {
      cout << jaccard_lib[i].first << " " << jaccard_lib[i].second
           << " jaccard_lib: " << jaccard_lib[i].coefficient << endl;
      cout << jaccard_custom[i].first << " " << jaccard_custom[i].second
           << " jaccard: " << jaccard_custom[i].coefficient << endl;

      // cout << "Neighbors " << jaccard_lib[i].first <<" "
      // <<layout->_degrees[jaccard_lib[i].first] <<" : ";

      // is_correct = false;
      // for (int j = layout->_edge_offsets[jaccard_lib[i].first]; j <
      // layout->_edge_offsets[jaccard_lib[i].first + 1]; j++)
      // {
      //     cout << layout->_vertex_neighbors[j] << " ";
      // }
      // cout << endl;

      // cout << "Neighbors " << jaccard_lib[i].second << " : ";
      // is_correct = false;
      // for (int j = layout->_edge_offsets[jaccard_lib[i].second]; j <
      // layout->_edge_offsets[jaccard_lib[i].second + 1]; j++)
      // {
      //     cout << layout->_vertex_neighbors[j] << " ";
      // }
      // cout << endl;

      // cout << "Neighbors " << jaccard[i].first << " : ";
      // is_correct = false;
      // for (int j = layout->_edge_offsets[jaccard[i].first]; j <
      // layout->_edge_offsets[jaccard[i].first + 1]; j++)
      // {
      //     cout << layout->_vertex_neighbors[j] << " ";
      // }
      // cout << endl;

      // cout << "Neighbors " << jaccard[i].second << " : ";

      // is_correct = false;
      // for (int j = layout->_edge_offsets[jaccard[i].second]; j <
      // layout->_edge_offsets[jaccard[i].second + 1]; j++)
      // {
      //     cout << layout->_vertex_neighbors[j] << " ";
      // }
      // cout << endl;

      for (int j = max(0, i - 5); j < i + 5; j++) {
        cout << jaccard_custom[j].first << " " << jaccard_custom[j].second
             << " jaccard: " << jaccard_custom[j].coefficient << "            ";

        cout << jaccard_lib[j].first << " " << jaccard_lib[j].second
             << " jaccard_lib: " << jaccard_lib[j].coefficient << endl;
      }

      break;
    }
  }
  if (is_correct == true) {
    cout << "Succesfull." << endl;
  } else {
    cout << "Failed." << endl;
  }
}
