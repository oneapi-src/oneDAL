// /*******************************************************************************
//  * Copyright 2020 Intel Corporation
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  *     http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  *******************************************************************************/

// #include "oneapi/dal/algo/jaccard.hpp"
// #include <iostream>
// #include "oneapi/dal/data/graph.hpp"
// #include "oneapi/dal/data/table.hpp"
// #include "oneapi/dal/util/csv_data_source.hpp"
// #include "oneapi/dal/util/load_graph.hpp"
// #include "oneapi/dal/exceptions.hpp"
//  #include "tbb/global_control.h"

// using namespace oneapi::dal;
// using namespace oneapi::dal::preview;

// #include <iostream>
// //#include "graph.hpp"
// //#include "utility.hpp"
// #include <chrono> 
// #include <stdexcept>
// #include <set>
// #include <mutex>
// #include <algorithm>

// #include "tbb/parallel_for.h"
// #include "tbb/parallel_reduce.h"
// #include "tbb/partitioner.h"
// #include "tbb/parallel_sort.h"
// #include "tbb/task_scheduler_init.h"

// #include "oneapi/dal/algo/jaccard.hpp"
// #include "oneapi/dal/data/undirected_adjacency_array_graph.hpp"
// #include "oneapi/dal/data/table.hpp"
// #include "oneapi/dal/util/csv_data_source.hpp"
// #include "oneapi/dal/util/load_graph.hpp"

// using namespace std;
// using namespace std::chrono;


// template<class NodeID_t>
// void jaccard_all_block(
//                 const graph& g, NodeID_t block_i, NodeID_t block_j,
//                 std::vector<NodeID_t>& jaccard_first,
//                 std::vector<NodeID_t>& jaccard_second,
//                 std::vector<float>& jaccard_coefficients, int number_of_threads)
// {
//     int max_nnz_block = 0;
//     NodeID_t num = get_vertex_count(g);
//     //cout << num << endl;

//     NodeID_t blocks = num / block_i;
//     NodeID_t remain_els = num - blocks * block_i;
//     NodeID_t delta = remain_els / blocks;
//     NodeID_t tail = remain_els - blocks * delta;

//     //EdgeID_t jac_size = 0;

//     //int number_of_threads = tbb::this_task_arena::max_concurrency();
//     cout << "tbb threads " << number_of_threads << endl;
//     std::vector< std::vector<NodeID_t>> blocks_first(tbb::this_task_arena::max_concurrency(), std::vector<NodeID_t> (block_i * block_j * 2));
//     std::vector< std::vector<NodeID_t>> blocks_second(tbb::this_task_arena::max_concurrency(), std::vector<NodeID_t> (block_i * block_j * 2));
//     std::vector< std::vector<float>> blocks_jaccards(tbb::this_task_arena::max_concurrency(), std::vector<float> (block_i * block_j * 2));


//     tbb::parallel_for(tbb::blocked_range<int>(0, blocks),
//         [&](const tbb::blocked_range<int>& r) {
//             for (int i = r.begin(); i != r.end(); ++i) {
//             //for (int i = 0; i < blocks; ++i) {
//                 int block_size_i = block_i + delta; int begin_i = 0; int i_tail = 0;
//                 NodeID_t shift = i * block_size_i;
//                 if (i >= blocks - tail) {
//                     block_size_i = block_i + delta + 1; begin_i = (blocks - tail) * (block_i + delta); i_tail = blocks - tail;
//                     shift = (blocks - tail) * (block_i + delta) + (i - (blocks - tail)) * block_size_i;
//                 }
//                 NodeID_t num_sh = (num - shift);
//                 int blocks_j = num_sh / block_j;
//                 int remain_els_j = num_sh - blocks_j * block_j;
//                 int delta_j = 0;
//                 if (blocks_j > 0) {
//                     delta_j = remain_els_j / blocks_j;
//                 }
//                 else { 
//                     const auto jaccard_desc_default = jaccard::descriptor<>().set_block(
//                         {begin_i + (i - i_tail) * block_size_i, begin_i + (i - i_tail + 1) * block_size_i},
//                         {begin_i + (i - i_tail) * block_size_i, (NodeID_t)num});
//                     NodeID_t jaccard_block_nnz = 0;
//                     auto result_default = vertex_similarity(jaccard_desc_default, g);
//                 }

//                 int tail_j = remain_els_j - blocks_j * delta_j;
//                 tbb::parallel_for(tbb::blocked_range<int>(0, blocks_j),
//                     [&](const tbb::blocked_range<int>& inner_r) {
//                         for (int j = inner_r.begin(); j != inner_r.end(); ++j) {
//                         //for (int j = 0; j < blocks_j ; ++j) {
//                             int block_size_j = block_j + delta_j; int begin_j = shift; int j_tail = 0; int block_size_j_end = block_j + delta_j;
//                             if (j >= blocks_j - tail_j) {
//                                 block_size_j = block_j + delta_j + 1;
//                                 begin_j = (blocks_j - tail_j) * (block_j + delta_j) + shift;
//                                 j_tail = blocks_j - tail_j; block_size_j_end = block_size_j;
//                             }

//                                 NodeID_t jaccard_block_nnz = 0;
//                                 //Mutex.lock();
//                                  //cout << begin_i + (i - i_tail) * block_size_i << " " << begin_i + (i - i_tail + 1) * block_size_i << " : " <<
//                                   //       begin_j + (j - j_tail) * block_size_j << " " << begin_j + (j - j_tail) * block_size_j + block_size_j_end << endl;
//                         const auto jaccard_desc_default = jaccard::descriptor<>().set_block(
//                         {begin_i + (i - i_tail) * block_size_i, begin_i + (i - i_tail + 1) * block_size_i},
//                         {begin_j + (j - j_tail) * block_size_j, begin_j + (j - j_tail) * block_size_j + block_size_j_end});
//                         auto result_default = vertex_similarity(jaccard_desc_default, g);

//                         }}, tbb::simple_partitioner{});
//                 }}, tbb::simple_partitioner{});

// }

// int main(int argc, char ** argv)
// {

//     //tbb::task_scheduler_init init(32);

//     int num_trials_custom = 10;
//     int num_trials_lib = 1;
//     int verify = 0;


//     if (argc < 2) return 0;
//     std::string filename = argv[1];
//     csv_data_source ds(filename);
//     load_graph::descriptor<> d;
//     auto my_graph = load_graph::load(d, ds);
//     cout << "edges number" << get_edge_count( my_graph);

//     int32_t block_size_x = 1;
//     int32_t block_size_y = 1024;

//     int32_t tbb_threads_number = 1;

//     if (argc > 6) {
//         block_size_x = stoi(argv[2]);
//         block_size_y = stoi(argv[3]);
//         num_trials_custom = stoi(argv[4]);
//         tbb_threads_number = stoi(argv[5]);
//         verify = stoi(argv[6]);
//     }

//     tbb::global_control c(tbb::global_control::max_allowed_parallelism, tbb_threads_number);
//  //   tbb::task_scheduler_init init(num_of_threads);

// //     std::vector<jaccard_pair<int32_t>> jaccard;

// ///*
//     std::vector<int32_t> jaccard_first;
//     std::vector<int32_t> jaccard_second;
//     std::vector<float> jaccard_coefficients;
// //*/
//     cout << "jaccard_non_existent with custom block_size = " << block_size_x <<"x" <<block_size_y << endl;
//     vector<double> time;
//     double median;

//     for(int i = 0; i < num_trials_custom; i++) {
//         auto start = high_resolution_clock::now();
//         //jaccard_all_row(my_graph, jaccard_first, jaccard_second, jaccard_coefficients, block_size_x, block_size_y);
//         jaccard_all_block(my_graph, block_size_x, block_size_y, 
//             jaccard_first, jaccard_second, jaccard_coefficients,
//             c.active_value(tbb::global_control::max_allowed_parallelism));
//         //jaccard_status = jaccard_all_row(g, jaccard, block_size_x, block_size_y, 1);
//         auto stop = high_resolution_clock::now();

//         time.push_back(std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count());
//         cout << i << " iter: " << time.back() << endl;
//     }


//     cout <<endl;

//   //  computing time metrics
//     sort(time.begin(), time.end());
//     if (num_trials_custom % 2 == 0)
//         median =  (time[time.size()/2] + time[time.size()/2 - 1]) / 2;
//     else
//         median = time[time.size()/2];
//     cout <<"median: " << median << endl;
//     cout <<"Min: " << time[0] << endl;
//     cout <<"Max: " << time.back() << endl;
//     return 0;
// }


#include "oneapi/dal/algo/jaccard.hpp"
//#include "oneapi/dal/data/undirected_adjacency_array_graph.hpp"
#include "oneapi/dal/data/table.hpp"
#include "oneapi/dal/util/csv_data_source.hpp"
#include "oneapi/dal/util/load_graph.hpp"

#include "oneapi/dal/data/graph.hpp"
 #include "tbb/global_control.h"

using namespace oneapi::dal;
using namespace oneapi::dal::preview;

#include <iostream>
#include <chrono> 
#include <stdexcept>
#include <set>
#include <mutex>
#include <algorithm>

#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
#include "tbb/partitioner.h"
#include "tbb/parallel_sort.h"
#include "tbb/task_scheduler_init.h"


using namespace std;
using namespace std::chrono;

void jaccard_all_block(const graph& my_graph, int block_i, int block_j, int number_of_threads)
{
    int num = get_vertex_count(my_graph);

    int blocks = num / block_i;
    int remain_els = num - blocks * block_i;
    int delta = remain_els / blocks;
    int tail = remain_els - blocks * delta;

    //tbb::parallel_for(tbb::blocked_range<int>(0, blocks),
        //[&](const tbb::blocked_range<int>& r) {
            //for (int i = r.begin(); i != r.end(); ++i) {
            for (int i = 0; i < blocks; ++i) {
                int block_size_i = block_i + delta; int begin_i = 0; int i_tail = 0;
                int shift = i * block_size_i;
                if (i >= blocks - tail) {
                    block_size_i = block_i + delta + 1; begin_i = (blocks - tail) * (block_i + delta); i_tail = blocks - tail;
                    shift = (blocks - tail) * (block_i + delta) + (i - (blocks - tail)) * block_size_i;
                }
                int num_sh = (num - shift);
                int blocks_j = num_sh / block_j;
                int remain_els_j = num_sh - blocks_j * block_j;
                int delta_j = 0;
                if (blocks_j > 0) {
                    delta_j = remain_els_j / blocks_j;
                }
                else { 
                    const auto jaccard_desc_default = jaccard::descriptor<>().set_block(
                        {begin_i + (i - i_tail) * block_size_i, begin_i + (i - i_tail + 1) * block_size_i},
                        {begin_i + (i - i_tail) * block_size_i, (int)num});
                    int jaccard_block_nnz = 0;
                    vertex_similarity(jaccard_desc_default, my_graph);
                }

                int tail_j = remain_els_j - blocks_j * delta_j;
                //tbb::parallel_for(tbb::blocked_range<int>(0, blocks_j),
                    //[&](const tbb::blocked_range<int>& inner_r) {
                        //for (int j = inner_r.begin(); j != inner_r.end(); ++j) {
                        for (int j = 0; j < blocks_j ; ++j) {
                            int block_size_j = block_j + delta_j; int begin_j = shift; int j_tail = 0; int block_size_j_end = block_j + delta_j;
                            if (j >= blocks_j - tail_j) {
                                block_size_j = block_j + delta_j + 1;
                                begin_j = (blocks_j - tail_j) * (block_j + delta_j) + shift;
                                j_tail = blocks_j - tail_j; block_size_j_end = block_size_j;
                            }

                                int jaccard_block_nnz = 0;
                                //Mutex.lock();
                                 //cout << begin_i + (i - i_tail) * block_size_i << " " << begin_i + (i - i_tail + 1) * block_size_i << " : " <<
                                  //       begin_j + (j - j_tail) * block_size_j << " " << begin_j + (j - j_tail) * block_size_j + block_size_j_end << endl;
                        const auto jaccard_desc_default = jaccard::descriptor<>().set_block(
                        {begin_i + (i - i_tail) * block_size_i, begin_i + (i - i_tail + 1) * block_size_i},
                        {begin_j + (j - j_tail) * block_size_j, begin_j + (j - j_tail) * block_size_j + block_size_j_end});
                        vertex_similarity(jaccard_desc_default, my_graph);

                        }//}, tbb::simple_partitioner{});
                }//}, tbb::simple_partitioner{});

}

int main(int argc, char **argv) {
   int num_trials_custom = 10;
    int num_trials_lib = 1;
    int verify = 0;


    if (argc < 2) return 0;
    std::string filename = argv[1];
    csv_data_source ds(filename);
    load_graph::descriptor<> d;
    auto my_graph = load_graph::load(d, ds);
    cout << "edges number " << get_edge_count( my_graph) << endl;

    int32_t block_size_x = 1;
    int32_t block_size_y = 1024;

    int32_t tbb_threads_number = 1;

    if (argc > 6) {
        block_size_x = stoi(argv[2]);
        block_size_y = stoi(argv[3]);
        num_trials_custom = stoi(argv[4]);
        tbb_threads_number = stoi(argv[5]);
        verify = stoi(argv[6]);
    }

    tbb::global_control c(tbb::global_control::max_allowed_parallelism, tbb_threads_number);
    cout << "jaccard_non_existent with custom block_size = " << block_size_x <<"x" <<block_size_y << endl;
    vector<double> time;
    double median;

    for(int p = 0; p < num_trials_custom; p++) {
        auto start = high_resolution_clock::now();
        //jaccard_all_row(my_graph, jaccard_first, jaccard_second, jaccard_coefficients, block_size_x, block_size_y);
        jaccard_all_block(my_graph, block_size_x,block_size_y,1);
        //jaccard_status = jaccard_all_row(g, jaccard, block_size_x, block_size_y, 1);
            int block_i = block_size_x;
    int block_j = block_size_y;
    int num = get_vertex_count(my_graph);

    int blocks = num / block_i;
    int remain_els = num - blocks * block_i;
    int delta = remain_els / blocks;
    int tail = remain_els - blocks * delta;

    //tbb::parallel_for(tbb::blocked_range<int>(0, blocks),
        //[&](const tbb::blocked_range<int>& r) {
            //for (int i = r.begin(); i != r.end(); ++i) {
            for (int i = 0; i < blocks; ++i) {
                int block_size_i = block_i + delta; int begin_i = 0; int i_tail = 0;
                int shift = i * block_size_i;
                if (i >= blocks - tail) {
                    block_size_i = block_i + delta + 1; begin_i = (blocks - tail) * (block_i + delta); i_tail = blocks - tail;
                    shift = (blocks - tail) * (block_i + delta) + (i - (blocks - tail)) * block_size_i;
                }
                int num_sh = (num - shift);
                int blocks_j = num_sh / block_j;
                int remain_els_j = num_sh - blocks_j * block_j;
                int delta_j = 0;
                if (blocks_j > 0) {
                    delta_j = remain_els_j / blocks_j;
                }
                else { 
                    const auto jaccard_desc_default = jaccard::descriptor<>().set_block(
                        {begin_i + (i - i_tail) * block_size_i, begin_i + (i - i_tail + 1) * block_size_i},
                        {begin_i + (i - i_tail) * block_size_i, (int)num});
                    int jaccard_block_nnz = 0;
                    vertex_similarity(jaccard_desc_default, my_graph);
                }

                int tail_j = remain_els_j - blocks_j * delta_j;
                //tbb::parallel_for(tbb::blocked_range<int>(0, blocks_j),
                    //[&](const tbb::blocked_range<int>& inner_r) {
                        //for (int j = inner_r.begin(); j != inner_r.end(); ++j) {
                        for (int j = 0; j < blocks_j ; ++j) {
                            int block_size_j = block_j + delta_j; int begin_j = shift; int j_tail = 0; int block_size_j_end = block_j + delta_j;
                            if (j >= blocks_j - tail_j) {
                                block_size_j = block_j + delta_j + 1;
                                begin_j = (blocks_j - tail_j) * (block_j + delta_j) + shift;
                                j_tail = blocks_j - tail_j; block_size_j_end = block_size_j;
                            }

                                int jaccard_block_nnz = 0;
                                //Mutex.lock();
                                 //cout << begin_i + (i - i_tail) * block_size_i << " " << begin_i + (i - i_tail + 1) * block_size_i << " : " <<
                                  //       begin_j + (j - j_tail) * block_size_j << " " << begin_j + (j - j_tail) * block_size_j + block_size_j_end << endl;
                        const auto jaccard_desc_default = jaccard::descriptor<>().set_block(
                        {begin_i + (i - i_tail) * block_size_i, begin_i + (i - i_tail + 1) * block_size_i},
                        {begin_j + (j - j_tail) * block_size_j, begin_j + (j - j_tail) * block_size_j + block_size_j_end});
                        vertex_similarity(jaccard_desc_default, my_graph);

                        }//}, tbb::simple_partitioner{});
                }//}, tbb::simple_partitioner{});

        auto stop = high_resolution_clock::now();

        time.push_back(std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count());
        cout << p << " iter: " << time.back() << endl;
    }

  //  computing time metrics
    sort(time.begin(), time.end());
    if (num_trials_custom % 2 == 0)
        median =  (time[time.size()/2] + time[time.size()/2 - 1]) / 2;
    else
        median = time[time.size()/2];
    cout <<"median: " << median << endl;
    cout <<"Min: " << time[0] << endl;
    cout <<"Max: " << time.back() << endl;
    return 0;
}