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

#include "oneapi/dal/algo/jaccard.hpp"
#include <iostream>
#include "oneapi/dal/data/graph.hpp"
#include "oneapi/dal/data/table.hpp"
#include "oneapi/dal/util/csv_data_source.hpp"
#include "oneapi/dal/util/load_graph.hpp"

using namespace oneapi::dal;
using namespace oneapi::dal::preview;

#include <iostream>
//#include "graph.hpp"
//#include "utility.hpp"
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

template<class NodeID_t>
__forceinline size_t  intersection_avx512(NodeID_t* neigh_u, NodeID_t* neigh_v, NodeID_t n_u, NodeID_t n_v)
{
    size_t total = 0;
    NodeID_t i_u = 0, i_v = 0;
        while (i_u < (n_u / 16) * 16 && i_v < (n_v / 16) * 16) {        // not in last n%16 elements

                       // assumes neighbor list is ordered
                NodeID_t minu = neigh_u[i_u];
                NodeID_t maxv = neigh_v[i_v + 15];
                
                                if(minu > maxv) {
                    if (minu > neigh_v[n_v - 1]) {
                        return total;
                    }
                    i_v += 16;
                    continue;
                }

                NodeID_t minv = neigh_v[i_v];
                NodeID_t maxu = neigh_u[i_u + 15]; 
                if(minv > maxu) { 
                    if (minv > neigh_u[n_u - 1]) {
                        return total;
                    }
                    i_u += 16;
                    continue; 
                }
                __m512i v_u = _mm512_loadu_si512((void*)(neigh_u + i_u)); // load 16 neighbors of u
                __m512i v_v = _mm512_loadu_si512((void*)(neigh_v + i_v)); // load 16 neighbors of v
                if (maxu >= maxv) i_v += 16;
                if (maxu <= maxv) i_u += 16;

                __mmask16 match = _mm512_cmpeq_epi32_mask(v_u, v_v);
                if (_mm512_mask2int(match) != 0xffff) {        // shortcut case where all neighbors match
                        __m512i circ1 = _mm512_set_epi32(0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);    // all possible circular shifts for 16 elements
                        __m512i circ2 = _mm512_set_epi32(1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2);
                        __m512i circ3 = _mm512_set_epi32(2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3);
                        __m512i circ4 = _mm512_set_epi32(3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4);
                        __m512i circ5 = _mm512_set_epi32(4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5);
                        __m512i circ6 = _mm512_set_epi32(5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6);
                        __m512i circ7 = _mm512_set_epi32(6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7);
                        __m512i circ8 = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);
                        __m512i circ9 = _mm512_set_epi32(8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9);
                        __m512i circ10 = _mm512_set_epi32(9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10);
                        __m512i circ11 = _mm512_set_epi32(10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11);
                        __m512i circ12 = _mm512_set_epi32(11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12);
                        __m512i circ13 = _mm512_set_epi32(12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13);
                        __m512i circ14 = _mm512_set_epi32(13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14);
                        __m512i circ15 = _mm512_set_epi32(14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15);
                        __m512i v_v1 = _mm512_permutexvar_epi32(circ1, v_v);
                        __m512i v_v2 = _mm512_permutexvar_epi32(circ2, v_v);
                        __m512i v_v3 = _mm512_permutexvar_epi32(circ3, v_v);
                        __m512i v_v4 = _mm512_permutexvar_epi32(circ4, v_v);
                        __m512i v_v5 = _mm512_permutexvar_epi32(circ5, v_v);
                        __m512i v_v6 = _mm512_permutexvar_epi32(circ6, v_v);
                        __m512i v_v7 = _mm512_permutexvar_epi32(circ7, v_v);
                        __m512i v_v8 = _mm512_permutexvar_epi32(circ8, v_v);
                        __m512i v_v9 = _mm512_permutexvar_epi32(circ9, v_v);
                        __m512i v_v10 = _mm512_permutexvar_epi32(circ10, v_v);
                        __m512i v_v11 = _mm512_permutexvar_epi32(circ11, v_v);
                        __m512i v_v12 = _mm512_permutexvar_epi32(circ12, v_v);
                        __m512i v_v13 = _mm512_permutexvar_epi32(circ13, v_v);
                        __m512i v_v14 = _mm512_permutexvar_epi32(circ14, v_v);
                        __m512i v_v15 = _mm512_permutexvar_epi32(circ15, v_v);
                        __mmask16 tmp_match1 = _mm512_cmpeq_epi32_mask(v_u, v_v1);        // find matches
                        __mmask16 tmp_match2 = _mm512_cmpeq_epi32_mask(v_u, v_v2);
                        __mmask16 tmp_match3 = _mm512_cmpeq_epi32_mask(v_u, v_v3);
                        __mmask16 tmp_match4 = _mm512_cmpeq_epi32_mask(v_u, v_v4);
                        __mmask16 tmp_match5 = _mm512_cmpeq_epi32_mask(v_u, v_v5);
                        __mmask16 tmp_match6 = _mm512_cmpeq_epi32_mask(v_u, v_v6);
                        __mmask16 tmp_match7 = _mm512_cmpeq_epi32_mask(v_u, v_v7);
                        __mmask16 tmp_match8 = _mm512_cmpeq_epi32_mask(v_u, v_v8);
                        __mmask16 tmp_match9 = _mm512_cmpeq_epi32_mask(v_u, v_v9);
                        __mmask16 tmp_match10 = _mm512_cmpeq_epi32_mask(v_u, v_v10);
                        __mmask16 tmp_match11 = _mm512_cmpeq_epi32_mask(v_u, v_v11);
                        __mmask16 tmp_match12 = _mm512_cmpeq_epi32_mask(v_u, v_v12);
                        __mmask16 tmp_match13 = _mm512_cmpeq_epi32_mask(v_u, v_v13);
                        __mmask16 tmp_match14 = _mm512_cmpeq_epi32_mask(v_u, v_v14);
                        __mmask16 tmp_match15 = _mm512_cmpeq_epi32_mask(v_u, v_v15);
                        match = _mm512_kor(_mm512_kor(_mm512_kor(_mm512_kor(match, tmp_match1), _mm512_kor(tmp_match2, tmp_match3)), _mm512_kor(_mm512_kor(tmp_match4, tmp_match5), _mm512_kor(tmp_match6, tmp_match7))), _mm512_kor(_mm512_kor(_mm512_kor(tmp_match8, tmp_match9), _mm512_kor(tmp_match10, tmp_match11)), _mm512_kor(_mm512_kor(tmp_match12, tmp_match13), _mm512_kor(tmp_match14, tmp_match15)))); // combine all matches
                    }
                total += _popcnt32(_mm512_mask2int(match));    //count number of matches
            }



            while (i_u < (n_u / 16) * 16 && i_v < n_v) {
                __m512i v_u = _mm512_loadu_si512((void*)(neigh_u + i_u));
                while (neigh_v[i_v] <= neigh_u[i_u + 15] && i_v < n_v) {
                    __m512i tmp_v_v = _mm512_set1_epi32(neigh_v[i_v]);
                    __mmask16 match = _mm512_cmpeq_epi32_mask(v_u, tmp_v_v);
                    if (_mm512_mask2int(match)) total++;
                    i_v++;
                }
                i_u += 16;
            }
            while (i_v < (n_v / 16) * 16 && i_u < n_u) {
                __m512i v_v = _mm512_loadu_si512((void*)(neigh_v + i_v));
                while (neigh_u[i_u] <= neigh_v[i_v + 15] && i_u < n_u) {
                    __m512i tmp_v_u = _mm512_set1_epi32(neigh_u[i_u]);
                    __mmask16 match = _mm512_cmpeq_epi32_mask(v_v, tmp_v_u);
                    if (_mm512_mask2int(match)) total++;
                    i_u++;
                }
                i_v += 16;
            }


            while (i_u <= (n_u - 8) && i_v <= (n_v - 8))
        {        // not in last n%8 elements
                        // assumes neighbor list is ordered
                NodeID_t minu = neigh_u[i_u];
                NodeID_t maxv = neigh_v[i_v + 7];
            

                                if(minu > maxv) {
                    if (minu > neigh_v[n_v - 1]) {
                        return total;
                    }
                    i_v += 8;
                    continue;
                }
                NodeID_t maxu = neigh_u[i_u + 7];
                NodeID_t minv = neigh_v[i_v];
                if(minv > maxu) { 
                    if (minv > neigh_u[n_u - 1]) {
                        return total;
                    }
                    i_u += 8;
                    continue; 
                }
                __m256i v_u = _mm256_loadu_epi32((void*)(neigh_u + i_u)); // load 8 neighbors of u
                __m256i v_v = _mm256_loadu_epi32((void*)(neigh_v + i_v)); // load 8 neighbors of v

                if (maxu >= maxv)
                    i_v += 8;
                if (maxu <= maxv)
                    i_u += 8;

                __mmask8 match = _mm256_cmpeq_epi32_mask(v_u, v_v);
                if (_cvtmask8_u32(match) != 0xff) {        // shortcut case where all neighbors match
                        __m256i circ1 = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);    // all possible circular shifts for 16 elements
                        __m256i circ2 = _mm256_set_epi32(1, 0, 7, 6, 5, 4, 3, 2);
                        __m256i circ3 = _mm256_set_epi32(2, 1, 0, 7, 6, 5, 4, 3);
                        __m256i circ4 = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
                        __m256i circ5 = _mm256_set_epi32(4, 3, 2, 1, 0, 7, 6, 5);
                        __m256i circ6 = _mm256_set_epi32(5, 4, 3, 2, 1, 0, 7, 6);
                        __m256i circ7 = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);

                        __m256i v_v1 = _mm256_permutexvar_epi32(circ1, v_v);
                        __m256i v_v2 = _mm256_permutexvar_epi32(circ2, v_v);
                        __m256i v_v3 = _mm256_permutexvar_epi32(circ3, v_v);
                        __m256i v_v4 = _mm256_permutexvar_epi32(circ4, v_v);
                        __m256i v_v5 = _mm256_permutexvar_epi32(circ5, v_v);
                        __m256i v_v6 = _mm256_permutexvar_epi32(circ6, v_v);
                        __m256i v_v7 = _mm256_permutexvar_epi32(circ7, v_v);

                        __mmask8 tmp_match1 = _mm256_cmpeq_epi32_mask(v_u, v_v1);        // find matches
                        __mmask8 tmp_match2 = _mm256_cmpeq_epi32_mask(v_u, v_v2);
                        __mmask8 tmp_match3 = _mm256_cmpeq_epi32_mask(v_u, v_v3);
                        __mmask8 tmp_match4 = _mm256_cmpeq_epi32_mask(v_u, v_v4);
                        __mmask8 tmp_match5 = _mm256_cmpeq_epi32_mask(v_u, v_v5);
                        __mmask8 tmp_match6 = _mm256_cmpeq_epi32_mask(v_u, v_v6);
                        __mmask8 tmp_match7 = _mm256_cmpeq_epi32_mask(v_u, v_v7);

                        match = _kor_mask8(_kor_mask8(
                            _kor_mask8(match, tmp_match1),
                            _kor_mask8(tmp_match2, tmp_match3)),
                        _kor_mask8(
                            _kor_mask8(tmp_match4, tmp_match5),
                                        _kor_mask8(tmp_match6, tmp_match7))); // combine all matches
                    }
                total += _popcnt32(_cvtmask8_u32(match));    //count number of matches
            }
            if (i_u <= (n_u - 8) && i_v < n_v)
            {
                __m256i v_u = _mm256_loadu_epi32((void*)(neigh_u + i_u));
                while (neigh_v[i_v] <= neigh_u[i_u + 7] && i_v < n_v)
                {
                    __m256i tmp_v_v = _mm256_set1_epi32(neigh_v[i_v]);
                    __mmask8 match = _mm256_cmpeq_epi32_mask(v_u, tmp_v_v);
                    if (_cvtmask8_u32(match))
                        total++;
                    i_v++;
                }
                i_u += 8;
            }
            if (i_v <= (n_v - 8) && i_u < n_u)
            {
                __m256i v_v = _mm256_loadu_epi32((void*)(neigh_v + i_v));
                while (neigh_u[i_u] <= neigh_v[i_v + 7] && i_u < n_u)
                {
                    __m256i tmp_v_u = _mm256_set1_epi32(neigh_u[i_u]);
                    __mmask8 match = _mm256_cmpeq_epi32_mask(v_v, tmp_v_u);
                    if (_cvtmask8_u32(match))
                        total++;
                    i_u++;
                }
                i_v += 8;
            }

            while (i_u <= (n_u - 4) && i_v <= (n_v - 4))
        {        // not in last n%8 elements

                        // assumes neighbor list is ordered
                NodeID_t minu = neigh_u[i_u];
                NodeID_t maxv = neigh_v[i_v + 3];
                


                                if(minu > maxv) {
                    if (minu > neigh_v[n_v - 1]) {
                        return total;
                    }
                    i_v += 4;
                    continue;
                }
                NodeID_t minv = neigh_v[i_v];
                NodeID_t maxu = neigh_u[i_u + 3];
                if(minv > maxu) { 
                    if (minv > neigh_u[n_u - 1]) {
                        return total;
                    }
                    i_u += 4;
                    continue; 
                }
                __m128i v_u = _mm_load_epi32((void*)(neigh_u + i_u)); // load 8 neighbors of u
                __m128i v_v = _mm_load_epi32((void*)(neigh_v + i_v)); // load 8 neighbors of v

                if (maxu >= maxv)
                    i_v += 4;
                if (maxu <= maxv)
                    i_u += 4;

                __mmask8 match = _mm_cmpeq_epi32_mask(v_u, v_v);
                if (_cvtmask8_u32(match) != 0xf) {        // shortcut case where all neighbors match
                    __m128i v_v1 = _mm_shuffle_epi32(v_v, _MM_SHUFFLE(0, 3, 2, 1));
                    __m128i v_v2 = _mm_shuffle_epi32(v_v, _MM_SHUFFLE(1, 0, 3, 2));
                    __m128i v_v3 = _mm_shuffle_epi32(v_v, _MM_SHUFFLE(2, 1, 0, 3));

                        __mmask8 tmp_match1 = _mm_cmpeq_epi32_mask(v_u, v_v1);        // find matches
                        __mmask8 tmp_match2 = _mm_cmpeq_epi32_mask(v_u, v_v2);
                        __mmask8 tmp_match3 = _mm_cmpeq_epi32_mask(v_u, v_v3);

                        match = _kor_mask8(_kor_mask8(match, tmp_match1), _kor_mask8(tmp_match2, tmp_match3)); // combine all matches
                    }
                total += _popcnt32(_cvtmask8_u32(match));    //count number of matches
            }
            if (i_u <= (n_u - 4) && i_v < n_v)
            {
                __m128i v_u = _mm_loadu_epi32((void*)(neigh_u + i_u));
                while (neigh_v[i_v] <= neigh_u[i_u + 3] && i_v < n_v)
                {
                    __m128i tmp_v_v = _mm_set1_epi32(neigh_v[i_v]);
                    __mmask8 match = _mm_cmpeq_epi32_mask(v_u, tmp_v_v);
                    if (_cvtmask8_u32(match))
                        total++;
                    i_v++;
                }
                i_u += 4;
            }
            if (i_v <= (n_v - 4) && i_u < n_u)
            {
                __m128i v_v = _mm_loadu_epi32((void*)(neigh_v + i_v));
                while (neigh_u[i_u] <= neigh_v[i_v + 3] && i_u < n_u)
                {
                    __m128i tmp_v_u = _mm_set1_epi32(neigh_u[i_u]);
                    __mmask8 match = _mm_cmpeq_epi32_mask(v_v, tmp_v_u);
                    if (_cvtmask8_u32(match))
                        total++;
                    i_u++;
                }
                i_v += 4;
            }


            while (i_u < n_u && i_v < n_v) {
                if ((neigh_u[i_u] > neigh_v[n_v -1]) || (neigh_v[i_v] > neigh_u[n_u - 1])) {
                    return total;
                }
                if (neigh_u[i_u] == neigh_v[i_v]) total++, i_u++, i_v++;
                else if (neigh_u[i_u] < neigh_v[i_v]) i_u++;
                else if (neigh_u[i_u] > neigh_v[i_v]) i_v++;
            }
            return total;
        }

template<class NodeID_t>
inline void jaccard_block_avx512_true(const graph& my_graph,
              NodeID_t vert00, NodeID_t vert01,
              NodeID_t vert10, NodeID_t vert11,
              std::vector<NodeID_t>& vertices_first,
              std::vector<NodeID_t>& vertices_second,
              std::vector<float>& jaccards,
              NodeID_t &jaccard_size);

template<class T> inline void Log512i(const __m512i & value)
{
    const size_t n = sizeof(__m512i) / sizeof(T);
    T buffer[n];
    _mm512_storeu_si512((__m512i*)buffer, value);
    for (int i = 0; i < n; i++)
        std::cout << buffer[i] << " ";
}

template<class T> inline void Log512(const __m512 & value)
{
    const size_t n = sizeof(__m512) / sizeof(T);
    T buffer[n];
    _mm512_store_ps((__m512*)buffer, value);
    for (int i = 0; i < n; i++)
        std::cout << buffer[i] << " ";
}


#include <immintrin.h>
template<class NodeID_t>
void jaccard_block_avx512(const graph my_graph,
              NodeID_t vert00, NodeID_t vert01,
              NodeID_t vert10, NodeID_t vert11,
              std::vector<NodeID_t>& vertices_first,
              std::vector<NodeID_t>& vertices_second,
              std::vector<float>& jaccards,
              NodeID_t &jaccard_size)
{
   auto g = oneapi::dal::preview::detail::get_impl(my_graph);
    auto g_edge_offsets = g->_edge_offsets.data();
    auto g_vertex_neighbors = g->_vertex_neighbors.data();
    auto g_degrees = g->_degrees.data();

    jaccard_size = 0;
    NodeID_t j = vert10;


    __m512i n_j_start_v = _mm512_set1_epi32(0);
    __m512i n_j_end_v = _mm512_set1_epi32(0);
    __m512i n_j_start_v1 = _mm512_set1_epi32(0);
    __m512i n_j_end_v1 = _mm512_set1_epi32(0);

    __m512i start_indices_j_v = _mm512_set1_epi32(0);
    __m512i end_indices_j_v_tmp = _mm512_set1_epi32(0);
    __m512i end_indices_j_v = _mm512_set1_epi32(0);
    
    __m512i j_vertices_tmp1 = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    
    __m512i j_vertices = _mm512_set1_epi32(0);
    __m512i j_vertices_tmp2 = _mm512_set1_epi32(0);
    int intersection_size = 0;
    __mmask16 cmpgt1 ;
    __mmask16 cmpgt2 ;
    __mmask16 worth_intersection;
    __declspec(align(64)) NodeID_t stack16_j_vertex[16] = {0};
    unsigned int ones_num = 0;

    for (NodeID_t i = vert00; i < vert01; i++) {

        NodeID_t size_i = g_degrees[i];
        auto n_i = g_vertex_neighbors+ g_edge_offsets[i];
        __m512i n_i_start_v = _mm512_set1_epi32(n_i[0]);
        __m512i n_i_end_v = _mm512_set1_epi32(n_i[size_i - 1]);
        __m512i i_vertex = _mm512_set1_epi32(i);

        //__m512i size_i_v = _mm512_set1_epi32(size_i);



        NodeID_t diagonal = std::min(i, vert11);

        for (j = vert10; j < diagonal; j++) {
            NodeID_t size_j = g_degrees[j];             
            auto n_j = g_vertex_neighbors + g_edge_offsets[j];
            

            if (!(n_i[0] > n_j[size_j -1]) && !(n_j[0] > n_i[size_i - 1])) {

                intersection_size = intersection_avx512(n_i, n_j, size_i, size_j);

                if (intersection_size) {
                    vertices_first[jaccard_size] = i;
                    vertices_second[jaccard_size] = j;
                    jaccards[jaccard_size] = static_cast<float>(intersection_size);
                    jaccard_size++;   
                }
            }
        }

        NodeID_t tmp_idx = vert10;
        if (diagonal >= vert10) {
            vertices_first[jaccard_size] = i;
            vertices_second[jaccard_size] = diagonal;
            jaccards[jaccard_size] = 1.0; 
            jaccard_size++;
            tmp_idx = diagonal + 1;
        }

        if (j < tmp_idx + ((vert11 - tmp_idx) / 16) * 16) {
            //load_data(0)
            start_indices_j_v = _mm512_load_epi32(g_edge_offsets + j);
            end_indices_j_v_tmp = _mm512_load_epi32(g_edge_offsets + j + 1);
            end_indices_j_v = _mm512_add_epi32(end_indices_j_v_tmp, _mm512_set1_epi32(-1));

            n_j_start_v = _mm512_permutevar_epi32(j_vertices_tmp1 ,_mm512_i32gather_epi32(start_indices_j_v, g_vertex_neighbors, 4));
            n_j_end_v = _mm512_permutevar_epi32(j_vertices_tmp1 ,_mm512_i32gather_epi32(end_indices_j_v, g_vertex_neighbors, 4));




            for (j; j + 16 < tmp_idx + ((vert11 - tmp_idx) / 16) * 16;) {

                start_indices_j_v = _mm512_load_epi32(g_edge_offsets + j + 16);
                end_indices_j_v_tmp = _mm512_load_epi32(g_edge_offsets + j + 17);
                end_indices_j_v = _mm512_add_epi32(end_indices_j_v_tmp, _mm512_set1_epi32(-1));


                n_j_start_v1 = _mm512_permutevar_epi32(j_vertices_tmp1 ,_mm512_i32gather_epi32(start_indices_j_v, g_vertex_neighbors, 4));
                n_j_end_v1 = _mm512_permutevar_epi32(j_vertices_tmp1 ,_mm512_i32gather_epi32(end_indices_j_v, g_vertex_neighbors, 4));


                cmpgt1 = _mm512_cmpgt_epi32_mask( n_i_start_v, n_j_end_v);
                cmpgt2 = _mm512_cmpgt_epi32_mask( n_j_start_v, n_i_end_v);

                worth_intersection = _mm512_knot(_mm512_kor(cmpgt1, cmpgt2));
                ones_num = _popcnt32(_cvtmask16_u32(worth_intersection));

                if (ones_num != 0) {
                    j_vertices_tmp2 = _mm512_set1_epi32(j);
                    j_vertices = _mm512_add_epi32(j_vertices_tmp1, j_vertices_tmp2);

                    _mm512_mask_compressstoreu_epi32((stack16_j_vertex), worth_intersection, j_vertices);

                    __declspec(align(64)) int stack16_intersections[16] = {0.0};
                    for (int s = 0; s < ones_num; s++) {
                        stack16_intersections[s] = (intersection_avx512(n_i, g_vertex_neighbors + g_edge_offsets[stack16_j_vertex[s]], size_i, g_degrees[stack16_j_vertex[s]]));           
                    }
                    __m512i intersections_v = _mm512_load_epi32(stack16_intersections);
                    j_vertices = _mm512_load_epi32(stack16_j_vertex);

                    __mmask16 non_zero_coefficients = _mm512_test_epi32_mask(intersections_v, intersections_v);//_mm512_knot(_mm512_cmpeq_ps_mask(intersections_v, _mm512_set1_ps(0.0)));
                    _mm512_mask_compressstoreu_epi32((vertices_first.data() + jaccard_size), non_zero_coefficients, i_vertex);
                    _mm512_mask_compressstoreu_epi32((vertices_second.data() + jaccard_size), non_zero_coefficients, j_vertices);
                    __m512 tmp_v = _mm512_cvtepi32_ps(intersections_v);
                    _mm512_mask_compressstoreu_ps((jaccards.data() + jaccard_size), non_zero_coefficients, tmp_v);
                    //cout << i << " " << j << endl;
                    //Log512<float>(tmp_v); cout << endl;

                    jaccard_size += _popcnt32(_cvtmask16_u32(non_zero_coefficients));

                }
                // if (i == 308 && j > 36600) {
                //     cout << " catch " << i << " " << j <<endl; 
                // }

                j += 16;

                n_j_start_v = n_j_start_v1;
                n_j_end_v = n_j_end_v1;
            }

            //process n data


            cmpgt1 = _mm512_cmpgt_epi32_mask( n_i_start_v, n_j_end_v);
            cmpgt2 = _mm512_cmpgt_epi32_mask( n_j_start_v, n_i_end_v);

            worth_intersection = _mm512_knot(_mm512_kor(cmpgt1, cmpgt2));
            ones_num = _popcnt32(_cvtmask16_u32(worth_intersection));

            if (ones_num != 0) {
                j_vertices_tmp2 = _mm512_set1_epi32(j);
                j_vertices = _mm512_add_epi32(j_vertices_tmp1, j_vertices_tmp2);

                _mm512_mask_compressstoreu_epi32((stack16_j_vertex), worth_intersection, j_vertices);

                __declspec(align(64)) int stack16_intersections[16] = {0.0};
                for (int s = 0; s < ones_num; s++) {
                    stack16_intersections[s] = (intersection_avx512(n_i, g_vertex_neighbors + g_edge_offsets[stack16_j_vertex[s]], size_i, g_degrees[stack16_j_vertex[s]]));
                    // if (intersection_size) {
                    //     vertices_first[jaccard_size] = i;
                    //     vertices_second[jaccard_size] = stack16_j_vertex[s];
                    //     jaccards[jaccard_size] = static_cast<float>(intersection_size) / static_cast<float> (size_i + g_degrees[stack16_j_vertex[s]] - intersection_size);
                    //     jaccard_size++;
                    // }                
                }
                __m512i intersections_v = _mm512_load_epi32(stack16_intersections);
                j_vertices = _mm512_load_epi32(stack16_j_vertex);
                __mmask16 non_zero_coefficients = _mm512_test_epi32_mask(intersections_v, intersections_v);//_mm512_knot(_mm512_cmpeq_ps_mask(intersections_v, _mm512_set1_ps(0.0)));
                _mm512_mask_compressstoreu_epi32((vertices_first.data() + jaccard_size), non_zero_coefficients, i_vertex);
                _mm512_mask_compressstoreu_epi32((vertices_second.data() + jaccard_size), non_zero_coefficients, j_vertices);
                __m512 tmp_v = _mm512_cvtepi32_ps(intersections_v);
                _mm512_mask_compressstoreu_ps((jaccards.data() + jaccard_size), non_zero_coefficients, tmp_v);

                jaccard_size += _popcnt32(_cvtmask16_u32(non_zero_coefficients));



            }


            j += 16;

            for (j = tmp_idx + ((vert11 - tmp_idx) / 16) * 16; j < vert11; j++) {
                NodeID_t size_j = g_degrees[j];             
                auto n_j = g_vertex_neighbors + g_edge_offsets[j];


                if (!(n_i[0] > n_j[size_j -1]) && !(n_j[0] > n_i[size_i - 1])) {

                    intersection_size = intersection_avx512(n_i, n_j, size_i, size_j);
                    //cout << i <<" " << j << " " << intersection_size << endl;

                    if (intersection_size) {
                        vertices_first[jaccard_size] = i;
                        vertices_second[jaccard_size] = j;
                        jaccards[jaccard_size] = static_cast<float>(intersection_size);
                        //cout << i <<" " << j << " " << jaccards[jaccard_size] << endl;
                        jaccard_size++;  

                    }
                }
            }
        }
        else {
            for (j = tmp_idx; j < vert11; j++) {
                NodeID_t size_j = g_degrees[j];             
                auto n_j = g_vertex_neighbors + g_edge_offsets[j];


                if (!(n_i[0] > n_j[size_j -1]) && !(n_j[0] > n_i[size_i - 1])) {

                    intersection_size = intersection_avx512(n_i, n_j, size_i, size_j);
                    //cout << i <<" " << j << " " << intersection_size << endl;

                    if (intersection_size) {
                        vertices_first[jaccard_size] = i;
                        vertices_second[jaccard_size] = j;
                        jaccards[jaccard_size] = static_cast<float>(intersection_size);
                        //cout << i <<" " << j << " " << jaccards[jaccard_size] << endl;
                        jaccard_size++;  

                    }
                }
            }
        }
    }

#pragma vector always
    for (int i = 0; i < jaccard_size; i++) {
        jaccards[i] = jaccards[i] / static_cast<float>(g_degrees[vertices_first[i]] + g_degrees[vertices_second[i]] - jaccards[i]);
    }

}


template<class NodeID_t>
void jaccard_all_row_v(
                const graph& g, 
                std::vector<NodeID_t>& jaccard_first,
                std::vector<NodeID_t>& jaccard_second,
                std::vector<float>& jaccard_coefficients,
                NodeID_t block_size_x, NodeID_t block_size_y);

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

#include "oneapi/dal/algo/jaccard.hpp"
#include <iostream>
#include "oneapi/dal/data/graph.hpp"
#include "oneapi/dal/data/table.hpp"
#include "oneapi/dal/util/csv_data_source.hpp"
#include "oneapi/dal/util/load_graph.hpp"

using namespace oneapi::dal;
using namespace oneapi::dal::preview;

#include <iostream>
//#include "graph.hpp"
//#include "utility.hpp"
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

template<class NodeID_t>
__forceinline size_t  intersection_avx512(NodeID_t* neigh_u, NodeID_t* neigh_v, NodeID_t n_u, NodeID_t n_v)
{
    size_t total = 0;
    NodeID_t i_u = 0, i_v = 0;
        while (i_u < (n_u / 16) * 16 && i_v < (n_v / 16) * 16) {        // not in last n%16 elements

                       // assumes neighbor list is ordered
                NodeID_t minu = neigh_u[i_u];
                NodeID_t maxv = neigh_v[i_v + 15];
                
                                if(minu > maxv) {
                    if (minu > neigh_v[n_v - 1]) {
                        return total;
                    }
                    i_v += 16;
                    continue;
                }

                NodeID_t minv = neigh_v[i_v];
                NodeID_t maxu = neigh_u[i_u + 15]; 
                if(minv > maxu) { 
                    if (minv > neigh_u[n_u - 1]) {
                        return total;
                    }
                    i_u += 16;
                    continue; 
                }
                __m512i v_u = _mm512_loadu_si512((void*)(neigh_u + i_u)); // load 16 neighbors of u
                __m512i v_v = _mm512_loadu_si512((void*)(neigh_v + i_v)); // load 16 neighbors of v
                if (maxu >= maxv) i_v += 16;
                if (maxu <= maxv) i_u += 16;

                __mmask16 match = _mm512_cmpeq_epi32_mask(v_u, v_v);
                if (_mm512_mask2int(match) != 0xffff) {        // shortcut case where all neighbors match
                        __m512i circ1 = _mm512_set_epi32(0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);    // all possible circular shifts for 16 elements
                        __m512i circ2 = _mm512_set_epi32(1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2);
                        __m512i circ3 = _mm512_set_epi32(2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3);
                        __m512i circ4 = _mm512_set_epi32(3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4);
                        __m512i circ5 = _mm512_set_epi32(4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5);
                        __m512i circ6 = _mm512_set_epi32(5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6);
                        __m512i circ7 = _mm512_set_epi32(6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7);
                        __m512i circ8 = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);
                        __m512i circ9 = _mm512_set_epi32(8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9);
                        __m512i circ10 = _mm512_set_epi32(9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10);
                        __m512i circ11 = _mm512_set_epi32(10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11);
                        __m512i circ12 = _mm512_set_epi32(11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12);
                        __m512i circ13 = _mm512_set_epi32(12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13);
                        __m512i circ14 = _mm512_set_epi32(13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14);
                        __m512i circ15 = _mm512_set_epi32(14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15);
                        __m512i v_v1 = _mm512_permutexvar_epi32(circ1, v_v);
                        __m512i v_v2 = _mm512_permutexvar_epi32(circ2, v_v);
                        __m512i v_v3 = _mm512_permutexvar_epi32(circ3, v_v);
                        __m512i v_v4 = _mm512_permutexvar_epi32(circ4, v_v);
                        __m512i v_v5 = _mm512_permutexvar_epi32(circ5, v_v);
                        __m512i v_v6 = _mm512_permutexvar_epi32(circ6, v_v);
                        __m512i v_v7 = _mm512_permutexvar_epi32(circ7, v_v);
                        __m512i v_v8 = _mm512_permutexvar_epi32(circ8, v_v);
                        __m512i v_v9 = _mm512_permutexvar_epi32(circ9, v_v);
                        __m512i v_v10 = _mm512_permutexvar_epi32(circ10, v_v);
                        __m512i v_v11 = _mm512_permutexvar_epi32(circ11, v_v);
                        __m512i v_v12 = _mm512_permutexvar_epi32(circ12, v_v);
                        __m512i v_v13 = _mm512_permutexvar_epi32(circ13, v_v);
                        __m512i v_v14 = _mm512_permutexvar_epi32(circ14, v_v);
                        __m512i v_v15 = _mm512_permutexvar_epi32(circ15, v_v);
                        __mmask16 tmp_match1 = _mm512_cmpeq_epi32_mask(v_u, v_v1);        // find matches
                        __mmask16 tmp_match2 = _mm512_cmpeq_epi32_mask(v_u, v_v2);
                        __mmask16 tmp_match3 = _mm512_cmpeq_epi32_mask(v_u, v_v3);
                        __mmask16 tmp_match4 = _mm512_cmpeq_epi32_mask(v_u, v_v4);
                        __mmask16 tmp_match5 = _mm512_cmpeq_epi32_mask(v_u, v_v5);
                        __mmask16 tmp_match6 = _mm512_cmpeq_epi32_mask(v_u, v_v6);
                        __mmask16 tmp_match7 = _mm512_cmpeq_epi32_mask(v_u, v_v7);
                        __mmask16 tmp_match8 = _mm512_cmpeq_epi32_mask(v_u, v_v8);
                        __mmask16 tmp_match9 = _mm512_cmpeq_epi32_mask(v_u, v_v9);
                        __mmask16 tmp_match10 = _mm512_cmpeq_epi32_mask(v_u, v_v10);
                        __mmask16 tmp_match11 = _mm512_cmpeq_epi32_mask(v_u, v_v11);
                        __mmask16 tmp_match12 = _mm512_cmpeq_epi32_mask(v_u, v_v12);
                        __mmask16 tmp_match13 = _mm512_cmpeq_epi32_mask(v_u, v_v13);
                        __mmask16 tmp_match14 = _mm512_cmpeq_epi32_mask(v_u, v_v14);
                        __mmask16 tmp_match15 = _mm512_cmpeq_epi32_mask(v_u, v_v15);
                        match = _mm512_kor(_mm512_kor(_mm512_kor(_mm512_kor(match, tmp_match1), _mm512_kor(tmp_match2, tmp_match3)), _mm512_kor(_mm512_kor(tmp_match4, tmp_match5), _mm512_kor(tmp_match6, tmp_match7))), _mm512_kor(_mm512_kor(_mm512_kor(tmp_match8, tmp_match9), _mm512_kor(tmp_match10, tmp_match11)), _mm512_kor(_mm512_kor(tmp_match12, tmp_match13), _mm512_kor(tmp_match14, tmp_match15)))); // combine all matches
                    }
                total += _popcnt32(_mm512_mask2int(match));    //count number of matches
            }



            while (i_u < (n_u / 16) * 16 && i_v < n_v) {
                __m512i v_u = _mm512_loadu_si512((void*)(neigh_u + i_u));
                while (neigh_v[i_v] <= neigh_u[i_u + 15] && i_v < n_v) {
                    __m512i tmp_v_v = _mm512_set1_epi32(neigh_v[i_v]);
                    __mmask16 match = _mm512_cmpeq_epi32_mask(v_u, tmp_v_v);
                    if (_mm512_mask2int(match)) total++;
                    i_v++;
                }
                i_u += 16;
            }
            while (i_v < (n_v / 16) * 16 && i_u < n_u) {
                __m512i v_v = _mm512_loadu_si512((void*)(neigh_v + i_v));
                while (neigh_u[i_u] <= neigh_v[i_v + 15] && i_u < n_u) {
                    __m512i tmp_v_u = _mm512_set1_epi32(neigh_u[i_u]);
                    __mmask16 match = _mm512_cmpeq_epi32_mask(v_v, tmp_v_u);
                    if (_mm512_mask2int(match)) total++;
                    i_u++;
                }
                i_v += 16;
            }


            while (i_u <= (n_u - 8) && i_v <= (n_v - 8))
        {        // not in last n%8 elements
                        // assumes neighbor list is ordered
                NodeID_t minu = neigh_u[i_u];
                NodeID_t maxv = neigh_v[i_v + 7];
            

                                if(minu > maxv) {
                    if (minu > neigh_v[n_v - 1]) {
                        return total;
                    }
                    i_v += 8;
                    continue;
                }
                NodeID_t maxu = neigh_u[i_u + 7];
                NodeID_t minv = neigh_v[i_v];
                if(minv > maxu) { 
                    if (minv > neigh_u[n_u - 1]) {
                        return total;
                    }
                    i_u += 8;
                    continue; 
                }
                __m256i v_u = _mm256_loadu_epi32((void*)(neigh_u + i_u)); // load 8 neighbors of u
                __m256i v_v = _mm256_loadu_epi32((void*)(neigh_v + i_v)); // load 8 neighbors of v

                if (maxu >= maxv)
                    i_v += 8;
                if (maxu <= maxv)
                    i_u += 8;

                __mmask8 match = _mm256_cmpeq_epi32_mask(v_u, v_v);
                if (_cvtmask8_u32(match) != 0xff) {        // shortcut case where all neighbors match
                        __m256i circ1 = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);    // all possible circular shifts for 16 elements
                        __m256i circ2 = _mm256_set_epi32(1, 0, 7, 6, 5, 4, 3, 2);
                        __m256i circ3 = _mm256_set_epi32(2, 1, 0, 7, 6, 5, 4, 3);
                        __m256i circ4 = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
                        __m256i circ5 = _mm256_set_epi32(4, 3, 2, 1, 0, 7, 6, 5);
                        __m256i circ6 = _mm256_set_epi32(5, 4, 3, 2, 1, 0, 7, 6);
                        __m256i circ7 = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);

                        __m256i v_v1 = _mm256_permutexvar_epi32(circ1, v_v);
                        __m256i v_v2 = _mm256_permutexvar_epi32(circ2, v_v);
                        __m256i v_v3 = _mm256_permutexvar_epi32(circ3, v_v);
                        __m256i v_v4 = _mm256_permutexvar_epi32(circ4, v_v);
                        __m256i v_v5 = _mm256_permutexvar_epi32(circ5, v_v);
                        __m256i v_v6 = _mm256_permutexvar_epi32(circ6, v_v);
                        __m256i v_v7 = _mm256_permutexvar_epi32(circ7, v_v);

                        __mmask8 tmp_match1 = _mm256_cmpeq_epi32_mask(v_u, v_v1);        // find matches
                        __mmask8 tmp_match2 = _mm256_cmpeq_epi32_mask(v_u, v_v2);
                        __mmask8 tmp_match3 = _mm256_cmpeq_epi32_mask(v_u, v_v3);
                        __mmask8 tmp_match4 = _mm256_cmpeq_epi32_mask(v_u, v_v4);
                        __mmask8 tmp_match5 = _mm256_cmpeq_epi32_mask(v_u, v_v5);
                        __mmask8 tmp_match6 = _mm256_cmpeq_epi32_mask(v_u, v_v6);
                        __mmask8 tmp_match7 = _mm256_cmpeq_epi32_mask(v_u, v_v7);

                        match = _kor_mask8(_kor_mask8(
                            _kor_mask8(match, tmp_match1),
                            _kor_mask8(tmp_match2, tmp_match3)),
                        _kor_mask8(
                            _kor_mask8(tmp_match4, tmp_match5),
                                        _kor_mask8(tmp_match6, tmp_match7))); // combine all matches
                    }
                total += _popcnt32(_cvtmask8_u32(match));    //count number of matches
            }
            if (i_u <= (n_u - 8) && i_v < n_v)
            {
                __m256i v_u = _mm256_loadu_epi32((void*)(neigh_u + i_u));
                while (neigh_v[i_v] <= neigh_u[i_u + 7] && i_v < n_v)
                {
                    __m256i tmp_v_v = _mm256_set1_epi32(neigh_v[i_v]);
                    __mmask8 match = _mm256_cmpeq_epi32_mask(v_u, tmp_v_v);
                    if (_cvtmask8_u32(match))
                        total++;
                    i_v++;
                }
                i_u += 8;
            }
            if (i_v <= (n_v - 8) && i_u < n_u)
            {
                __m256i v_v = _mm256_loadu_epi32((void*)(neigh_v + i_v));
                while (neigh_u[i_u] <= neigh_v[i_v + 7] && i_u < n_u)
                {
                    __m256i tmp_v_u = _mm256_set1_epi32(neigh_u[i_u]);
                    __mmask8 match = _mm256_cmpeq_epi32_mask(v_v, tmp_v_u);
                    if (_cvtmask8_u32(match))
                        total++;
                    i_u++;
                }
                i_v += 8;
            }

            while (i_u <= (n_u - 4) && i_v <= (n_v - 4))
        {        // not in last n%8 elements

                        // assumes neighbor list is ordered
                NodeID_t minu = neigh_u[i_u];
                NodeID_t maxv = neigh_v[i_v + 3];
                


                                if(minu > maxv) {
                    if (minu > neigh_v[n_v - 1]) {
                        return total;
                    }
                    i_v += 4;
                    continue;
                }
                NodeID_t minv = neigh_v[i_v];
                NodeID_t maxu = neigh_u[i_u + 3];
                if(minv > maxu) { 
                    if (minv > neigh_u[n_u - 1]) {
                        return total;
                    }
                    i_u += 4;
                    continue; 
                }
                __m128i v_u = _mm_load_epi32((void*)(neigh_u + i_u)); // load 8 neighbors of u
                __m128i v_v = _mm_load_epi32((void*)(neigh_v + i_v)); // load 8 neighbors of v

                if (maxu >= maxv)
                    i_v += 4;
                if (maxu <= maxv)
                    i_u += 4;

                __mmask8 match = _mm_cmpeq_epi32_mask(v_u, v_v);
                if (_cvtmask8_u32(match) != 0xf) {        // shortcut case where all neighbors match
                    __m128i v_v1 = _mm_shuffle_epi32(v_v, _MM_SHUFFLE(0, 3, 2, 1));
                    __m128i v_v2 = _mm_shuffle_epi32(v_v, _MM_SHUFFLE(1, 0, 3, 2));
                    __m128i v_v3 = _mm_shuffle_epi32(v_v, _MM_SHUFFLE(2, 1, 0, 3));

                        __mmask8 tmp_match1 = _mm_cmpeq_epi32_mask(v_u, v_v1);        // find matches
                        __mmask8 tmp_match2 = _mm_cmpeq_epi32_mask(v_u, v_v2);
                        __mmask8 tmp_match3 = _mm_cmpeq_epi32_mask(v_u, v_v3);

                        match = _kor_mask8(_kor_mask8(match, tmp_match1), _kor_mask8(tmp_match2, tmp_match3)); // combine all matches
                    }
                total += _popcnt32(_cvtmask8_u32(match));    //count number of matches
            }
            if (i_u <= (n_u - 4) && i_v < n_v)
            {
                __m128i v_u = _mm_loadu_epi32((void*)(neigh_u + i_u));
                while (neigh_v[i_v] <= neigh_u[i_u + 3] && i_v < n_v)
                {
                    __m128i tmp_v_v = _mm_set1_epi32(neigh_v[i_v]);
                    __mmask8 match = _mm_cmpeq_epi32_mask(v_u, tmp_v_v);
                    if (_cvtmask8_u32(match))
                        total++;
                    i_v++;
                }
                i_u += 4;
            }
            if (i_v <= (n_v - 4) && i_u < n_u)
            {
                __m128i v_v = _mm_loadu_epi32((void*)(neigh_v + i_v));
                while (neigh_u[i_u] <= neigh_v[i_v + 3] && i_u < n_u)
                {
                    __m128i tmp_v_u = _mm_set1_epi32(neigh_u[i_u]);
                    __mmask8 match = _mm_cmpeq_epi32_mask(v_v, tmp_v_u);
                    if (_cvtmask8_u32(match))
                        total++;
                    i_u++;
                }
                i_v += 4;
            }


            while (i_u < n_u && i_v < n_v) {
                if ((neigh_u[i_u] > neigh_v[n_v -1]) || (neigh_v[i_v] > neigh_u[n_u - 1])) {
                    return total;
                }
                if (neigh_u[i_u] == neigh_v[i_v]) total++, i_u++, i_v++;
                else if (neigh_u[i_u] < neigh_v[i_v]) i_u++;
                else if (neigh_u[i_u] > neigh_v[i_v]) i_v++;
            }
            return total;
        }

template<class NodeID_t>
inline void jaccard_block_avx512_true(const graph& my_graph,
              NodeID_t vert00, NodeID_t vert01,
              NodeID_t vert10, NodeID_t vert11,
              std::vector<NodeID_t>& vertices_first,
              std::vector<NodeID_t>& vertices_second,
              std::vector<float>& jaccards,
              NodeID_t &jaccard_size);

template<class T> inline void Log512i(const __m512i & value)
{
    const size_t n = sizeof(__m512i) / sizeof(T);
    T buffer[n];
    _mm512_storeu_si512((__m512i*)buffer, value);
    for (int i = 0; i < n; i++)
        std::cout << buffer[i] << " ";
}

template<class T> inline void Log512(const __m512 & value)
{
    const size_t n = sizeof(__m512) / sizeof(T);
    T buffer[n];
    _mm512_store_ps((__m512*)buffer, value);
    for (int i = 0; i < n; i++)
        std::cout << buffer[i] << " ";
}


#include <immintrin.h>
template<class NodeID_t>
void jaccard_block_avx512(const graph my_graph,
              NodeID_t vert00, NodeID_t vert01,
              NodeID_t vert10, NodeID_t vert11,
              std::vector<NodeID_t>& vertices_first,
              std::vector<NodeID_t>& vertices_second,
              std::vector<float>& jaccards,
              NodeID_t &jaccard_size)
{
   auto g = oneapi::dal::preview::detail::get_impl(my_graph);
    auto g_edge_offsets = g->_edge_offsets.data();
    auto g_vertex_neighbors = g->_vertex_neighbors.data();
    auto g_degrees = g->_degrees.data();

    jaccard_size = 0;
    NodeID_t j = vert10;


    __m512i n_j_start_v = _mm512_set1_epi32(0);
    __m512i n_j_end_v = _mm512_set1_epi32(0);
    __m512i n_j_start_v1 = _mm512_set1_epi32(0);
    __m512i n_j_end_v1 = _mm512_set1_epi32(0);

    __m512i start_indices_j_v = _mm512_set1_epi32(0);
    __m512i end_indices_j_v_tmp = _mm512_set1_epi32(0);
    __m512i end_indices_j_v = _mm512_set1_epi32(0);
    
    __m512i j_vertices_tmp1 = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    
    __m512i j_vertices = _mm512_set1_epi32(0);
    __m512i j_vertices_tmp2 = _mm512_set1_epi32(0);
    int intersection_size = 0;
    __mmask16 cmpgt1 ;
    __mmask16 cmpgt2 ;
    __mmask16 worth_intersection;
    __declspec(align(64)) NodeID_t stack16_j_vertex[16] = {0};
    unsigned int ones_num = 0;

    for (NodeID_t i = vert00; i < vert01; i++) {

        NodeID_t size_i = g_degrees[i];
        auto n_i = g_vertex_neighbors+ g_edge_offsets[i];
        __m512i n_i_start_v = _mm512_set1_epi32(n_i[0]);
        __m512i n_i_end_v = _mm512_set1_epi32(n_i[size_i - 1]);
        __m512i i_vertex = _mm512_set1_epi32(i);

        //__m512i size_i_v = _mm512_set1_epi32(size_i);



        NodeID_t diagonal = std::min(i, vert11);

        if (j < vert10 + ((diagonal - vert10) / 16) * 16) {
            //load_data(0)
            start_indices_j_v = _mm512_load_epi32(g_edge_offsets + j);
            end_indices_j_v_tmp = _mm512_load_epi32(g_edge_offsets + j + 1);
            end_indices_j_v = _mm512_add_epi32(end_indices_j_v_tmp, _mm512_set1_epi32(-1));

            n_j_start_v = _mm512_permutevar_epi32(j_vertices_tmp1 ,_mm512_i32gather_epi32(start_indices_j_v, g_vertex_neighbors, 4));
            n_j_end_v = _mm512_permutevar_epi32(j_vertices_tmp1 ,_mm512_i32gather_epi32(end_indices_j_v, g_vertex_neighbors, 4));




            for (j; j + 16 <  vert10 + ((diagonal - vert10) / 16) * 16;) {

                start_indices_j_v = _mm512_load_epi32(g_edge_offsets + j + 16);
                end_indices_j_v_tmp = _mm512_load_epi32(g_edge_offsets + j + 17);
                end_indices_j_v = _mm512_add_epi32(end_indices_j_v_tmp, _mm512_set1_epi32(-1));


                n_j_start_v1 = _mm512_permutevar_epi32(j_vertices_tmp1 ,_mm512_i32gather_epi32(start_indices_j_v, g_vertex_neighbors, 4));
                n_j_end_v1 = _mm512_permutevar_epi32(j_vertices_tmp1 ,_mm512_i32gather_epi32(end_indices_j_v, g_vertex_neighbors, 4));


                cmpgt1 = _mm512_cmpgt_epi32_mask( n_i_start_v, n_j_end_v);
                cmpgt2 = _mm512_cmpgt_epi32_mask( n_j_start_v, n_i_end_v);

                worth_intersection = _mm512_knot(_mm512_kor(cmpgt1, cmpgt2));
                ones_num = _popcnt32(_cvtmask16_u32(worth_intersection));

                if (ones_num != 0) {
                    j_vertices_tmp2 = _mm512_set1_epi32(j);
                    j_vertices = _mm512_add_epi32(j_vertices_tmp1, j_vertices_tmp2);

                    _mm512_mask_compressstoreu_epi32((stack16_j_vertex), worth_intersection, j_vertices);

                    __declspec(align(64)) int stack16_intersections[16] = {0.0};
                    for (int s = 0; s < ones_num; s++) {
                        stack16_intersections[s] = (intersection_avx512(n_i, g_vertex_neighbors + g_edge_offsets[stack16_j_vertex[s]], size_i, g_degrees[stack16_j_vertex[s]]));           
                    }
                    __m512i intersections_v = _mm512_load_epi32(stack16_intersections);
                    j_vertices = _mm512_load_epi32(stack16_j_vertex);

                    __mmask16 non_zero_coefficients = _mm512_test_epi32_mask(intersections_v, intersections_v);//_mm512_knot(_mm512_cmpeq_ps_mask(intersections_v, _mm512_set1_ps(0.0)));
                    _mm512_mask_compressstoreu_epi32((vertices_first.data() + jaccard_size), non_zero_coefficients, i_vertex);
                    _mm512_mask_compressstoreu_epi32((vertices_second.data() + jaccard_size), non_zero_coefficients, j_vertices);
                    __m512 tmp_v = _mm512_cvtepi32_ps(intersections_v);
                    _mm512_mask_compressstoreu_ps((jaccards.data() + jaccard_size), non_zero_coefficients, tmp_v);
                    //cout << i << " " << j << endl;
                    //Log512<float>(tmp_v); cout << endl;

                    jaccard_size += _popcnt32(_cvtmask16_u32(non_zero_coefficients));

                }
                // if (i == 308 && j > 36600) {
                //     cout << " catch " << i << " " << j <<endl; 
                // }

                j += 16;

                n_j_start_v = n_j_start_v1;
                n_j_end_v = n_j_end_v1;
            }

            //process n data


            cmpgt1 = _mm512_cmpgt_epi32_mask( n_i_start_v, n_j_end_v);
            cmpgt2 = _mm512_cmpgt_epi32_mask( n_j_start_v, n_i_end_v);

            worth_intersection = _mm512_knot(_mm512_kor(cmpgt1, cmpgt2));
            ones_num = _popcnt32(_cvtmask16_u32(worth_intersection));

            if (ones_num != 0) {
                j_vertices_tmp2 = _mm512_set1_epi32(j);
                j_vertices = _mm512_add_epi32(j_vertices_tmp1, j_vertices_tmp2);

                _mm512_mask_compressstoreu_epi32((stack16_j_vertex), worth_intersection, j_vertices);

                __declspec(align(64)) int stack16_intersections[16] = {0.0};
                for (int s = 0; s < ones_num; s++) {
                    stack16_intersections[s] = (intersection_avx512(n_i, g_vertex_neighbors + g_edge_offsets[stack16_j_vertex[s]], size_i, g_degrees[stack16_j_vertex[s]]));
                    // if (intersection_size) {
                    //     vertices_first[jaccard_size] = i;
                    //     vertices_second[jaccard_size] = stack16_j_vertex[s];
                    //     jaccards[jaccard_size] = static_cast<float>(intersection_size) / static_cast<float> (size_i + g_degrees[stack16_j_vertex[s]] - intersection_size);
                    //     jaccard_size++;
                    // }                
                }
                __m512i intersections_v = _mm512_load_epi32(stack16_intersections);
                j_vertices = _mm512_load_epi32(stack16_j_vertex);
                __mmask16 non_zero_coefficients = _mm512_test_epi32_mask(intersections_v, intersections_v);//_mm512_knot(_mm512_cmpeq_ps_mask(intersections_v, _mm512_set1_ps(0.0)));
                _mm512_mask_compressstoreu_epi32((vertices_first.data() + jaccard_size), non_zero_coefficients, i_vertex);
                _mm512_mask_compressstoreu_epi32((vertices_second.data() + jaccard_size), non_zero_coefficients, j_vertices);
                __m512 tmp_v = _mm512_cvtepi32_ps(intersections_v);
                _mm512_mask_compressstoreu_ps((jaccards.data() + jaccard_size), non_zero_coefficients, tmp_v);

                jaccard_size += _popcnt32(_cvtmask16_u32(non_zero_coefficients));



            }


            j += 16;

            for (j = vert10 + ((diagonal - vert10) / 16) * 16; j < diagonal; j++) {
                NodeID_t size_j = g_degrees[j];             
                auto n_j = g_vertex_neighbors + g_edge_offsets[j];


                if (!(n_i[0] > n_j[size_j -1]) && !(n_j[0] > n_i[size_i - 1])) {

                    intersection_size = intersection_avx512(n_i, n_j, size_i, size_j);
                    //cout << i <<" " << j << " " << intersection_size << endl;

                    if (intersection_size) {
                        vertices_first[jaccard_size] = i;
                        vertices_second[jaccard_size] = j;
                        jaccards[jaccard_size] = static_cast<float>(intersection_size);
                        //cout << i <<" " << j << " " << jaccards[jaccard_size] << endl;
                        jaccard_size++;  

                    }
                }
            }
        }
        else {
            for (j = vert10; j < diagonal; j++) {
                NodeID_t size_j = g_degrees[j];             
                auto n_j = g_vertex_neighbors + g_edge_offsets[j];


                if (!(n_i[0] > n_j[size_j -1]) && !(n_j[0] > n_i[size_i - 1])) {

                    intersection_size = intersection_avx512(n_i, n_j, size_i, size_j);
                    //cout << i <<" " << j << " " << intersection_size << endl;

                    if (intersection_size) {
                        vertices_first[jaccard_size] = i;
                        vertices_second[jaccard_size] = j;
                        jaccards[jaccard_size] = static_cast<float>(intersection_size);
                        //cout << i <<" " << j << " " << jaccards[jaccard_size] << endl;
                        jaccard_size++;  

                    }
                }
            }
        }


        NodeID_t tmp_idx = vert10;
        if (diagonal >= vert10) {
            vertices_first[jaccard_size] = i;
            vertices_second[jaccard_size] = diagonal;
            jaccards[jaccard_size] = 1.0; 
            jaccard_size++;
            tmp_idx = diagonal + 1; 
        }
        j = tmp_idx;


        if (j < tmp_idx + ((vert11 - tmp_idx) / 16) * 16) {
            //load_data(0)
            start_indices_j_v = _mm512_load_epi32(g_edge_offsets + j);
            end_indices_j_v_tmp = _mm512_load_epi32(g_edge_offsets + j + 1);
            end_indices_j_v = _mm512_add_epi32(end_indices_j_v_tmp, _mm512_set1_epi32(-1));

            n_j_start_v = _mm512_permutevar_epi32(j_vertices_tmp1 ,_mm512_i32gather_epi32(start_indices_j_v, g_vertex_neighbors, 4));
            n_j_end_v = _mm512_permutevar_epi32(j_vertices_tmp1 ,_mm512_i32gather_epi32(end_indices_j_v, g_vertex_neighbors, 4));




            for (j; j + 16 < tmp_idx + ((vert11 - tmp_idx) / 16) * 16;) {

                    // if (i == 34644 && vert10 == 34644) {
                    //     cout << " catch1 "  << i << " " << j << " " << vert11 <<endl;
                    // }
                start_indices_j_v = _mm512_load_epi32(g_edge_offsets + j + 16);

                end_indices_j_v_tmp = _mm512_load_epi32(g_edge_offsets + j + 17);

                end_indices_j_v = _mm512_add_epi32(end_indices_j_v_tmp, _mm512_set1_epi32(-1));


                n_j_start_v1 = _mm512_permutevar_epi32(j_vertices_tmp1 ,_mm512_i32gather_epi32(start_indices_j_v, g_vertex_neighbors, 4));


                n_j_end_v1 = _mm512_permutevar_epi32(j_vertices_tmp1 ,_mm512_i32gather_epi32(end_indices_j_v, g_vertex_neighbors, 4));

                    //                                 if (i == 34644 && j == 36644) {
                    //     cout << " catch2 "  << i << " j =" << j << " vert11=" << vert11 <<" vertex neighs size =" << g->_vertex_neighbors.size() << " edges size=" << g->_edge_offsets.size() <<endl;
                    //     Log512i<int>(end_indices_j_v);
                    //     cout << endl;
                    //     for (int s = 0; s < 16; s++) {
                    //         cout << j + 17 + s << " : " << *(g_edge_offsets + j + 17 + s) << " ";

                    //     }
                    //     cout << "border " << tmp_idx + ((vert11 - tmp_idx) / 16) * 16  - 16<< " " <<tmp_idx << " " <<diagonal <<  endl;
                    //     cout << endl;

                    // }

                cmpgt1 = _mm512_cmpgt_epi32_mask( n_i_start_v, n_j_end_v);
                cmpgt2 = _mm512_cmpgt_epi32_mask( n_j_start_v, n_i_end_v);

                worth_intersection = _mm512_knot(_mm512_kor(cmpgt1, cmpgt2));
                ones_num = _popcnt32(_cvtmask16_u32(worth_intersection));

                if (ones_num != 0) {
                    j_vertices_tmp2 = _mm512_set1_epi32(j);
                    j_vertices = _mm512_add_epi32(j_vertices_tmp1, j_vertices_tmp2);

                    _mm512_mask_compressstoreu_epi32((stack16_j_vertex), worth_intersection, j_vertices);

                    __declspec(align(64)) int stack16_intersections[16] = {0.0};
                    for (int s = 0; s < ones_num; s++) {
                        stack16_intersections[s] = (intersection_avx512(n_i, g_vertex_neighbors + g_edge_offsets[stack16_j_vertex[s]], size_i, g_degrees[stack16_j_vertex[s]]));           
                    }
                    __m512i intersections_v = _mm512_load_epi32(stack16_intersections);
                    j_vertices = _mm512_load_epi32(stack16_j_vertex);

                    __mmask16 non_zero_coefficients = _mm512_test_epi32_mask(intersections_v, intersections_v);//_mm512_knot(_mm512_cmpeq_ps_mask(intersections_v, _mm512_set1_ps(0.0)));
                    _mm512_mask_compressstoreu_epi32((vertices_first.data() + jaccard_size), non_zero_coefficients, i_vertex);
                    _mm512_mask_compressstoreu_epi32((vertices_second.data() + jaccard_size), non_zero_coefficients, j_vertices);
                    __m512 tmp_v = _mm512_cvtepi32_ps(intersections_v);
                    _mm512_mask_compressstoreu_ps((jaccards.data() + jaccard_size), non_zero_coefficients, tmp_v);
                    //cout << i << " " << j << endl;
                    //Log512<float>(tmp_v); cout << endl;

                    jaccard_size += _popcnt32(_cvtmask16_u32(non_zero_coefficients));

                }
                // if (i == 308 && j > 36600) {
                //     cout << " catch " << i << " " << j <<endl; 
                // }
                    //                 if (i == 32598 && j == 36662) {
                    //     cout << " catch3 "  << i << " " << j << " " << vert11 <<endl;
                    // }

                j += 16;

                n_j_start_v = n_j_start_v1;
                n_j_end_v = n_j_end_v1;

            }

            //process n data

                    // if (i == 34644 && vert10 == 34644) {
                    //     cout << " catch1 "  << i << " " << j << " " << vert11 <<endl;
                    // }
            cmpgt1 = _mm512_cmpgt_epi32_mask( n_i_start_v, n_j_end_v);
            cmpgt2 = _mm512_cmpgt_epi32_mask( n_j_start_v, n_i_end_v);

            worth_intersection = _mm512_knot(_mm512_kor(cmpgt1, cmpgt2));
            ones_num = _popcnt32(_cvtmask16_u32(worth_intersection));
                    //             if (i == 34644 && vert10 == 34644) {
                    //     cout << " catch1 "  << i << " " << j << " " << vert11 <<endl;
                    // }

            if (ones_num != 0) {
                j_vertices_tmp2 = _mm512_set1_epi32(j);
                j_vertices = _mm512_add_epi32(j_vertices_tmp1, j_vertices_tmp2);

                _mm512_mask_compressstoreu_epi32((stack16_j_vertex), worth_intersection, j_vertices);

                __declspec(align(64)) int stack16_intersections[16] = {0.0};
                for (int s = 0; s < ones_num; s++) {
                    stack16_intersections[s] = (intersection_avx512(n_i, g_vertex_neighbors + g_edge_offsets[stack16_j_vertex[s]], size_i, g_degrees[stack16_j_vertex[s]]));
               
                }
                __m512i intersections_v = _mm512_load_epi32(stack16_intersections);
                j_vertices = _mm512_load_epi32(stack16_j_vertex);
                __mmask16 non_zero_coefficients = _mm512_test_epi32_mask(intersections_v, intersections_v);//_mm512_knot(_mm512_cmpeq_ps_mask(intersections_v, _mm512_set1_ps(0.0)));
                _mm512_mask_compressstoreu_epi32((vertices_first.data() + jaccard_size), non_zero_coefficients, i_vertex);
                _mm512_mask_compressstoreu_epi32((vertices_second.data() + jaccard_size), non_zero_coefficients, j_vertices);
                __m512 tmp_v = _mm512_cvtepi32_ps(intersections_v);
                _mm512_mask_compressstoreu_ps((jaccards.data() + jaccard_size), non_zero_coefficients, tmp_v);

                jaccard_size += _popcnt32(_cvtmask16_u32(non_zero_coefficients));



            }


            j += 16;

            for (j = tmp_idx + ((vert11 - tmp_idx) / 16) * 16; j < vert11; j++) {
                NodeID_t size_j = g_degrees[j];             
                auto n_j = g_vertex_neighbors + g_edge_offsets[j];


                if (!(n_i[0] > n_j[size_j -1]) && !(n_j[0] > n_i[size_i - 1])) {

                    intersection_size = intersection_avx512(n_i, n_j, size_i, size_j);
                    //cout << i <<" " << j << " " << intersection_size << endl;

                    if (intersection_size) {
                        vertices_first[jaccard_size] = i;
                        vertices_second[jaccard_size] = j;
                        jaccards[jaccard_size] = static_cast<float>(intersection_size);
                        //cout << i <<" " << j << " " << jaccards[jaccard_size] << endl;
                        jaccard_size++;  

                    }
                }
            }
        }
        else {
            for (j = tmp_idx; j < vert11; j++) {
                NodeID_t size_j = g_degrees[j];             
                auto n_j = g_vertex_neighbors + g_edge_offsets[j];


                if (!(n_i[0] > n_j[size_j -1]) && !(n_j[0] > n_i[size_i - 1])) {

                    intersection_size = intersection_avx512(n_i, n_j, size_i, size_j);
                    //cout << i <<" " << j << " " << intersection_size << endl;

                    if (intersection_size) {
                        vertices_first[jaccard_size] = i;
                        vertices_second[jaccard_size] = j;
                        jaccards[jaccard_size] = static_cast<float>(intersection_size);
                        //cout << i <<" " << j << " " << jaccards[jaccard_size] << endl;
                        jaccard_size++;  

                    }
                }
            }
        }
    }

// if (vert00 == 34644 && vert10 == 34644) {
//     cout << " catch " <<endl;
// }
#pragma vector always
    for (int i = 0; i < jaccard_size; i++) {
        jaccards[i] = jaccards[i] / static_cast<float>(g_degrees[vertices_first[i]] + g_degrees[vertices_second[i]] - jaccards[i]);
    }
// if (vert00 == 34644 && vert10 == 34644) {
//     cout << " catch " <<endl;
// }

}


template<class NodeID_t>
void jaccard_all_block_v(
                const graph& g, NodeID_t block_i, NodeID_t block_j,
                std::vector<NodeID_t>& jaccard_first,
                std::vector<NodeID_t>& jaccard_second,
                std::vector<float>& jaccard_coefficients);

template<class NodeID_t>
void jaccard_all_block(
                const graph& g, NodeID_t block_i, NodeID_t block_j,
                std::vector<NodeID_t>& jaccard_first,
                std::vector<NodeID_t>& jaccard_second,
                std::vector<float>& jaccard_coefficients)
{


    int max_nnz_block = 0;
    NodeID_t num = get_vertex_count(g);
    cout << num << endl;

    NodeID_t blocks = num / block_i;
    NodeID_t remain_els = num - blocks * block_i;
    NodeID_t delta = remain_els / blocks;
    NodeID_t tail = remain_els - blocks * delta;

    //EdgeID_t jac_size = 0;

    NodeID_t num_blocks = 0;
    for (int i = 0; i < blocks; i++) {
        int block_size_i = block_i + delta; int begin_i = 0; int i_tail = 0;
        NodeID_t shift = i * block_size_i;
        if (i >= blocks - tail) {
            block_size_i = block_i + delta + 1; begin_i = (blocks - tail) * (block_i + delta); i_tail = blocks - tail;
            shift = (blocks - tail) * (block_i + delta) + (i - (blocks - tail)) * block_size_i;
        }
        NodeID_t num_sh = (num - shift);
        int blocks_j = num_sh / block_j;
        num_blocks += blocks_j;
        if (blocks_j == 0) {
            ++num_blocks;
        }
    }

    int number_of_threads = tbb::this_task_arena::max_concurrency();
    cout << "tbb threads " << number_of_threads << endl;
    std::vector< std::vector<NodeID_t>> blocks_first(number_of_threads, std::vector<NodeID_t> (block_i * block_j * 2));
    std::vector< std::vector<NodeID_t>> blocks_second(number_of_threads, std::vector<NodeID_t> (block_i * block_j * 2));
    std::vector< std::vector<float>> blocks_jaccards(number_of_threads, std::vector<float> (block_i * block_j * 2));


    /*
    size_t num_non_exists = (num * num - num) / 2 - g.num_edges();
    EdgeID_t jaccard_size = 0;
    if (!jaccard_first.empty()) {
        jaccard_first.clear();
    }

    if (!jaccard_second.empty()) {
        jaccard_second.clear();
    }

    if (!jaccard_coefficients.empty()) {
        jaccard_coefficients.clear();
    }
    int32_t ratio_all_coefs_with_nnz_coeffs = 7; 
    jaccard_first.resize(num_non_exists / ratio_all_coefs_with_nnz_coeffs);
    jaccard_second.resize(num_non_exists / ratio_all_coefs_with_nnz_coeffs);
    jaccard_coefficients.resize(num_non_exists / ratio_all_coefs_with_nnz_coeffs);
    

    if (jaccard_first.data() == nullptr) {
        return bad_allocation;
    }
    if (jaccard_second.data() == nullptr) {
        return bad_allocation;
    }
    if (jaccard_coefficients.data() == nullptr) {
        return bad_allocation;
    }
    std::mutex Mutex;
    */
   
    tbb::parallel_for(tbb::blocked_range<int>(0, blocks),
        [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i != r.end(); ++i) {
                int block_size_i = block_i + delta; int begin_i = 0; int i_tail = 0;
                NodeID_t shift = i * block_size_i;
                if (i >= blocks - tail) {
                    block_size_i = block_i + delta + 1; begin_i = (blocks - tail) * (block_i + delta); i_tail = blocks - tail;
                    shift = (blocks - tail) * (block_i + delta) + (i - (blocks - tail)) * block_size_i;
                }
                NodeID_t num_sh = (num - shift);
                int blocks_j = num_sh / block_j;
                int remain_els_j = num_sh - blocks_j * block_j;
                int delta_j = 0;
                if (blocks_j > 0) {
                    delta_j = remain_els_j / blocks_j;
                }
                else { 
                    NodeID_t jaccard_block_nnz = 0;
                    //Mutex.lock();
                                // cout << begin_i + (i - i_tail) * block_size_i << " " << begin_i + (i - i_tail + 1) * block_size_i << " : " <<
                                 //   begin_i + (i - i_tail) * block_size_i << " " << (NodeID_t)num << endl;
                    jaccard_block_avx512(g,
                                    begin_i + (i - i_tail) * block_size_i, begin_i + (i - i_tail + 1) * block_size_i,
                                    begin_i + (i - i_tail) * block_size_i, (NodeID_t)num,
                                    blocks_first[tbb::this_task_arena::current_thread_index()], 
                                    blocks_second[tbb::this_task_arena::current_thread_index()], 
                                    blocks_jaccards[tbb::this_task_arena::current_thread_index()], 
                                    //blocks[0],
                                    jaccard_block_nnz);
                    // add_nnz_jaccards(jaccard_first, jaccard_second, jaccard_coefficients,
                    //     blocks_first[tbb::this_task_arena::current_thread_index()], 
                    //     blocks_second[tbb::this_task_arena::current_thread_index()], 
                    //     blocks_jaccards[tbb::this_task_arena::current_thread_index()], 
                    //     jaccard_block_nnz, jaccard_size, Mutex);
                    /*
                    Mutex.lock();
                    EdgeID_t jaccard_block_size = jaccard_block_nnz;
                    if (jaccard_first.size() < jaccard_size + jaccard_block_size) {
                        jaccard_first.resize(jaccard_size + jaccard_block_size);
                    }
                    if (jaccard_second.size() < jaccard_size + jaccard_block_size) {
                        jaccard_second.resize(jaccard_size + jaccard_block_size);
                    }
                    if (jaccard_coefficients.size() < jaccard_size + jaccard_block_size) {
                        jaccard_coefficients.resize(jaccard_size + jaccard_block_size);
                    }                                        
                    //add_block_nonzero_coeffs_in_jaccard_vector(blocks[tbb::this_task_arena::current_thread_index()].data(), jaccard_block_size, jaccard.data(), jaccard_size, Mutex);
                    int count = 0;
                    //cout << jaccard_block_size << endl;
                    for (EdgeID_t i = 0; i < jaccard_block_size; i++) {
                        if (blocks_first[tbb::this_task_arena::current_thread_index()][i] < blocks_second[tbb::this_task_arena::current_thread_index()][i]) {
                            jaccard_first[jaccard_size + count] = (blocks_first[tbb::this_task_arena::current_thread_index()])[i];
                            jaccard_second[jaccard_size + count] = (blocks_second[tbb::this_task_arena::current_thread_index()])[i];
                            jaccard_coefficients[jaccard_size + count] = (blocks_jaccards[tbb::this_task_arena::current_thread_index()])[i];
                            count++;
                        }
                    }
                    //cout << count << endl;
                    jaccard_size += count;
                    //cout << "ok" << endl;
                    // cout << block_x_begin << " " << block_x_end << " : " <<
                    //         block_y_begin << " " << block_y_end << endl;
                    Mutex.unlock();
                    */

                }

                int tail_j = remain_els_j - blocks_j * delta_j;
                tbb::parallel_for(tbb::blocked_range<int>(0, blocks_j),
                    [&](const tbb::blocked_range<int>& inner_r) {
                        for (int j = inner_r.begin(); j != inner_r.end(); ++j) {
                            int block_size_j = block_j + delta_j; int begin_j = shift; int j_tail = 0; int block_size_j_end = block_j + delta_j;
                            if (j >= blocks_j - tail_j) {
                                block_size_j = block_j + delta_j + 1;
                                begin_j = (blocks_j - tail_j) * (block_j + delta_j) + shift;
                                j_tail = blocks_j - tail_j; block_size_j_end = block_size_j;
                            }

                                NodeID_t jaccard_block_nnz = 0;
                                //Mutex.lock();
                                 //cout << begin_i + (i - i_tail) * block_size_i << " " << begin_i + (i - i_tail + 1) * block_size_i << " : " <<
                                  //       begin_j + (j - j_tail) * block_size_j << " " << begin_j + (j - j_tail) * block_size_j + block_size_j_end << endl;
                                 jaccard_block_avx512(g,
                                                 begin_i + (i - i_tail) * block_size_i, begin_i + (i - i_tail + 1) * block_size_i,
                                                 begin_j + (j - j_tail) * block_size_j, begin_j + (j - j_tail) * block_size_j + block_size_j_end,
                                                 blocks_first[tbb::this_task_arena::current_thread_index()], 
                                                 blocks_second[tbb::this_task_arena::current_thread_index()], 
                                                 blocks_jaccards[tbb::this_task_arena::current_thread_index()], 
                                                 jaccard_block_nnz);
                                 /*
                                Mutex.lock();
                                EdgeID_t jaccard_block_size = jaccard_block_nnz;
                                if (jaccard_first.size() < jaccard_size + jaccard_block_size) {
                                    jaccard_first.resize(jaccard_size + jaccard_block_size);
                                }
                                if (jaccard_second.size() < jaccard_size + jaccard_block_size) {
                                    jaccard_second.resize(jaccard_size + jaccard_block_size);
                                }
                                if (jaccard_coefficients.size() < jaccard_size + jaccard_block_size) {
                                    jaccard_coefficients.resize(jaccard_size + jaccard_block_size);
                                }                                        
                                //add_block_nonzero_coeffs_in_jaccard_vector(blocks[tbb::this_task_arena::current_thread_index()].data(), jaccard_block_size, jaccard.data(), jaccard_size, Mutex);
                                int count = 0;
                                //cout << jaccard_block_size << endl;
                                for (EdgeID_t i = 0; i < jaccard_block_size; i++) {
                                    if (blocks_first[tbb::this_task_arena::current_thread_index()][i] < blocks_second[tbb::this_task_arena::current_thread_index()][i]) {
                                        jaccard_first[jaccard_size + count] = (blocks_first[tbb::this_task_arena::current_thread_index()])[i];
                                        jaccard_second[jaccard_size + count] = (blocks_second[tbb::this_task_arena::current_thread_index()])[i];
                                        jaccard_coefficients[jaccard_size + count] = (blocks_jaccards[tbb::this_task_arena::current_thread_index()])[i];
                                        count++;
                                    }
                                }
                                //cout << count << endl;
                                jaccard_size += count;
                                //cout << "ok" << endl;
                                // cout << block_x_begin << " " << block_x_end << " : " <<
                                //         block_y_begin << " " << block_y_end << endl;
                                Mutex.unlock();
                                */

                        }}, tbb::simple_partitioner{});
                }}, tbb::simple_partitioner{});
    /*
    jaccard_first.resize(jaccard_size);//
    jaccard_second.resize(jaccard_size);//
    jaccard_coefficients.resize(jaccard_size);//
    */

}





template<class NodeID_t>
void jaccard_all_row_true(
                const graph& g, 
                std::vector<NodeID_t>& jaccard_first,
                std::vector<NodeID_t>& jaccard_second,
                std::vector<float>& jaccard_coefficients,
                NodeID_t block_size_x, NodeID_t block_size_y);

void verify_results(oneapi::dal::preview::graph g, vector<int32_t>& jaccard_first, vector<int32_t>& jaccard_second, vector<float>& jaccard_coefficients, string& path);

int main(int argc, char ** argv)
{
    tbb::task_scheduler_init init(32);

    int num_trials_custom = 10;
    int num_trials_lib = 1;
    int verify = 0;


    if (argc < 2) return 0;
    std::string filename = argv[1];
    csv_data_source ds(filename);
    edge_list_to_csr_descriptor d;
     auto my_graph = load_graph<graph>(d, ds);
    cout << "edges number" << get_edge_count( my_graph);
    auto layout = oneapi::dal::preview::detail::get_impl(my_graph);
    //layout->_vertex_neighbors; // dal::array -> std::vector
    //layout->_edge_offsets; // dal::array -> std::vector

    int32_t block_size_x = 1;
    int32_t block_size_y = 1024;

    if (argc > 5) {
        block_size_x = stoi(argv[2]);
        block_size_y = stoi(argv[3]);
        num_trials_custom = stoi(argv[4]);
        verify = stoi(argv[5]);
    }
 //   tbb::task_scheduler_init init(num_of_threads);

//     std::vector<jaccard_pair<int32_t>> jaccard;

///*
    std::vector<int32_t> jaccard_first;
    std::vector<int32_t> jaccard_second;
    std::vector<float> jaccard_coefficients;
//*/
    cout << "jaccard_non_existent with custom block_size = " << block_size_x <<"x" <<block_size_y << endl;
    vector<double> time;
    double median;

    for(int i = 0; i < num_trials_custom; i++) {
        auto start = high_resolution_clock::now();
        //jaccard_all_row(my_graph, jaccard_first, jaccard_second, jaccard_coefficients, block_size_x, block_size_y);
        jaccard_all_block(my_graph, block_size_x, block_size_y, jaccard_first, jaccard_second, jaccard_coefficients);
        //jaccard_status = jaccard_all_row(g, jaccard, block_size_x, block_size_y, 1);
        auto stop = high_resolution_clock::now();

        time.push_back(std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count());
        cout << i << " iter: " << time.back() << endl;
    }

// for (int i = 1; i < 5000; i *= 2) {
//     for (int j = max(i, 16); j < 40000; j *= 2) {
//         auto start = high_resolution_clock::now();
//         jaccard_status = jaccard_all_block(g, i, j, jaccard_first, jaccard_second, jaccard_coefficients);
//         auto stop = high_resolution_clock::now();
//         cout << "block " << i <<"x"<<j <<endl;
//         cout << "time       " << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count() <<endl;
//         cout << endl;
//     }
// }

    //computing time metrics
    sort(time.begin(), time.end());
    if (num_trials_custom % 2 == 0)
        median =  (time[time.size()/2] + time[time.size()/2 - 1]) / 2;
    else
        median = time[time.size()/2];
    cout <<"median: " << median << endl;
    cout <<"Min: " << time[0] << endl;
    cout <<"Max: " << time.back() << endl;

    string verify_file("/nfs/inn/proj/numerics1/Users/akumin/oneDAL/results_Enron.txt");
    if (verify) {
        jaccard_all_block_v(my_graph, block_size_x, block_size_y, jaccard_first, jaccard_second, jaccard_coefficients);
        verify_results(my_graph, jaccard_first, jaccard_second, jaccard_coefficients, verify_file);
    }

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

void verify_results(oneapi::dal::preview::graph g, vector<int32_t>& jaccard_first, vector<int32_t>& jaccard_second, vector<float>& jaccard_coefficients, string& path) {
    cout << "Results verification." << endl;

    auto layout = oneapi::dal::preview::detail::get_impl(g);
    layout->_vertex_neighbors; // dal::array -> std::vector
    layout->_edge_offsets; // dal::array -> std::vector

    vector<jaccard_pair> jaccard(jaccard_first.size());
    for (int i = 0 ; i < jaccard.size(); i++) {
        jaccard[i]= jaccard_pair(jaccard_first[i], jaccard_second[i], jaccard_coefficients[i]);
    }
    cout << " jaccard custom: " << jaccard.size() << endl;



    sort(jaccard.begin(), jaccard.end(), [] (const auto& lhs, const auto& rhs) { 
      if (lhs.first < rhs.first) {
          return true;
      }
      else if (lhs.first == rhs.first) {
              if (lhs.second < rhs.second) {
                  return true;
              }
              else
                  return false;
      }
      else{
          return false;
      }
      });

    
    // std::vector<int32_t> jaccard_first1;
    // std::vector<int32_t> jaccard_second1;
    // std::vector<float> jaccard_coefficients1;
    // jaccard_all_row_true(g, jaccard_first1, jaccard_second1, jaccard_coefficients1, 1, 2048);

    // vector<jaccard_pair> jaccard_lib(jaccard_first1.size());
    // for (int i = 0 ; i < jaccard_lib.size(); i++) {
    //     jaccard_lib[i]= jaccard_pair(jaccard_first1[i], jaccard_second1[i], jaccard_coefficients1[i]);
    // }

    // sort(jaccard_lib.begin(), jaccard_lib.end(), [] (const auto& lhs, const auto& rhs) { 
    //   if (lhs.first < rhs.first) {
    //       return true;
    //   }
    //   else if (lhs.first == rhs.first) {
    //           if (lhs.second < rhs.second) {
    //               return true;
    //           }
    //           else
    //               return false;
    //   }
    //   else{
    //       return false;
    //   }
    //   });

    vector<jaccard_pair> jaccard_lib(0);
    ifstream myfile;
    myfile.open(path);
    if (myfile.is_open())
    {
        int size_jaccard = 0;
        myfile >> size_jaccard;
        jaccard_lib.resize(size_jaccard);
        for (int i = 0; i < jaccard_lib.size(); i++) {
            myfile >> jaccard_lib[i].first >> jaccard_lib[i].second >> jaccard_lib[i].coefficient;
        }
        myfile.close();
    }



    cout << " jaccard lib: " << jaccard_lib.size() << endl;

    bool is_correct = true;
      float eps = 0.000001;
      for(int i = 0; i < jaccard_lib.size(); i++) {
          if (!((jaccard[i].coefficient < jaccard_lib[i].coefficient + eps) && (jaccard[i].coefficient > jaccard_lib[i].coefficient - eps) )) {
              cout << jaccard_lib[i].first << " " << jaccard_lib[i].second << " jaccard_lib: " <<
              jaccard_lib[i].coefficient << endl; 
              cout << jaccard[i].first << " " << jaccard[i].second << " jaccard: " <<
              jaccard[i].coefficient << endl;

              cout << "Neighbors " << jaccard_lib[i].first <<" " <<layout->_degrees[jaccard_lib[i].first] <<" : ";

              is_correct = false;
              for (int j = layout->_edge_offsets[jaccard_lib[i].first]; j < layout->_edge_offsets[jaccard_lib[i].first + 1]; j++)
              {
                  cout << layout->_vertex_neighbors[j] << " ";
              }
              cout << endl;

              cout << "Neighbors " << jaccard_lib[i].second << " : ";
              is_correct = false;
              for (int j = layout->_edge_offsets[jaccard_lib[i].second]; j < layout->_edge_offsets[jaccard_lib[i].second + 1]; j++)
              {
                  cout << layout->_vertex_neighbors[j] << " ";
              }
              cout << endl;

              cout << "Neighbors " << jaccard[i].first << " : ";
              is_correct = false;
              for (int j = layout->_edge_offsets[jaccard[i].first]; j < layout->_edge_offsets[jaccard[i].first + 1]; j++)
              {
                  cout << layout->_vertex_neighbors[j] << " ";
              }
              cout << endl;

              cout << "Neighbors " << jaccard[i].second << " : ";

              is_correct = false;
              for (int j = layout->_edge_offsets[jaccard[i].second]; j < layout->_edge_offsets[jaccard[i].second + 1]; j++)
              {
                  cout << layout->_vertex_neighbors[j] << " ";
              }
              cout << endl;

              for (int j = max(0, i - 5) ; j < i + 5; j++)
              {
                  cout << jaccard[j].first << " " << jaccard[j].second << " jaccard: " <<
                  jaccard[j].coefficient << "            ";

                  cout << jaccard_lib[j].first << " " << jaccard_lib[j].second << " jaccard_lib: " <<
                  jaccard_lib[j].coefficient << endl;
              }

              break;
          }
      }
  if (is_correct == true) {
      cout << "Succesfull." << endl;
  }
  else {
      cout << "Failed." << endl;
  }

}

template<class NodeID_t>
void jaccard_all_block_v(
                const graph& g, NodeID_t block_i, NodeID_t block_j,
                std::vector<NodeID_t>& jaccard_first,
                std::vector<NodeID_t>& jaccard_second,
                std::vector<float>& jaccard_coefficients) {


    int64_t max_nnz_block = 0;
    NodeID_t num = get_vertex_count( g);

    NodeID_t blocks = num / block_i;
    NodeID_t remain_els = num - blocks * block_i;
    NodeID_t delta = remain_els / blocks;
    NodeID_t tail = remain_els - blocks * delta;

    //int64_t jac_size = 0;

    NodeID_t num_blocks = 0;
    for (int i = 0; i < blocks; i++) {
        int block_size_i = block_i + delta; int begin_i = 0; int i_tail = 0;
        NodeID_t shift = i * block_size_i;
        if (i >= blocks - tail) {
            block_size_i = block_i + delta + 1; begin_i = (blocks - tail) * (block_i + delta); i_tail = blocks - tail;
            shift = (blocks - tail) * (block_i + delta) + (i - (blocks - tail)) * block_size_i;
        }
        NodeID_t num_sh = (num - shift);
        int blocks_j = num_sh / block_j;
        num_blocks += blocks_j;
        if (blocks_j == 0) {
            ++num_blocks;
        }
    }

    int number_of_threads = tbb::this_task_arena::max_concurrency();
    cout << "tbb threads " << number_of_threads << endl;
    std::vector< std::vector<NodeID_t>> blocks_first(number_of_threads, std::vector<NodeID_t> (block_i * block_j * 2));
    std::vector< std::vector<NodeID_t>> blocks_second(number_of_threads, std::vector<NodeID_t> (block_i * block_j * 2));
    std::vector< std::vector<float>> blocks_jaccards(number_of_threads, std::vector<float> (block_i * block_j * 2));

    ///*
    size_t num_non_exists = (num * num - num) / 2 - get_edge_count(g);
    int64_t jaccard_size = 0;
    if (!jaccard_first.empty()) {
        jaccard_first.clear();
    }

    if (!jaccard_second.empty()) {
        jaccard_second.clear();
    }

    if (!jaccard_coefficients.empty()) {
        jaccard_coefficients.clear();
    }
    int32_t ratio_all_coefs_with_nnz_coeffs = 7; 
    jaccard_first.resize(num_non_exists / ratio_all_coefs_with_nnz_coeffs);
    jaccard_second.resize(num_non_exists / ratio_all_coefs_with_nnz_coeffs);
    jaccard_coefficients.resize(num_non_exists / ratio_all_coefs_with_nnz_coeffs);
    

    std::mutex Mutex;
    //*/
   
    tbb::parallel_for(tbb::blocked_range<int>(0, blocks),
        [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i != r.end(); ++i) {
                int block_size_i = block_i + delta; int begin_i = 0; int i_tail = 0;
                NodeID_t shift = i * block_size_i;
                if (i >= blocks - tail) {
                    block_size_i = block_i + delta + 1; begin_i = (blocks - tail) * (block_i + delta); i_tail = blocks - tail;
                    shift = (blocks - tail) * (block_i + delta) + (i - (blocks - tail)) * block_size_i;
                }
                NodeID_t num_sh = (num - shift);
                int blocks_j = num_sh / block_j;
                int remain_els_j = num_sh - blocks_j * block_j;
                int delta_j = 0;
                if (blocks_j > 0) {
                    delta_j = remain_els_j / blocks_j;
                }
                else { 
                    NodeID_t jaccard_block_nnz = 0;
                    //Mutex.lock();
                     //cout << block_x_begin << " " << block_x_end << " : " <<
                     //        block_y_begin << " " << block_y_end << endl;
                    jaccard_block_avx512(g,
                                    begin_i + (i - i_tail) * block_size_i, begin_i + (i - i_tail + 1) * block_size_i,
                                    begin_i + (i - i_tail) * block_size_i, (NodeID_t)num,
                                    blocks_first[tbb::this_task_arena::current_thread_index()], 
                                    blocks_second[tbb::this_task_arena::current_thread_index()], 
                                    blocks_jaccards[tbb::this_task_arena::current_thread_index()], 
                                    //blocks[0],
                                    jaccard_block_nnz);
                    // add_nnz_jaccards(jaccard_first, jaccard_second, jaccard_coefficients,
                    //     blocks_first[tbb::this_task_arena::current_thread_index()], 
                    //     blocks_second[tbb::this_task_arena::current_thread_index()], 
                    //     blocks_jaccards[tbb::this_task_arena::current_thread_index()], 
                    //     jaccard_block_nnz, jaccard_size, Mutex);
                    ///*
                    Mutex.lock();
                    int64_t jaccard_block_size = jaccard_block_nnz;
                    if (jaccard_first.size() < jaccard_size + jaccard_block_size) {
                        jaccard_first.resize(jaccard_size + jaccard_block_size);
                    }
                    if (jaccard_second.size() < jaccard_size + jaccard_block_size) {
                        jaccard_second.resize(jaccard_size + jaccard_block_size);
                    }
                    if (jaccard_coefficients.size() < jaccard_size + jaccard_block_size) {
                        jaccard_coefficients.resize(jaccard_size + jaccard_block_size);
                    }                                        
                    //add_block_nonzero_coeffs_in_jaccard_vector(blocks[tbb::this_task_arena::current_thread_index()].data(), jaccard_block_size, jaccard.data(), jaccard_size, Mutex);
                    int count = 0;
                    //cout << jaccard_block_size << endl;
                    for (int64_t i = 0; i < jaccard_block_size; i++) {
                        if (blocks_first[tbb::this_task_arena::current_thread_index()][i] < blocks_second[tbb::this_task_arena::current_thread_index()][i]) {
                            jaccard_first[jaccard_size + count] = (blocks_first[tbb::this_task_arena::current_thread_index()])[i];
                            jaccard_second[jaccard_size + count] = (blocks_second[tbb::this_task_arena::current_thread_index()])[i];
                            jaccard_coefficients[jaccard_size + count] = (blocks_jaccards[tbb::this_task_arena::current_thread_index()])[i];
                            count++;
                        }
                    }
                    //cout << count << endl;
                    jaccard_size += count;
                    //cout << "ok" << endl;
                    // cout << block_x_begin << " " << block_x_end << " : " <<
                    //         block_y_begin << " " << block_y_end << endl;
                    Mutex.unlock();
                    //*/

                }

                int tail_j = remain_els_j - blocks_j * delta_j;
                tbb::parallel_for(tbb::blocked_range<int>(0, blocks_j),
                    [&](const tbb::blocked_range<int>& inner_r) {
                        for (int j = inner_r.begin(); j != inner_r.end(); ++j) {
                            int block_size_j = block_j + delta_j; int begin_j = shift; int j_tail = 0; int block_size_j_end = block_j + delta_j;
                            if (j >= blocks_j - tail_j) {
                                block_size_j = block_j + delta_j + 1;
                                begin_j = (blocks_j - tail_j) * (block_j + delta_j) + shift;
                                j_tail = blocks_j - tail_j; block_size_j_end = block_size_j;
                            }

                                NodeID_t jaccard_block_nnz = 0;
                                //Mutex.lock();
                                 //cout << begin_i + (i - i_tail) * block_size_i << " " << begin_i + (i - i_tail + 1) * block_size_i << " : " <<
                                 //        begin_j + (j - j_tail) * block_size_j << " " << begin_j + (j - j_tail) * block_size_j + block_size_j_end << endl;
                                 jaccard_block_avx512(g,
                                                 begin_i + (i - i_tail) * block_size_i, begin_i + (i - i_tail + 1) * block_size_i,
                                                 begin_j + (j - j_tail) * block_size_j, begin_j + (j - j_tail) * block_size_j + block_size_j_end,
                                                 blocks_first[tbb::this_task_arena::current_thread_index()], 
                                                 blocks_second[tbb::this_task_arena::current_thread_index()], 
                                                 blocks_jaccards[tbb::this_task_arena::current_thread_index()], 
                                                 jaccard_block_nnz);
                                // /*
                                Mutex.lock();
                                int64_t jaccard_block_size = jaccard_block_nnz;
                                if (jaccard_first.size() < jaccard_size + jaccard_block_size) {
                                    jaccard_first.resize(jaccard_size + jaccard_block_size);
                                }
                                if (jaccard_second.size() < jaccard_size + jaccard_block_size) {
                                    jaccard_second.resize(jaccard_size + jaccard_block_size);
                                }
                                if (jaccard_coefficients.size() < jaccard_size + jaccard_block_size) {
                                    jaccard_coefficients.resize(jaccard_size + jaccard_block_size);
                                }                                        
                                //add_block_nonzero_coeffs_in_jaccard_vector(blocks[tbb::this_task_arena::current_thread_index()].data(), jaccard_block_size, jaccard.data(), jaccard_size, Mutex);
                                int count = 0;
                                //cout << jaccard_block_size << endl;
                                for (int64_t i = 0; i < jaccard_block_size; i++) {
                                    if (blocks_first[tbb::this_task_arena::current_thread_index()][i] < blocks_second[tbb::this_task_arena::current_thread_index()][i]) {
                                        jaccard_first[jaccard_size + count] = (blocks_first[tbb::this_task_arena::current_thread_index()])[i];
                                        jaccard_second[jaccard_size + count] = (blocks_second[tbb::this_task_arena::current_thread_index()])[i];
                                        jaccard_coefficients[jaccard_size + count] = (blocks_jaccards[tbb::this_task_arena::current_thread_index()])[i];
                                        count++;
                                    }
                                }
                                //cout << count << endl;
                                jaccard_size += count;
                                //cout << "ok" << endl;
                                // cout << block_x_begin << " " << block_x_end << " : " <<
                                //         block_y_begin << " " << block_y_end << endl;
                                Mutex.unlock();
                                //*/

                        }}, tbb::simple_partitioner{});
                }}, tbb::simple_partitioner{});
    ///*
    jaccard_first.resize(jaccard_size);//
    jaccard_second.resize(jaccard_size);//
    jaccard_coefficients.resize(jaccard_size);//
    //*/

}


template<class NodeID_t>
void jaccard_all_row_true(
                const graph& g, 
                std::vector<NodeID_t>& jaccard_first,
                std::vector<NodeID_t>& jaccard_second,
                std::vector<float>& jaccard_coefficients,
                NodeID_t block_size_x, NodeID_t block_size_y) 
{

    size_t num_nodes = get_vertex_count( g);

    if (block_size_y > num_nodes) {
        block_size_y = num_nodes;
    }

    NodeID_t blocks_x = num_nodes;
    int number_of_threads = tbb::this_task_arena::max_concurrency();
    cout << "tbb threads " << number_of_threads << endl;
    std::vector< std::vector<NodeID_t>> blocks_first(number_of_threads, std::vector<NodeID_t> (block_size_y * block_size_x));
    std::vector< std::vector<NodeID_t>> blocks_second(number_of_threads, std::vector<NodeID_t> (block_size_y * block_size_x));
    std::vector< std::vector<float>> blocks_jaccards(number_of_threads, std::vector<float> (block_size_y * block_size_x));

    ///*
    size_t num_non_exists = (num_nodes * num_nodes - num_nodes) / 2 - get_edge_count(g);
    int64_t jaccard_size = 0;
    if (!jaccard_first.empty()) {
        jaccard_first.clear();
    }

    if (!jaccard_second.empty()) {
        jaccard_second.clear();
    }

    if (!jaccard_coefficients.empty()) {
        jaccard_coefficients.clear();
    }
    int32_t ratio_all_coefs_with_nnz_coeffs = 7; 
    jaccard_first.resize(num_non_exists / ratio_all_coefs_with_nnz_coeffs);
    jaccard_second.resize(num_non_exists / ratio_all_coefs_with_nnz_coeffs);
    jaccard_coefficients.resize(num_non_exists / ratio_all_coefs_with_nnz_coeffs);
    

    std::mutex Mutex;
    //*/
    //std::mutex Mutex;

    tbb::parallel_for(tbb::blocked_range<NodeID_t>(0, num_nodes - 1),
    [&](const tbb::blocked_range<int32_t>& r) {
        for (NodeID_t i = r.begin(); i != r.end(); ++i) {
    //for (NodeID_t i = 0; i < num_nodes - 1; ++i) {
            NodeID_t block_x_begin = i * block_size_x;
            NodeID_t block_x_end = block_x_begin + block_size_x;
            if ((i + 1) == blocks_x) {
                block_x_end = num_nodes;
            }

            NodeID_t y_start = block_x_begin + 1;
            NodeID_t blocks_y = (num_nodes - y_start) / block_size_y;
            if (block_size_y * blocks_y != (num_nodes - y_start)) {
                blocks_y++;
            }

            tbb::parallel_for(tbb::blocked_range<NodeID_t>(0, blocks_y),
                [&](const tbb::blocked_range<int32_t>& inner_r) {
                for (NodeID_t j = inner_r.begin(); j != inner_r.end(); ++j) {
                //for (NodeID_t j = 0; j < blocks_y; ++j) {
                    NodeID_t block_y_begin = y_start + j * block_size_y;
                    NodeID_t block_y_end = block_y_begin + block_size_y;
                    if ((j + 1) == blocks_y ) {
                        block_y_end = num_nodes;
                    }
                    NodeID_t jaccard_block_nnz = 0;
                    //Mutex.lock();
                     //cout << block_x_begin << " " << block_x_end << " : " <<
                      //       block_y_begin << " " << block_y_end << endl;
                    jaccard_block_avx512_true(g,
                                    block_x_begin, block_x_end,
                                    block_y_begin, block_y_end,
                                    blocks_first[tbb::this_task_arena::current_thread_index()], 
                                    blocks_second[tbb::this_task_arena::current_thread_index()], 
                                    blocks_jaccards[tbb::this_task_arena::current_thread_index()], 
                                    //blocks[0],
                                    jaccard_block_nnz);
                    //cout << " out : " << tbb::this_task_arena::current_thread_index() << endl;

                    ///*
                    Mutex.lock();
                    int64_t jaccard_block_size = jaccard_block_nnz;
                    if (jaccard_first.size() < jaccard_size + jaccard_block_size) {
                        jaccard_first.resize(jaccard_size + jaccard_block_size);
                    }
                    if (jaccard_second.size() < jaccard_size + jaccard_block_size) {
                        jaccard_second.resize(jaccard_size + jaccard_block_size);
                    }
                    if (jaccard_coefficients.size() < jaccard_size + jaccard_block_size) {
                        jaccard_coefficients.resize(jaccard_size + jaccard_block_size);
                    }                                        
                    //add_block_nonzero_coeffs_in_jaccard_vector(blocks[tbb::this_task_arena::current_thread_index()].data(), jaccard_block_size, jaccard.data(), jaccard_size, Mutex);
                    int count = 0;
                    //cout << jaccard_block_size << endl;
                    for (int64_t i = 0; i < jaccard_block_size; i++) {
                        if ((blocks_first[tbb::this_task_arena::current_thread_index()])[i] < (blocks_second[tbb::this_task_arena::current_thread_index()])[i]) {
                            jaccard_first[jaccard_size + count] = (blocks_first[tbb::this_task_arena::current_thread_index()])[i];
                            jaccard_second[jaccard_size + count] = (blocks_second[tbb::this_task_arena::current_thread_index()])[i];
                            jaccard_coefficients[jaccard_size + count] = (blocks_jaccards[tbb::this_task_arena::current_thread_index()])[i];
                            count++;
                        }
                    }
                    //cout << count << endl;
                    jaccard_size += count;
                    //cout << "ok" << endl;
                    // cout << block_x_begin << " " << block_x_end << " : " <<
                    //         block_y_begin << " " << block_y_end << endl;
                    Mutex.unlock();
                    //*/
            }}, tbb::simple_partitioner{});
    }}, tbb::auto_partitioner{});
    ///*
    jaccard_first.resize(jaccard_size);//
    jaccard_second.resize(jaccard_size);//
    jaccard_coefficients.resize(jaccard_size);//
    //*/
    //cout << "time_tail - " << time_tail << endl;
    //cout << "time_vect - " << time_vect << endl;
    //cout << "time_all - " << time_vect + time_tail << endl;
    //cout << "ratio time_tail / time_all - " << time_tail / (time_tail + time_vect) << endl;
};

template<class NodeID_t>
inline void jaccard_block_avx512_true(const graph& my_graph,
              NodeID_t vert00, NodeID_t vert01,
              NodeID_t vert10, NodeID_t vert11,
              std::vector<NodeID_t>& vertices_first,
              std::vector<NodeID_t>& vertices_second,
              std::vector<float>& jaccards,
              NodeID_t &jaccard_size) 
{
    auto g = oneapi::dal::preview::detail::get_impl(my_graph);
    jaccard_size = 0;
    for (NodeID_t i = vert00; i < vert01; i++) {
        NodeID_t size_i = g->_degrees[i];
        //auto n_i = g->_vertex_neighbors.data() + g->_edge_offsets[i];

        NodeID_t diagonal = std::min(i, vert11);

        for (NodeID_t j = vert10; j < diagonal; j++) {
            NodeID_t size_j = g->_degrees[j];             
            //auto n_j = g->_vertex_neighbors.data() + g->_edge_offsets[j];
            


            if (!(g->_vertex_neighbors[g->_edge_offsets[i]] > g->_vertex_neighbors[g->_edge_offsets[j + 1] - 1]) && 
                !(g->_vertex_neighbors[g->_edge_offsets[j]] > g->_vertex_neighbors[g->_edge_offsets[i + 1] - 1])) {

                size_t intersection_size = intersection_avx512((g->_vertex_neighbors.data() + g->_edge_offsets[i]), (g->_vertex_neighbors.data() +g->_edge_offsets[j]), size_i, size_j);

                if (intersection_size) {
                    vertices_first[jaccard_size] = i;
                    vertices_second[jaccard_size] = j;
                    jaccards[jaccard_size] = static_cast<float>(intersection_size) / static_cast<float>(size_i + size_j - intersection_size);
                    jaccard_size++;   
                }
            }
        }

        NodeID_t tmp_idx = vert10;
        if (diagonal >= vert10) {
            vertices_first[jaccard_size] = i;
            vertices_second[jaccard_size] = diagonal;
            jaccards[jaccard_size] = 1.0; 
            jaccard_size++;
            tmp_idx = diagonal + 1;
        }

        for (NodeID_t j = tmp_idx; j < vert11; j++) {
            NodeID_t size_j = g->_degrees[j];             
            //auto n_j = g->_vertex_neighbors.data() + g->_edge_offsets[j];
                // if (i == 0 && j < 10) {
                //     cout << i << " " << j << endl;
                //     cout << "offset i, i + 1 = " << g->_edge_offsets[i] << " " << g->_edge_offsets[i + 1] << " " << size_i << endl;
                //     cout << "offset j, j + 1 = " << g->_edge_offsets[j] << " " << g->_edge_offsets[j + 1] << " " << size_j << endl;
                //     cout << "neighbors i : ";
                //     for (int s = g->_edge_offsets[i]; s < g->_edge_offsets[i + 1]; s++) {
                //         cout << g->_vertex_neighbors[s] << " ";
                //     }
                //     cout << endl;

                //     cout << "neighbors j : ";
                //     for (int s = g->_edge_offsets[j]; s < g->_edge_offsets[j + 1]; s++) {
                //         cout << g->_vertex_neighbors[s] << " ";
                //     }
                //     cout << endl;                    

                // }

            if (!(g->_vertex_neighbors[g->_edge_offsets[i]] > g->_vertex_neighbors[g->_edge_offsets[j + 1] - 1]) && 
                !(g->_vertex_neighbors[g->_edge_offsets[j]] > g->_vertex_neighbors[g->_edge_offsets[i + 1] - 1])) {


                size_t intersection_size = intersection_avx512((g->_vertex_neighbors.data() + g->_edge_offsets[i]), (g->_vertex_neighbors.data() +g->_edge_offsets[j]), size_i, size_j);

                if (intersection_size) {
                    vertices_first[jaccard_size] = i;
                    vertices_second[jaccard_size] = j;
                    jaccards[jaccard_size] = static_cast<float>(intersection_size) / static_cast<float>(size_i + size_j - intersection_size);
                    jaccard_size++;   
                }
            }
        }
    }
}



template<class NodeID_t>
void jaccard_all_row_true(
                const graph& g, 
                std::vector<NodeID_t>& jaccard_first,
                std::vector<NodeID_t>& jaccard_second,
                std::vector<float>& jaccard_coefficients,
                NodeID_t block_size_x, NodeID_t block_size_y);

void verify_results(oneapi::dal::preview::graph g, vector<int32_t>& jaccard_first, vector<int32_t>& jaccard_second, vector<float>& jaccard_coefficients, string& path);

int main(int argc, char ** argv)
{
    tbb::task_scheduler_init init(1);

    int num_trials_custom = 10;
    int num_trials_lib = 1;
    int verify = 0;


    if (argc < 2) return 0;
    std::string filename = argv[1];
    csv_data_source ds(filename);
    edge_list_to_csr_descriptor d;
     auto my_graph = load_graph<graph>(d, ds);
    cout << "edges number" << get_edge_count( my_graph);
    auto layout = oneapi::dal::preview::detail::get_impl(my_graph);
    //layout->_vertex_neighbors; // dal::array -> std::vector
    //layout->_edge_offsets; // dal::array -> std::vector

    int32_t block_size_x = 1;
    int32_t block_size_y = 1024;

    if (argc > 5) {
        block_size_x = stoi(argv[2]);
        block_size_y = stoi(argv[3]);
        num_trials_custom = stoi(argv[4]);
        verify = stoi(argv[5]);
    }
 //   tbb::task_scheduler_init init(num_of_threads);

//     std::vector<jaccard_pair<int32_t>> jaccard;

///*
    std::vector<int32_t> jaccard_first;
    std::vector<int32_t> jaccard_second;
    std::vector<float> jaccard_coefficients;
//*/
    cout << "jaccard_non_existent with custom block_size = " << block_size_x <<"x" <<block_size_y << endl;
    vector<double> time;
    double median;

    for(int i = 0; i < num_trials_custom; i++) {
        auto start = high_resolution_clock::now();
        jaccard_all_row(my_graph, jaccard_first, jaccard_second, jaccard_coefficients, block_size_x, block_size_y);
        //jaccard_all_r(my_graph, block_size_x, block_size_y, jaccard_first, jaccard_second, jaccard_coefficients);
        //jaccard_status = jaccard_all_row(g, jaccard, block_size_x, block_size_y, 1);
        auto stop = high_resolution_clock::now();

        time.push_back(std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count());
        cout << i << " iter: " << time.back() << endl;
    }

// for (int i = 1; i < 5000; i *= 2) {
//     for (int j = max(i, 16); j < 40000; j *= 2) {
//         auto start = high_resolution_clock::now();
//         jaccard_status = jaccard_all_block(g, i, j, jaccard_first, jaccard_second, jaccard_coefficients);
//         auto stop = high_resolution_clock::now();
//         cout << "block " << i <<"x"<<j <<endl;
//         cout << "time       " << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count() <<endl;
//         cout << endl;
//     }
// }

    //computing time metrics
    sort(time.begin(), time.end());
    if (num_trials_custom % 2 == 0)
        median =  (time[time.size()/2] + time[time.size()/2 - 1]) / 2;
    else
        median = time[time.size()/2];
    cout <<"median: " << median << endl;
    cout <<"Min: " << time[0] << endl;
    cout <<"Max: " << time.back() << endl;

    string verify_file("/nfs/inn/proj/numerics1/Users/akumin/oneDAL/results_Enron.txt");
    if (verify) {
        jaccard_all_row_v(my_graph, jaccard_first, jaccard_second, jaccard_coefficients, block_size_x, block_size_y);
        verify_results(my_graph, jaccard_first, jaccard_second, jaccard_coefficients, verify_file);
    }

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

void verify_results(oneapi::dal::preview::graph g, vector<int32_t>& jaccard_first, vector<int32_t>& jaccard_second, vector<float>& jaccard_coefficients, string& path) {
    cout << "Results verification." << endl;

    auto layout = oneapi::dal::preview::detail::get_impl(g);
    layout->_vertex_neighbors; // dal::array -> std::vector
    layout->_edge_offsets; // dal::array -> std::vector

    vector<jaccard_pair> jaccard(jaccard_first.size());
    for (int i = 0 ; i < jaccard.size(); i++) {
        jaccard[i]= jaccard_pair(jaccard_first[i], jaccard_second[i], jaccard_coefficients[i]);
    }
    cout << " jaccard custom: " << jaccard.size() << endl;



    sort(jaccard.begin(), jaccard.end(), [] (const auto& lhs, const auto& rhs) { 
      if (lhs.first < rhs.first) {
          return true;
      }
      else if (lhs.first == rhs.first) {
              if (lhs.second < rhs.second) {
                  return true;
              }
              else
                  return false;
      }
      else{
          return false;
      }
      });

    
    std::vector<int32_t> jaccard_first1;
    std::vector<int32_t> jaccard_second1;
    std::vector<float> jaccard_coefficients1;
    jaccard_all_row_true(g, jaccard_first1, jaccard_second1, jaccard_coefficients1, 1, 2048);

    vector<jaccard_pair> jaccard_lib(jaccard_first1.size());
    for (int i = 0 ; i < jaccard_lib.size(); i++) {
        jaccard_lib[i]= jaccard_pair(jaccard_first1[i], jaccard_second1[i], jaccard_coefficients1[i]);
    }

    sort(jaccard_lib.begin(), jaccard_lib.end(), [] (const auto& lhs, const auto& rhs) { 
      if (lhs.first < rhs.first) {
          return true;
      }
      else if (lhs.first == rhs.first) {
              if (lhs.second < rhs.second) {
                  return true;
              }
              else
                  return false;
      }
      else{
          return false;
      }
      });

    // vector<jaccard_pair> jaccard_lib(0);
    // ifstream myfile;
    // myfile.open(path);
    // if (myfile.is_open())
    // {
    //     int size_jaccard = 0;
    //     myfile >> size_jaccard;
    //     jaccard_lib.resize(size_jaccard);
    //     for (int i = 0; i < jaccard_lib.size(); i++) {
    //         myfile >> jaccard_lib[i].first >> jaccard_lib[i].second >> jaccard_lib[i].coefficient;
    //     }
    //     myfile.close();
    // }



    cout << " jaccard lib: " << jaccard_lib.size() << endl;

    bool is_correct = true;
      float eps = 0.000001;
      for(int i = 0; i < jaccard_lib.size(); i++) {
          if (!((jaccard[i].coefficient < jaccard_lib[i].coefficient + eps) && (jaccard[i].coefficient > jaccard_lib[i].coefficient - eps) )) {
              cout << jaccard_lib[i].first << " " << jaccard_lib[i].second << " jaccard_lib: " <<
              jaccard_lib[i].coefficient << endl; 
              cout << jaccard[i].first << " " << jaccard[i].second << " jaccard: " <<
              jaccard[i].coefficient << endl;

              cout << "Neighbors " << jaccard_lib[i].first <<" " <<layout->_degrees[jaccard_lib[i].first] <<" : ";

              is_correct = false;
              for (int j = layout->_edge_offsets[jaccard_lib[i].first]; j < layout->_edge_offsets[jaccard_lib[i].first + 1]; j++)
              {
                  cout << layout->_vertex_neighbors[j] << " ";
              }
              cout << endl;

              cout << "Neighbors " << jaccard_lib[i].second << " : ";
              is_correct = false;
              for (int j = layout->_edge_offsets[jaccard_lib[i].second]; j < layout->_edge_offsets[jaccard_lib[i].second + 1]; j++)
              {
                  cout << layout->_vertex_neighbors[j] << " ";
              }
              cout << endl;

              cout << "Neighbors " << jaccard[i].first << " : ";
              is_correct = false;
              for (int j = layout->_edge_offsets[jaccard[i].first]; j < layout->_edge_offsets[jaccard[i].first + 1]; j++)
              {
                  cout << layout->_vertex_neighbors[j] << " ";
              }
              cout << endl;

              cout << "Neighbors " << jaccard[i].second << " : ";

              is_correct = false;
              for (int j = layout->_edge_offsets[jaccard[i].second]; j < layout->_edge_offsets[jaccard[i].second + 1]; j++)
              {
                  cout << layout->_vertex_neighbors[j] << " ";
              }
              cout << endl;

              for (int j = max(0, i - 5) ; j < i + 5; j++)
              {
                  cout << jaccard[j].first << " " << jaccard[j].second << " jaccard: " <<
                  jaccard[j].coefficient << "            ";

                  cout << jaccard_lib[j].first << " " << jaccard_lib[j].second << " jaccard_lib: " <<
                  jaccard_lib[j].coefficient << endl;
              }

              break;
          }
      }
  if (is_correct == true) {
      cout << "Succesfull." << endl;
  }
  else {
      cout << "Failed." << endl;
  }

}

template<class NodeID_t>
void jaccard_all_row_v(
                const graph& g, 
                std::vector<NodeID_t>& jaccard_first,
                std::vector<NodeID_t>& jaccard_second,
                std::vector<float>& jaccard_coefficients,
                NodeID_t block_size_x, NodeID_t block_size_y) 
{

    size_t num_nodes = get_vertex_count( g);

    if (block_size_y > num_nodes) {
        block_size_y = num_nodes;
    }

    NodeID_t blocks_x = num_nodes;
    int number_of_threads = tbb::this_task_arena::max_concurrency();
    cout << "tbb threads " << number_of_threads << endl;
    std::vector< std::vector<NodeID_t>> blocks_first(number_of_threads, std::vector<NodeID_t> (block_size_y * block_size_x));
    std::vector< std::vector<NodeID_t>> blocks_second(number_of_threads, std::vector<NodeID_t> (block_size_y * block_size_x));
    std::vector< std::vector<float>> blocks_jaccards(number_of_threads, std::vector<float> (block_size_y * block_size_x));

    ///*
    size_t num_non_exists = (num_nodes * num_nodes - num_nodes) / 2 - get_edge_count(g);
    int64_t jaccard_size = 0;
    if (!jaccard_first.empty()) {
        jaccard_first.clear();
    }

    if (!jaccard_second.empty()) {
        jaccard_second.clear();
    }

    if (!jaccard_coefficients.empty()) {
        jaccard_coefficients.clear();
    }
    int32_t ratio_all_coefs_with_nnz_coeffs = 7; 
    jaccard_first.resize(num_non_exists / ratio_all_coefs_with_nnz_coeffs);
    jaccard_second.resize(num_non_exists / ratio_all_coefs_with_nnz_coeffs);
    jaccard_coefficients.resize(num_non_exists / ratio_all_coefs_with_nnz_coeffs);
    

    std::mutex Mutex;
    //*/
    //std::mutex Mutex;

    tbb::parallel_for(tbb::blocked_range<NodeID_t>(0, num_nodes - 1),
    [&](const tbb::blocked_range<int32_t>& r) {
        for (NodeID_t i = r.begin(); i != r.end(); ++i) {
    //for (NodeID_t i = 0; i < num_nodes - 1; ++i) {
            NodeID_t block_x_begin = i * block_size_x;
            NodeID_t block_x_end = block_x_begin + block_size_x;
            if ((i + 1) == blocks_x) {
                block_x_end = num_nodes;
            }

            NodeID_t y_start = block_x_begin + 1;
            NodeID_t blocks_y = (num_nodes - y_start) / block_size_y;
            if (block_size_y * blocks_y != (num_nodes - y_start)) {
                blocks_y++;
            }

            tbb::parallel_for(tbb::blocked_range<NodeID_t>(0, blocks_y),
                [&](const tbb::blocked_range<int32_t>& inner_r) {
                for (NodeID_t j = inner_r.begin(); j != inner_r.end(); ++j) {
                //for (NodeID_t j = 0; j < blocks_y; ++j) {
                    NodeID_t block_y_begin = y_start + j * block_size_y;
                    NodeID_t block_y_end = block_y_begin + block_size_y;
                    if ((j + 1) == blocks_y ) {
                        block_y_end = num_nodes;
                    }
                    NodeID_t jaccard_block_nnz = 0;
                    //Mutex.lock();
                     //cout << block_x_begin << " " << block_x_end << " : " <<
                      //       block_y_begin << " " << block_y_end << endl;
                    jaccard_block_avx512(g,
                                    block_x_begin, block_x_end,
                                    block_y_begin, block_y_end,
                                    blocks_first[tbb::this_task_arena::current_thread_index()], 
                                    blocks_second[tbb::this_task_arena::current_thread_index()], 
                                    blocks_jaccards[tbb::this_task_arena::current_thread_index()], 
                                    //blocks[0],
                                    jaccard_block_nnz);
                    //cout << " out : " << tbb::this_task_arena::current_thread_index() << endl;

                    ///*
                    Mutex.lock();
                    int64_t jaccard_block_size = jaccard_block_nnz;
                    if (jaccard_first.size() < jaccard_size + jaccard_block_size) {
                        jaccard_first.resize(jaccard_size + jaccard_block_size);
                    }
                    if (jaccard_second.size() < jaccard_size + jaccard_block_size) {
                        jaccard_second.resize(jaccard_size + jaccard_block_size);
                    }
                    if (jaccard_coefficients.size() < jaccard_size + jaccard_block_size) {
                        jaccard_coefficients.resize(jaccard_size + jaccard_block_size);
                    }                                        
                    //add_block_nonzero_coeffs_in_jaccard_vector(blocks[tbb::this_task_arena::current_thread_index()].data(), jaccard_block_size, jaccard.data(), jaccard_size, Mutex);
                    int count = 0;
                    //cout << jaccard_block_size << endl;
                    for (int64_t i = 0; i < jaccard_block_size; i++) {
                        if ((blocks_first[tbb::this_task_arena::current_thread_index()])[i] < (blocks_second[tbb::this_task_arena::current_thread_index()])[i]) {
                            jaccard_first[jaccard_size + count] = (blocks_first[tbb::this_task_arena::current_thread_index()])[i];
                            jaccard_second[jaccard_size + count] = (blocks_second[tbb::this_task_arena::current_thread_index()])[i];
                            jaccard_coefficients[jaccard_size + count] = (blocks_jaccards[tbb::this_task_arena::current_thread_index()])[i];
                            count++;
                        }
                    }
                    //cout << count << endl;
                    jaccard_size += count;
                    //cout << "ok" << endl;
                    // cout << block_x_begin << " " << block_x_end << " : " <<
                    //         block_y_begin << " " << block_y_end << endl;
                    Mutex.unlock();
                    //*/
            }}, tbb::simple_partitioner{});
    }}, tbb::auto_partitioner{});
    ///*
    jaccard_first.resize(jaccard_size);//
    jaccard_second.resize(jaccard_size);//
    jaccard_coefficients.resize(jaccard_size);//
    //*/
    //cout << "time_tail - " << time_tail << endl;
    //cout << "time_vect - " << time_vect << endl;
    //cout << "time_all - " << time_vect + time_tail << endl;
    //cout << "ratio time_tail / time_all - " << time_tail / (time_tail + time_vect) << endl;
};


template<class NodeID_t>
void jaccard_all_row_true(
                const graph& g, 
                std::vector<NodeID_t>& jaccard_first,
                std::vector<NodeID_t>& jaccard_second,
                std::vector<float>& jaccard_coefficients,
                NodeID_t block_size_x, NodeID_t block_size_y) 
{

    size_t num_nodes = get_vertex_count( g);

    if (block_size_y > num_nodes) {
        block_size_y = num_nodes;
    }

    NodeID_t blocks_x = num_nodes;
    int number_of_threads = tbb::this_task_arena::max_concurrency();
    cout << "tbb threads " << number_of_threads << endl;
    std::vector< std::vector<NodeID_t>> blocks_first(number_of_threads, std::vector<NodeID_t> (block_size_y * block_size_x));
    std::vector< std::vector<NodeID_t>> blocks_second(number_of_threads, std::vector<NodeID_t> (block_size_y * block_size_x));
    std::vector< std::vector<float>> blocks_jaccards(number_of_threads, std::vector<float> (block_size_y * block_size_x));

    ///*
    size_t num_non_exists = (num_nodes * num_nodes - num_nodes) / 2 - get_edge_count(g);
    int64_t jaccard_size = 0;
    if (!jaccard_first.empty()) {
        jaccard_first.clear();
    }

    if (!jaccard_second.empty()) {
        jaccard_second.clear();
    }

    if (!jaccard_coefficients.empty()) {
        jaccard_coefficients.clear();
    }
    int32_t ratio_all_coefs_with_nnz_coeffs = 7; 
    jaccard_first.resize(num_non_exists / ratio_all_coefs_with_nnz_coeffs);
    jaccard_second.resize(num_non_exists / ratio_all_coefs_with_nnz_coeffs);
    jaccard_coefficients.resize(num_non_exists / ratio_all_coefs_with_nnz_coeffs);
    

    std::mutex Mutex;
    //*/
    //std::mutex Mutex;

    tbb::parallel_for(tbb::blocked_range<NodeID_t>(0, num_nodes - 1),
    [&](const tbb::blocked_range<int32_t>& r) {
        for (NodeID_t i = r.begin(); i != r.end(); ++i) {
    //for (NodeID_t i = 0; i < num_nodes - 1; ++i) {
            NodeID_t block_x_begin = i * block_size_x;
            NodeID_t block_x_end = block_x_begin + block_size_x;
            if ((i + 1) == blocks_x) {
                block_x_end = num_nodes;
            }

            NodeID_t y_start = block_x_begin + 1;
            NodeID_t blocks_y = (num_nodes - y_start) / block_size_y;
            if (block_size_y * blocks_y != (num_nodes - y_start)) {
                blocks_y++;
            }

            tbb::parallel_for(tbb::blocked_range<NodeID_t>(0, blocks_y),
                [&](const tbb::blocked_range<int32_t>& inner_r) {
                for (NodeID_t j = inner_r.begin(); j != inner_r.end(); ++j) {
                //for (NodeID_t j = 0; j < blocks_y; ++j) {
                    NodeID_t block_y_begin = y_start + j * block_size_y;
                    NodeID_t block_y_end = block_y_begin + block_size_y;
                    if ((j + 1) == blocks_y ) {
                        block_y_end = num_nodes;
                    }
                    NodeID_t jaccard_block_nnz = 0;
                    //Mutex.lock();
                     //cout << block_x_begin << " " << block_x_end << " : " <<
                      //       block_y_begin << " " << block_y_end << endl;
                    jaccard_block_avx512_true(g,
                                    block_x_begin, block_x_end,
                                    block_y_begin, block_y_end,
                                    blocks_first[tbb::this_task_arena::current_thread_index()], 
                                    blocks_second[tbb::this_task_arena::current_thread_index()], 
                                    blocks_jaccards[tbb::this_task_arena::current_thread_index()], 
                                    //blocks[0],
                                    jaccard_block_nnz);
                    //cout << " out : " << tbb::this_task_arena::current_thread_index() << endl;

                    ///*
                    Mutex.lock();
                    int64_t jaccard_block_size = jaccard_block_nnz;
                    if (jaccard_first.size() < jaccard_size + jaccard_block_size) {
                        jaccard_first.resize(jaccard_size + jaccard_block_size);
                    }
                    if (jaccard_second.size() < jaccard_size + jaccard_block_size) {
                        jaccard_second.resize(jaccard_size + jaccard_block_size);
                    }
                    if (jaccard_coefficients.size() < jaccard_size + jaccard_block_size) {
                        jaccard_coefficients.resize(jaccard_size + jaccard_block_size);
                    }                                        
                    //add_block_nonzero_coeffs_in_jaccard_vector(blocks[tbb::this_task_arena::current_thread_index()].data(), jaccard_block_size, jaccard.data(), jaccard_size, Mutex);
                    int count = 0;
                    //cout << jaccard_block_size << endl;
                    for (int64_t i = 0; i < jaccard_block_size; i++) {
                        if ((blocks_first[tbb::this_task_arena::current_thread_index()])[i] < (blocks_second[tbb::this_task_arena::current_thread_index()])[i]) {
                            jaccard_first[jaccard_size + count] = (blocks_first[tbb::this_task_arena::current_thread_index()])[i];
                            jaccard_second[jaccard_size + count] = (blocks_second[tbb::this_task_arena::current_thread_index()])[i];
                            jaccard_coefficients[jaccard_size + count] = (blocks_jaccards[tbb::this_task_arena::current_thread_index()])[i];
                            count++;
                        }
                    }
                    //cout << count << endl;
                    jaccard_size += count;
                    //cout << "ok" << endl;
                    // cout << block_x_begin << " " << block_x_end << " : " <<
                    //         block_y_begin << " " << block_y_end << endl;
                    Mutex.unlock();
                    //*/
            }}, tbb::simple_partitioner{});
    }}, tbb::auto_partitioner{});
    ///*
    jaccard_first.resize(jaccard_size);//
    jaccard_second.resize(jaccard_size);//
    jaccard_coefficients.resize(jaccard_size);//
    //*/
    //cout << "time_tail - " << time_tail << endl;
    //cout << "time_vect - " << time_vect << endl;
    //cout << "time_all - " << time_vect + time_tail << endl;
    //cout << "ratio time_tail / time_all - " << time_tail / (time_tail + time_vect) << endl;
};

template<class NodeID_t>
inline void jaccard_block_avx512_true(const graph& my_graph,
              NodeID_t vert00, NodeID_t vert01,
              NodeID_t vert10, NodeID_t vert11,
              std::vector<NodeID_t>& vertices_first,
              std::vector<NodeID_t>& vertices_second,
              std::vector<float>& jaccards,
              NodeID_t &jaccard_size) 
{
    auto g = oneapi::dal::preview::detail::get_impl(my_graph);
    jaccard_size = 0;
    for (NodeID_t i = vert00; i < vert01; i++) {
        NodeID_t size_i = g->_degrees[i];
        //auto n_i = g->_vertex_neighbors.data() + g->_edge_offsets[i];

        NodeID_t diagonal = std::min(i, vert11);

        for (NodeID_t j = vert10; j < diagonal; j++) {
            NodeID_t size_j = g->_degrees[j];             
            //auto n_j = g->_vertex_neighbors.data() + g->_edge_offsets[j];
            


            if (!(g->_vertex_neighbors[g->_edge_offsets[i]] > g->_vertex_neighbors[g->_edge_offsets[j + 1] - 1]) && 
                !(g->_vertex_neighbors[g->_edge_offsets[j]] > g->_vertex_neighbors[g->_edge_offsets[i + 1] - 1])) {

                size_t intersection_size = intersection_avx512((g->_vertex_neighbors.data() + g->_edge_offsets[i]), (g->_vertex_neighbors.data() +g->_edge_offsets[j]), size_i, size_j);

                if (intersection_size) {
                    vertices_first[jaccard_size] = i;
                    vertices_second[jaccard_size] = j;
                    jaccards[jaccard_size] = static_cast<float>(intersection_size) / static_cast<float>(size_i + size_j - intersection_size);
                    jaccard_size++;   
                }
            }
        }

        NodeID_t tmp_idx = vert10;
        if (diagonal >= vert10) {
            vertices_first[jaccard_size] = i;
            vertices_second[jaccard_size] = diagonal;
            jaccards[jaccard_size] = 1.0; 
            jaccard_size++;
            tmp_idx = diagonal + 1;
        }

        for (NodeID_t j = tmp_idx; j < vert11; j++) {
            NodeID_t size_j = g->_degrees[j];             
            //auto n_j = g->_vertex_neighbors.data() + g->_edge_offsets[j];
                // if (i == 0 && j < 10) {
                //     cout << i << " " << j << endl;
                //     cout << "offset i, i + 1 = " << g->_edge_offsets[i] << " " << g->_edge_offsets[i + 1] << " " << size_i << endl;
                //     cout << "offset j, j + 1 = " << g->_edge_offsets[j] << " " << g->_edge_offsets[j + 1] << " " << size_j << endl;
                //     cout << "neighbors i : ";
                //     for (int s = g->_edge_offsets[i]; s < g->_edge_offsets[i + 1]; s++) {
                //         cout << g->_vertex_neighbors[s] << " ";
                //     }
                //     cout << endl;

                //     cout << "neighbors j : ";
                //     for (int s = g->_edge_offsets[j]; s < g->_edge_offsets[j + 1]; s++) {
                //         cout << g->_vertex_neighbors[s] << " ";
                //     }
                //     cout << endl;                    

                // }

            if (!(g->_vertex_neighbors[g->_edge_offsets[i]] > g->_vertex_neighbors[g->_edge_offsets[j + 1] - 1]) && 
                !(g->_vertex_neighbors[g->_edge_offsets[j]] > g->_vertex_neighbors[g->_edge_offsets[i + 1] - 1])) {


                size_t intersection_size = intersection_avx512((g->_vertex_neighbors.data() + g->_edge_offsets[i]), (g->_vertex_neighbors.data() +g->_edge_offsets[j]), size_i, size_j);

                if (intersection_size) {
                    vertices_first[jaccard_size] = i;
                    vertices_second[jaccard_size] = j;
                    jaccards[jaccard_size] = static_cast<float>(intersection_size) / static_cast<float>(size_i + size_j - intersection_size);
                    jaccard_size++;   
                }
            }
        }
    }
}