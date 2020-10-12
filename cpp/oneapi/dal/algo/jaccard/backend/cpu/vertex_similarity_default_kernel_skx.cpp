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

#include <immintrin.h>

#include <daal/src/services/service_defines.h>

#include "oneapi/dal/algo/jaccard/backend/cpu/vertex_similarity_default_kernel.hpp"
#include "oneapi/dal/algo/jaccard/common.hpp"
#include "oneapi/dal/algo/jaccard/vertex_similarity_types.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/graph/detail/graph_service_functions_impl.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::preview {
namespace jaccard {
namespace detail {

#if defined(__INTEL_COMPILER)
DAAL_FORCEINLINE std::int32_t _popcnt32_redef(const std::int32_t &x) {
    return _popcnt32(x);
}
#define GRAPH_STACK_ALING(x) __declspec(align(x))
#else
DAAL_FORCEINLINE std::int32_t _popcnt32_redef(const std::int32_t &x) {
    std::int32_t count = 0;
    std::int32_t a = x;
    while (a != 0) {
        a = a & (a - 1);
        count++;
    }
    return count;
}
#define GRAPH_STACK_ALING(x) \
    {}
#endif

DAAL_FORCEINLINE std::size_t intersection(std::int32_t *neigh_u,
                                          std::int32_t *neigh_v,
                                          std::int32_t n_u,
                                          std::int32_t n_v) {
    size_t total = 0;
    std::int32_t i_u = 0, i_v = 0;
#if defined(__INTEL_COMPILER)
    while (i_u < (n_u / 16) * 16 && i_v < (n_v / 16) * 16) { // not in last n%16 elements
        // assumes neighbor list is ordered
        std::int32_t minu = neigh_u[i_u];
        std::int32_t maxv = neigh_v[i_v + 15];

        if (minu > maxv) {
            if (minu > neigh_v[n_v - 1]) {
                return total;
            }
            i_v += 16;
            continue;
        }

        std::int32_t minv = neigh_v[i_v];
        std::int32_t maxu = neigh_u[i_u + 15];
        if (minv > maxu) {
            if (minv > neigh_u[n_u - 1]) {
                return total;
            }
            i_u += 16;
            continue;
        }
        __m512i v_u = _mm512_loadu_si512((void *)(neigh_u + i_u)); // load 16 neighbors of u
        __m512i v_v = _mm512_loadu_si512((void *)(neigh_v + i_v)); // load 16 neighbors of v
        if (maxu >= maxv)
            i_v += 16;
        if (maxu <= maxv)
            i_u += 16;

        __mmask16 match = _mm512_cmpeq_epi32_mask(v_u, v_v);
        if (_mm512_mask2int(match) != 0xffff) { // shortcut case where all neighbors match
            __m512i circ1 = _mm512_set_epi32(0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
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
            __mmask16 tmp_match1 = _mm512_cmpeq_epi32_mask(v_u, v_v1); // find matches
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
            match = _mm512_kor(
                _mm512_kor(
                    _mm512_kor(_mm512_kor(match, tmp_match1), _mm512_kor(tmp_match2, tmp_match3)),
                    _mm512_kor(_mm512_kor(tmp_match4, tmp_match5),
                               _mm512_kor(tmp_match6, tmp_match7))),
                _mm512_kor(
                    _mm512_kor(_mm512_kor(tmp_match8, tmp_match9),
                               _mm512_kor(tmp_match10, tmp_match11)),
                    _mm512_kor(_mm512_kor(tmp_match12, tmp_match13),
                               _mm512_kor(tmp_match14, tmp_match15)))); // combine all matches
        }
        total += _popcnt32_redef(_mm512_mask2int(match)); //count number of matches
    }

    while (i_u < (n_u / 16) * 16 && i_v < n_v) {
        __m512i v_u = _mm512_loadu_si512((void *)(neigh_u + i_u));
        while (neigh_v[i_v] <= neigh_u[i_u + 15] && i_v < n_v) {
            __m512i tmp_v_v = _mm512_set1_epi32(neigh_v[i_v]);
            __mmask16 match = _mm512_cmpeq_epi32_mask(v_u, tmp_v_v);
            if (_mm512_mask2int(match))
                total++;
            i_v++;
        }
        i_u += 16;
    }
    while (i_v < (n_v / 16) * 16 && i_u < n_u) {
        __m512i v_v = _mm512_loadu_si512((void *)(neigh_v + i_v));
        while (neigh_u[i_u] <= neigh_v[i_v + 15] && i_u < n_u) {
            __m512i tmp_v_u = _mm512_set1_epi32(neigh_u[i_u]);
            __mmask16 match = _mm512_cmpeq_epi32_mask(v_v, tmp_v_u);
            if (_mm512_mask2int(match))
                total++;
            i_u++;
        }
        i_v += 16;
    }

    while (i_u <= (n_u - 8) && i_v <= (n_v - 8)) { // not in last n%8 elements
        // assumes neighbor list is ordered
        std::int32_t minu = neigh_u[i_u];
        std::int32_t maxv = neigh_v[i_v + 7];

        if (minu > maxv) {
            if (minu > neigh_v[n_v - 1]) {
                return total;
            }
            i_v += 8;
            continue;
        }
        std::int32_t maxu = neigh_u[i_u + 7];
        std::int32_t minv = neigh_v[i_v];
        if (minv > maxu) {
            if (minv > neigh_u[n_u - 1]) {
                return total;
            }
            i_u += 8;
            continue;
        }
        __m256i v_u = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(neigh_u + i_u)); // load 8 neighbors of u
        __m256i v_v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(neigh_v + i_v)); // load 8 neighbors of v

        if (maxu >= maxv)
            i_v += 8;
        if (maxu <= maxv)
            i_u += 8;

        __mmask8 match = _mm256_cmpeq_epi32_mask(v_u, v_v);
        if (_cvtmask8_u32(match) != 0xff) { // shortcut case where all neighbors match
            __m256i circ1 = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);
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

            __mmask8 tmp_match1 = _mm256_cmpeq_epi32_mask(v_u, v_v1); // find matches
            __mmask8 tmp_match2 = _mm256_cmpeq_epi32_mask(v_u, v_v2);
            __mmask8 tmp_match3 = _mm256_cmpeq_epi32_mask(v_u, v_v3);
            __mmask8 tmp_match4 = _mm256_cmpeq_epi32_mask(v_u, v_v4);
            __mmask8 tmp_match5 = _mm256_cmpeq_epi32_mask(v_u, v_v5);
            __mmask8 tmp_match6 = _mm256_cmpeq_epi32_mask(v_u, v_v6);
            __mmask8 tmp_match7 = _mm256_cmpeq_epi32_mask(v_u, v_v7);

            match = _kor_mask8(
                _kor_mask8(_kor_mask8(match, tmp_match1), _kor_mask8(tmp_match2, tmp_match3)),
                _kor_mask8(_kor_mask8(tmp_match4, tmp_match5),
                           _kor_mask8(tmp_match6, tmp_match7))); // combine all matches
        }
        total += _popcnt32_redef(_cvtmask8_u32(match)); //count number of matches
    }
    if (i_u <= (n_u - 8) && i_v < n_v) {
        __m256i v_u = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(neigh_u + i_u));
        while (neigh_v[i_v] <= neigh_u[i_u + 7] && i_v < n_v) {
            __m256i tmp_v_v = _mm256_set1_epi32(neigh_v[i_v]);
            __mmask8 match = _mm256_cmpeq_epi32_mask(v_u, tmp_v_v);
            if (_cvtmask8_u32(match))
                total++;
            i_v++;
        }
        i_u += 8;
    }
    if (i_v <= (n_v - 8) && i_u < n_u) {
        __m256i v_v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(neigh_v + i_v));
        while (neigh_u[i_u] <= neigh_v[i_v + 7] && i_u < n_u) {
            __m256i tmp_v_u = _mm256_set1_epi32(neigh_u[i_u]);
            __mmask8 match = _mm256_cmpeq_epi32_mask(v_v, tmp_v_u);
            if (_cvtmask8_u32(match))
                total++;
            i_u++;
        }
        i_v += 8;
    }

    while (i_u <= (n_u - 4) && i_v <= (n_v - 4)) { // not in last n%8 elements
        // assumes neighbor list is ordered
        std::int32_t minu = neigh_u[i_u];
        std::int32_t maxv = neigh_v[i_v + 3];

        if (minu > maxv) {
            if (minu > neigh_v[n_v - 1]) {
                return total;
            }
            i_v += 4;
            continue;
        }
        std::int32_t minv = neigh_v[i_v];
        std::int32_t maxu = neigh_u[i_u + 3];
        if (minv > maxu) {
            if (minv > neigh_u[n_u - 1]) {
                return total;
            }
            i_u += 4;
            continue;
        }
        __m128i v_u = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(neigh_u + i_u)); // load 8 neighbors of u
        __m128i v_v = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(neigh_v + i_v)); // load 8 neighbors of v

        if (maxu >= maxv)
            i_v += 4;
        if (maxu <= maxv)
            i_u += 4;

        __mmask8 match = _mm_cmpeq_epi32_mask(v_u, v_v);
        if (_cvtmask8_u32(match) != 0xf) { // shortcut case where all neighbors match
            __m128i v_v1 = _mm_shuffle_epi32(v_v, _MM_SHUFFLE(0, 3, 2, 1));
            __m128i v_v2 = _mm_shuffle_epi32(v_v, _MM_SHUFFLE(1, 0, 3, 2));
            __m128i v_v3 = _mm_shuffle_epi32(v_v, _MM_SHUFFLE(2, 1, 0, 3));

            __mmask8 tmp_match1 = _mm_cmpeq_epi32_mask(v_u, v_v1); // find matches
            __mmask8 tmp_match2 = _mm_cmpeq_epi32_mask(v_u, v_v2);
            __mmask8 tmp_match3 = _mm_cmpeq_epi32_mask(v_u, v_v3);

            match = _kor_mask8(_kor_mask8(match, tmp_match1),
                               _kor_mask8(tmp_match2, tmp_match3)); // combine all matches
        }
        total += _popcnt32_redef(_cvtmask8_u32(match)); //count number of matches
    }
    if (i_u <= (n_u - 4) && i_v < n_v) {
        __m128i v_u = _mm_loadu_si128(reinterpret_cast<const __m128i *>(neigh_u + i_u));
        while (neigh_v[i_v] <= neigh_u[i_u + 3] && i_v < n_v) {
            __m128i tmp_v_v = _mm_set1_epi32(neigh_v[i_v]);
            __mmask8 match = _mm_cmpeq_epi32_mask(v_u, tmp_v_v);
            if (_cvtmask8_u32(match))
                total++;
            i_v++;
        }
        i_u += 4;
    }
    if (i_v <= (n_v - 4) && i_u < n_u) {
        __m128i v_v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(neigh_v + i_v));
        while (neigh_u[i_u] <= neigh_v[i_v + 3] && i_u < n_u) {
            __m128i tmp_v_u = _mm_set1_epi32(neigh_u[i_u]);
            __mmask8 match = _mm_cmpeq_epi32_mask(v_v, tmp_v_u);
            if (_cvtmask8_u32(match))
                total++;
            i_u++;
        }
        i_v += 4;
    }
#endif
    while (i_u < n_u && i_v < n_v) {
        if ((neigh_u[i_u] > neigh_v[n_v - 1]) || (neigh_v[i_v] > neigh_u[n_u - 1])) {
            return total;
        }
        if (neigh_u[i_u] == neigh_v[i_v])
            total++, i_u++, i_v++;
        else if (neigh_u[i_u] < neigh_v[i_v])
            i_u++;
        else if (neigh_u[i_u] > neigh_v[i_v])
            i_v++;
    }
    return total;
}

template <>
vertex_similarity_result call_jaccard_default_kernel<undirected_adjacency_array_graph<>,
                                                     oneapi::dal::backend::cpu_dispatch_avx512>(
    const descriptor_base &desc,
    vertex_similarity_input<undirected_adjacency_array_graph<>> &input) {
    const auto &my_graph = input.get_graph();
    const auto &g = oneapi::dal::preview::detail::get_impl(my_graph);

    auto g_edge_offsets = g->_edge_offsets.data();
    auto g_vertex_neighbors = g->_vertex_neighbors.data();
    auto g_degrees = g->_degrees.data();

    const auto row_begin = static_cast<std::int32_t>(desc.get_row_range_begin());
    const auto row_end = static_cast<std::int32_t>(desc.get_row_range_end());
    const auto column_begin = static_cast<std::int32_t>(desc.get_column_range_begin());
    const auto column_end = static_cast<std::int32_t>(desc.get_column_range_end());

    const auto number_elements_in_block = (row_end - row_begin) * (column_end - column_begin);
    const size_t max_block_size =
        compute_max_block_size(row_begin, row_end, column_begin, column_end);

    void *result_ptr = input.get_caching_builder()(max_block_size);
    std::int32_t *first_vertices = reinterpret_cast<std::int32_t *>(result_ptr);
    std::int32_t *second_vertices = first_vertices + number_elements_in_block;
    float *jaccard = reinterpret_cast<float *>(first_vertices + 2 * number_elements_in_block);

    std::int64_t nnz = 0;
    std::int32_t j = column_begin;
#if defined(__INTEL_COMPILER)
    __m512i j_vertices_tmp1 =
        _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    GRAPH_STACK_ALING(64) std::int32_t stack16_j_vertex[16] = { 0 };

    std::int32_t ones_num = 0;
#endif

    for (std::int32_t i = row_begin; i < row_end; ++i) {
        const auto i_neighbor_size = g_degrees[i];
        const auto i_neigbhors = g_vertex_neighbors + g_edge_offsets[i];
        const auto diagonal = min(i, column_end);

#if defined(__INTEL_COMPILER)
        __m512i n_i_start_v = _mm512_set1_epi32(i_neigbhors[0]);
        __m512i n_i_end_v = _mm512_set1_epi32(i_neigbhors[i_neighbor_size - 1]);
        __m512i i_vertex = _mm512_set1_epi32(i);

        if (j < column_begin + ((diagonal - column_begin) / 16) * 16) {
            //load_data(0)
            __m512i start_indices_j_v = _mm512_load_epi32(g_edge_offsets + j);
            __m512i end_indices_j_v_tmp = _mm512_load_epi32(g_edge_offsets + j + 1);
            __m512i end_indices_j_v = _mm512_add_epi32(end_indices_j_v_tmp, _mm512_set1_epi32(-1));

            __m512i n_j_start_v = _mm512_permutexvar_epi32(
                j_vertices_tmp1,
                _mm512_i32gather_epi32(start_indices_j_v, g_vertex_neighbors, 4));
            __m512i n_j_end_v = _mm512_permutexvar_epi32(
                j_vertices_tmp1,
                _mm512_i32gather_epi32(end_indices_j_v, g_vertex_neighbors, 4));

            for (; j + 16 < column_begin + ((diagonal - column_begin) / 16) * 16;) {
                __m512i start_indices_j_v = _mm512_load_epi32(g_edge_offsets + j + 16);
                __m512i end_indices_j_v_tmp = _mm512_load_epi32(g_edge_offsets + j + 17);
                __m512i end_indices_j_v =
                    _mm512_add_epi32(end_indices_j_v_tmp, _mm512_set1_epi32(-1));

                __m512i n_j_start_v1 = _mm512_permutexvar_epi32(
                    j_vertices_tmp1,
                    _mm512_i32gather_epi32(start_indices_j_v, g_vertex_neighbors, 4));
                __m512i n_j_end_v1 = _mm512_permutexvar_epi32(
                    j_vertices_tmp1,
                    _mm512_i32gather_epi32(end_indices_j_v, g_vertex_neighbors, 4));

                __mmask16 cmpgt1 = _mm512_cmpgt_epi32_mask(n_i_start_v, n_j_end_v);
                __mmask16 cmpgt2 = _mm512_cmpgt_epi32_mask(n_j_start_v, n_i_end_v);

                __mmask16 worth_intersection = _mm512_knot(_mm512_kor(cmpgt1, cmpgt2));
                ones_num = _popcnt32_redef(_cvtmask16_u32(worth_intersection));

                if (ones_num != 0) {
                    __m512i j_vertices_tmp2 = _mm512_set1_epi32(j);
                    __m512i j_vertices = _mm512_add_epi32(j_vertices_tmp1, j_vertices_tmp2);

                    _mm512_mask_compressstoreu_epi32((stack16_j_vertex),
                                                     worth_intersection,
                                                     j_vertices);

                    GRAPH_STACK_ALING(64) std::int32_t stack16_intersections[16] = { 0 };
                    for (std::int32_t s = 0; s < ones_num; s++) {
                        const auto j_neighbor_size = g_degrees[stack16_j_vertex[s]];
                        const auto j_neigbhors =
                            g_vertex_neighbors + g_edge_offsets[stack16_j_vertex[s]];
                        stack16_intersections[s] = intersection(i_neigbhors,
                                                                j_neigbhors,
                                                                i_neighbor_size,
                                                                j_neighbor_size);
                    }
                    __m512i intersections_v = _mm512_load_epi32(stack16_intersections);
                    j_vertices = _mm512_load_epi32(stack16_j_vertex);

                    __mmask16 non_zero_coefficients =
                        _mm512_test_epi32_mask(intersections_v, intersections_v);
                    _mm512_mask_compressstoreu_epi32((first_vertices + nnz),
                                                     non_zero_coefficients,
                                                     i_vertex);
                    _mm512_mask_compressstoreu_epi32((second_vertices + nnz),
                                                     non_zero_coefficients,
                                                     j_vertices);
                    __m512 tmp_v = _mm512_cvtepi32_ps(intersections_v);
                    _mm512_mask_compressstoreu_ps((jaccard + nnz), non_zero_coefficients, tmp_v);

                    nnz += _popcnt32_redef(_cvtmask16_u32(non_zero_coefficients));
                }

                j += 16;

                n_j_start_v = n_j_start_v1;
                n_j_end_v = n_j_end_v1;
            }

            //process n data

            __mmask16 cmpgt1 = _mm512_cmpgt_epi32_mask(n_i_start_v, n_j_end_v);
            __mmask16 cmpgt2 = _mm512_cmpgt_epi32_mask(n_j_start_v, n_i_end_v);

            __mmask16 worth_intersection = _mm512_knot(_mm512_kor(cmpgt1, cmpgt2));
            ones_num = _popcnt32_redef(_cvtmask16_u32(worth_intersection));

            if (ones_num != 0) {
                __m512i j_vertices_tmp2 = _mm512_set1_epi32(j);
                __m512i j_vertices = _mm512_add_epi32(j_vertices_tmp1, j_vertices_tmp2);

                _mm512_mask_compressstoreu_epi32((stack16_j_vertex),
                                                 worth_intersection,
                                                 j_vertices);

                GRAPH_STACK_ALING(64) std::int32_t stack16_intersections[16] = { 0 };
                for (std::int32_t s = 0; s < ones_num; s++) {
                    const auto j_neighbor_size = g_degrees[stack16_j_vertex[s]];
                    const auto j_neigbhors =
                        g_vertex_neighbors + g_edge_offsets[stack16_j_vertex[s]];
                    stack16_intersections[s] =
                        intersection(i_neigbhors, j_neigbhors, i_neighbor_size, j_neighbor_size);
                }
                __m512i intersections_v = _mm512_load_epi32(stack16_intersections);
                j_vertices = _mm512_load_epi32(stack16_j_vertex);
                __mmask16 non_zero_coefficients =
                    _mm512_test_epi32_mask(intersections_v, intersections_v);
                _mm512_mask_compressstoreu_epi32((first_vertices + nnz),
                                                 non_zero_coefficients,
                                                 i_vertex);
                _mm512_mask_compressstoreu_epi32((second_vertices + nnz),
                                                 non_zero_coefficients,
                                                 j_vertices);
                __m512 tmp_v = _mm512_cvtepi32_ps(intersections_v);
                _mm512_mask_compressstoreu_ps((jaccard + nnz), non_zero_coefficients, tmp_v);

                nnz += _popcnt32_redef(_cvtmask16_u32(non_zero_coefficients));
            }

            j += 16;

            for (j = column_begin + ((diagonal - column_begin) / 16); j < diagonal; j++) {
                const auto j_neighbor_size = g_degrees[j];
                const auto j_neigbhors = g_vertex_neighbors + g_edge_offsets[j];
                if (!(i_neigbhors[0] > j_neigbhors[j_neighbor_size - 1]) &&
                    !(j_neigbhors[0] > i_neigbhors[i_neighbor_size - 1])) {
                    auto intersection_value =
                        intersection(i_neigbhors, j_neigbhors, i_neighbor_size, j_neighbor_size);
                    if (intersection_value) {
                        jaccard[nnz] = static_cast<float>(intersection_value);
                        first_vertices[nnz] = i;
                        second_vertices[nnz] = j;
                        nnz++;
                    }
                }
            }
        }
        else {
#endif
            for (j = column_begin; j < diagonal; j++) {
                const auto j_neighbor_size = g_degrees[j];
                const auto j_neigbhors = g_vertex_neighbors + g_edge_offsets[j];
                if (!(i_neigbhors[0] > j_neigbhors[j_neighbor_size - 1]) &&
                    !(j_neigbhors[0] > i_neigbhors[i_neighbor_size - 1])) {
                    auto intersection_value =
                        intersection(i_neigbhors, j_neigbhors, i_neighbor_size, j_neighbor_size);
                    if (intersection_value) {
                        jaccard[nnz] = static_cast<float>(intersection_value);
                        first_vertices[nnz] = i;
                        second_vertices[nnz] = j;
                        nnz++;
                    }
                }
            }
#if defined(__INTEL_COMPILER)
        }
#endif

        std::int32_t tmp_idx = column_begin;
        if (diagonal >= column_begin) {
            jaccard[nnz] = 1.0;
            first_vertices[nnz] = i;
            second_vertices[nnz] = diagonal;
            nnz++;
            tmp_idx = diagonal + 1;
        }
        j = tmp_idx;

#if defined(__INTEL_COMPILER)
        if (j < tmp_idx + ((column_end - tmp_idx) / 16) * 16) {
            //load_data(0)
            __m512i start_indices_j_v = _mm512_load_epi32(g_edge_offsets + j);
            __m512i end_indices_j_v_tmp = _mm512_load_epi32(g_edge_offsets + j + 1);
            __m512i end_indices_j_v = _mm512_add_epi32(end_indices_j_v_tmp, _mm512_set1_epi32(-1));

            __m512i n_j_start_v = _mm512_permutexvar_epi32(
                j_vertices_tmp1,
                _mm512_i32gather_epi32(start_indices_j_v, g_vertex_neighbors, 4));
            __m512i n_j_end_v = _mm512_permutexvar_epi32(
                j_vertices_tmp1,
                _mm512_i32gather_epi32(end_indices_j_v, g_vertex_neighbors, 4));

            for (; j + 16 < tmp_idx + ((column_end - tmp_idx) / 16) * 16;) {
                __m512i start_indices_j_v = _mm512_load_epi32(g_edge_offsets + j + 16);
                __m512i end_indices_j_v_tmp = _mm512_load_epi32(g_edge_offsets + j + 17);
                __m512i end_indices_j_v =
                    _mm512_add_epi32(end_indices_j_v_tmp, _mm512_set1_epi32(-1));

                __m512i n_j_start_v1 = _mm512_permutexvar_epi32(
                    j_vertices_tmp1,
                    _mm512_i32gather_epi32(start_indices_j_v, g_vertex_neighbors, 4));
                __m512i n_j_end_v1 = _mm512_permutexvar_epi32(
                    j_vertices_tmp1,
                    _mm512_i32gather_epi32(end_indices_j_v, g_vertex_neighbors, 4));

                __mmask16 cmpgt1 = _mm512_cmpgt_epi32_mask(n_i_start_v, n_j_end_v);
                __mmask16 cmpgt2 = _mm512_cmpgt_epi32_mask(n_j_start_v, n_i_end_v);

                __mmask16 worth_intersection = _mm512_knot(_mm512_kor(cmpgt1, cmpgt2));
                ones_num = _popcnt32_redef(_cvtmask16_u32(worth_intersection));

                if (ones_num != 0) {
                    __m512i j_vertices_tmp2 = _mm512_set1_epi32(j);
                    __m512i j_vertices = _mm512_add_epi32(j_vertices_tmp1, j_vertices_tmp2);

                    _mm512_mask_compressstoreu_epi32((stack16_j_vertex),
                                                     worth_intersection,
                                                     j_vertices);

                    GRAPH_STACK_ALING(64) std::int32_t stack16_intersections[16] = { 0 };
                    for (std::int32_t s = 0; s < ones_num; s++) {
                        const auto j_neighbor_size = g_degrees[stack16_j_vertex[s]];
                        const auto j_neigbhors =
                            g_vertex_neighbors + g_edge_offsets[stack16_j_vertex[s]];
                        stack16_intersections[s] = intersection(i_neigbhors,
                                                                j_neigbhors,
                                                                i_neighbor_size,
                                                                j_neighbor_size);
                    }
                    __m512i intersections_v = _mm512_load_epi32(stack16_intersections);
                    j_vertices = _mm512_load_epi32(stack16_j_vertex);

                    __mmask16 non_zero_coefficients =
                        _mm512_test_epi32_mask(intersections_v, intersections_v);
                    _mm512_mask_compressstoreu_epi32((first_vertices + nnz),
                                                     non_zero_coefficients,
                                                     i_vertex);
                    _mm512_mask_compressstoreu_epi32((second_vertices + nnz),
                                                     non_zero_coefficients,
                                                     j_vertices);
                    __m512 tmp_v = _mm512_cvtepi32_ps(intersections_v);
                    _mm512_mask_compressstoreu_ps((jaccard + nnz), non_zero_coefficients, tmp_v);

                    nnz += _popcnt32_redef(_cvtmask16_u32(non_zero_coefficients));
                }

                j += 16;

                n_j_start_v = n_j_start_v1;
                n_j_end_v = n_j_end_v1;
            }

            //process n data

            __mmask16 cmpgt1 = _mm512_cmpgt_epi32_mask(n_i_start_v, n_j_end_v);
            __mmask16 cmpgt2 = _mm512_cmpgt_epi32_mask(n_j_start_v, n_i_end_v);

            __mmask16 worth_intersection = _mm512_knot(_mm512_kor(cmpgt1, cmpgt2));
            ones_num = _popcnt32_redef(_cvtmask16_u32(worth_intersection));

            if (ones_num != 0) {
                __m512i j_vertices_tmp2 = _mm512_set1_epi32(j);
                __m512i j_vertices = _mm512_add_epi32(j_vertices_tmp1, j_vertices_tmp2);

                _mm512_mask_compressstoreu_epi32((stack16_j_vertex),
                                                 worth_intersection,
                                                 j_vertices);

                GRAPH_STACK_ALING(64) std::int32_t stack16_intersections[16] = { 0 };
                for (std::int32_t s = 0; s < ones_num; s++) {
                    const auto j_neighbor_size = g_degrees[stack16_j_vertex[s]];
                    const auto j_neigbhors =
                        g_vertex_neighbors + g_edge_offsets[stack16_j_vertex[s]];
                    stack16_intersections[s] =
                        intersection(i_neigbhors, j_neigbhors, i_neighbor_size, j_neighbor_size);
                }
                __m512i intersections_v = _mm512_load_epi32(stack16_intersections);
                j_vertices = _mm512_load_epi32(stack16_j_vertex);
                __mmask16 non_zero_coefficients =
                    _mm512_test_epi32_mask(intersections_v, intersections_v);
                _mm512_mask_compressstoreu_epi32((first_vertices + nnz),
                                                 non_zero_coefficients,
                                                 i_vertex);
                _mm512_mask_compressstoreu_epi32((second_vertices + nnz),
                                                 non_zero_coefficients,
                                                 j_vertices);
                __m512 tmp_v = _mm512_cvtepi32_ps(intersections_v);
                _mm512_mask_compressstoreu_ps((jaccard + nnz), non_zero_coefficients, tmp_v);

                nnz += _popcnt32_redef(_cvtmask16_u32(non_zero_coefficients));
            }

            j += 16;

            for (j = tmp_idx + ((column_end - tmp_idx) / 16) * 16; j < column_end; j++) {
                const auto j_neighbor_size = g_degrees[j];
                const auto j_neigbhors = g_vertex_neighbors + g_edge_offsets[j];
                if (!(i_neigbhors[0] > j_neigbhors[j_neighbor_size - 1]) &&
                    !(j_neigbhors[0] > i_neigbhors[i_neighbor_size - 1])) {
                    auto intersection_value =
                        intersection(i_neigbhors, j_neigbhors, i_neighbor_size, j_neighbor_size);
                    if (intersection_value) {
                        jaccard[nnz] = static_cast<float>(intersection_value);
                        first_vertices[nnz] = i;
                        second_vertices[nnz] = j;
                        nnz++;
                    }
                }
            }
        }
        else {
#endif
            for (j = tmp_idx; j < column_end; j++) {
                const auto j_neighbor_size = g_degrees[j];
                const auto j_neigbhors = g_vertex_neighbors + g_edge_offsets[j];
                if (!(i_neigbhors[0] > j_neigbhors[j_neighbor_size - 1]) &&
                    !(j_neigbhors[0] > i_neigbhors[i_neighbor_size - 1])) {
                    auto intersection_value =
                        intersection(i_neigbhors, j_neigbhors, i_neighbor_size, j_neighbor_size);
                    if (intersection_value) {
                        jaccard[nnz] = static_cast<float>(intersection_value);
                        first_vertices[nnz] = i;
                        second_vertices[nnz] = j;
                        nnz++;
                    }
                }
            }
#if defined(__INTEL_COMPILER)
        }
#endif
    }

    PRAGMA_VECTOR_ALWAYS
    for (int i = 0; i < nnz; i++) {
        if (first_vertices[i] != second_vertices[i])
            jaccard[i] =
                jaccard[i] / static_cast<float>(g_degrees[first_vertices[i]] +
                                                g_degrees[second_vertices[i]] - jaccard[i]);
    }

    vertex_similarity_result res(
        homogen_table::wrap(first_vertices, number_elements_in_block, 2, data_layout::column_major),
        homogen_table::wrap(jaccard, number_elements_in_block, 1, data_layout::column_major),
        nnz);
    return res;
}
} // namespace detail
} // namespace jaccard
} // namespace oneapi::dal::preview
