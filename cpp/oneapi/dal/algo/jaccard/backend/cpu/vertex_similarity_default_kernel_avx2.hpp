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

#pragma once

#include <immintrin.h>

#include "oneapi/dal/algo/jaccard/backend/cpu/vertex_similarity_default_kernel.hpp"
#include "oneapi/dal/algo/jaccard/common.hpp"
#include "oneapi/dal/algo/jaccard/vertex_similarity_types.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/graph/detail/graph_service_functions_impl.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_array_graph_impl.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::preview {
namespace jaccard {
namespace detail {

#if defined(__INTEL_COMPILER)
DAAL_FORCEINLINE std::int32_t _popcnt32_redef(const std::int32_t &x) {
    return _popcnt32(x);
}
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
#endif

DAAL_FORCEINLINE std::size_t intersection(std::int32_t *neigh_u,
                                          std::int32_t *neigh_v,
                                          std::int32_t n_u,
                                          std::int32_t n_v) {
    std::size_t total = 0;
    std::int32_t i_u = 0, i_v = 0;

    const std::int32_t n_u_8_end = n_u - 8;
    const std::int32_t n_v_8_end = n_v - 8;
    while (i_u <= n_u_8_end && i_v <= n_v_8_end) {
        const std::int32_t minu = neigh_u[i_u];
        const std::int32_t maxv = neigh_v[i_v + 7];

        if (minu > maxv) {
            if (minu > neigh_v[n_v - 1]) {
                return total;
            }
            i_v += 8;
            continue;
        }

        const std::int32_t maxu = neigh_u[i_u + 7]; // assumes neighbor list is ordered
        const std::int32_t minv = neigh_v[i_v];

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

        i_v = (maxu >= maxv) ? i_v + 8 : i_v;
        i_u = (maxu <= maxv) ? i_u + 8 : i_u;

        __m256i match = _mm256_cmpeq_epi32(v_u, v_v);
        unsigned int scalar_match = _mm256_movemask_ps(_mm256_castsi256_ps(match));

        if (scalar_match != 255) { // shortcut case where all neighbors match
            __m256i circ1 = _mm256_set_epi32(0,
                                             7,
                                             6,
                                             5,
                                             4,
                                             3,
                                             2,
                                             1); // all possible circular shifts for 16 elements
            __m256i circ2 = _mm256_set_epi32(1, 0, 7, 6, 5, 4, 3, 2);
            __m256i circ3 = _mm256_set_epi32(2, 1, 0, 7, 6, 5, 4, 3);
            __m256i circ4 = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
            __m256i circ5 = _mm256_set_epi32(4, 3, 2, 1, 0, 7, 6, 5);
            __m256i circ6 = _mm256_set_epi32(5, 4, 3, 2, 1, 0, 7, 6);
            __m256i circ7 = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);

            __m256i v_v1 = _mm256_permutevar8x32_epi32(v_v, circ1);
            __m256i v_v2 = _mm256_permutevar8x32_epi32(v_v, circ2);
            __m256i v_v3 = _mm256_permutevar8x32_epi32(v_v, circ3);
            __m256i v_v4 = _mm256_permutevar8x32_epi32(v_v, circ4);
            __m256i v_v5 = _mm256_permutevar8x32_epi32(v_v, circ5);
            __m256i v_v6 = _mm256_permutevar8x32_epi32(v_v, circ6);
            __m256i v_v7 = _mm256_permutevar8x32_epi32(v_v, circ7);

            __m256i tmp_match1 = _mm256_cmpeq_epi32(v_u, v_v1); // find matches
            __m256i tmp_match2 = _mm256_cmpeq_epi32(v_u, v_v2);
            __m256i tmp_match3 = _mm256_cmpeq_epi32(v_u, v_v3);
            __m256i tmp_match4 = _mm256_cmpeq_epi32(v_u, v_v4);
            __m256i tmp_match5 = _mm256_cmpeq_epi32(v_u, v_v5);
            __m256i tmp_match6 = _mm256_cmpeq_epi32(v_u, v_v6);
            __m256i tmp_match7 = _mm256_cmpeq_epi32(v_u, v_v7);

            unsigned int scalar_match1 = _mm256_movemask_ps(_mm256_castsi256_ps(tmp_match1));
            unsigned int scalar_match2 = _mm256_movemask_ps(_mm256_castsi256_ps(tmp_match2));
            unsigned int scalar_match3 = _mm256_movemask_ps(_mm256_castsi256_ps(tmp_match3));
            unsigned int scalar_match4 = _mm256_movemask_ps(_mm256_castsi256_ps(tmp_match4));
            unsigned int scalar_match5 = _mm256_movemask_ps(_mm256_castsi256_ps(tmp_match5));
            unsigned int scalar_match6 = _mm256_movemask_ps(_mm256_castsi256_ps(tmp_match6));
            unsigned int scalar_match7 = _mm256_movemask_ps(_mm256_castsi256_ps(tmp_match7));
            unsigned int final_match = scalar_match | scalar_match1 | scalar_match2 |
                                       scalar_match3 | scalar_match4 | scalar_match5 |
                                       scalar_match6 | scalar_match7;

            total += _popcnt32_redef(final_match);
        }
        else {
            total += 8; //count number of matches
        }
    }

    for (; i_u <= n_u_8_end && i_v < n_v; i_u += 8) {
        __m256i v_u = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(neigh_u + i_u));

        const std::int32_t neighu_iu = neigh_u[i_u + 7];
        for (; neigh_v[i_v] <= neighu_iu && i_v < n_v; i_v++) {
            __m256i tmp_v_v = _mm256_set1_epi32(neigh_v[i_v]);

            __m256i match = _mm256_cmpeq_epi32(v_u, tmp_v_v);
            unsigned int scalar_match = _mm256_movemask_ps(_mm256_castsi256_ps(match));
            total = scalar_match != 0 ? total + 1 : total;
        }
    }
    for (; i_v <= n_v_8_end && i_u < n_u; i_v += 8) {
        __m256i v_v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(neigh_v + i_v)); // load 8 neighbors of v
        const std::int32_t neighv_iv = neigh_v[i_v + 7];
        for (; neigh_u[i_u] <= neighv_iv && i_u < n_u; i_u++) {
            __m256i tmp_v_u = _mm256_set1_epi32(neigh_u[i_u]);
            __m256i match = _mm256_cmpeq_epi32(v_v, tmp_v_u);
            unsigned int scalar_match = _mm256_movemask_ps(_mm256_castsi256_ps(match));
            total = scalar_match != 0 ? total + 1 : total;
        }
    }

    const std::int32_t n_u_4_end = n_u - 4;
    const std::int32_t n_v_4_end = n_v - 4;

    while (i_u <= n_u_4_end && i_v <= n_v_4_end) { // not in last n%8 elements

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

        i_v = (maxu >= maxv) ? i_v + 4 : i_v;
        i_u = (maxu <= maxv) ? i_u + 4 : i_u;

        __m128i match = _mm_cmpeq_epi32(v_u, v_v);
        unsigned int scalar_match = _mm_movemask_ps(_mm_castsi128_ps(match));

        if (scalar_match != 155) { // shortcut case where all neighbors match
            __m128i v_v1 = _mm_shuffle_epi32(v_v, _MM_SHUFFLE(0, 3, 2, 1));
            __m128i v_v2 = _mm_shuffle_epi32(v_v, _MM_SHUFFLE(1, 0, 3, 2));
            __m128i v_v3 = _mm_shuffle_epi32(v_v, _MM_SHUFFLE(2, 1, 0, 3));

            __m128i tmp_match1 = _mm_cmpeq_epi32(v_u, v_v1); // find matches
            __m128i tmp_match2 = _mm_cmpeq_epi32(v_u, v_v2);
            __m128i tmp_match3 = _mm_cmpeq_epi32(v_u, v_v3);

            unsigned int scalar_match1 = _mm_movemask_ps(_mm_castsi128_ps(tmp_match1));
            unsigned int scalar_match2 = _mm_movemask_ps(_mm_castsi128_ps(tmp_match2));
            unsigned int scalar_match3 = _mm_movemask_ps(_mm_castsi128_ps(tmp_match3));

            unsigned int final_match = scalar_match | scalar_match1 | scalar_match2 | scalar_match3;

            total += _popcnt32_redef(final_match);
        }
        else {
            total += 4; //count number of matches
        }
    }

    if (i_u <= n_u_4_end && i_v < n_v) {
        __m128i v_u = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(neigh_u + i_u)); // load 8 neighbors of u
        const std::int32_t neighu_iu = neigh_u[i_u + 3];
        for (; neigh_v[i_v] <= neighu_iu && i_v < n_v; i_v++) {
            __m128i tmp_v_v = _mm_set1_epi32(neigh_v[i_v]);
            __m128i match = _mm_cmpeq_epi32(v_u, tmp_v_v);
            unsigned int scalar_match = _mm_movemask_ps(_mm_castsi128_ps(match));
            total = scalar_match != 0 ? total + 1 : total;
        }
        i_u += 4;
    }
    if (i_v <= n_v_4_end && i_u < n_u) {
        __m128i v_v = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(neigh_v + i_v)); // load 8 neighbors of v
        const std::int32_t neighv_iv = neigh_v[i_v + 3];
        for (; neigh_u[i_u] <= neighv_iv && i_u < n_u; i_u++) {
            __m128i tmp_v_u = _mm_set1_epi32(neigh_u[i_u]);
            __m128i match = _mm_cmpeq_epi32(v_v, tmp_v_u);
            unsigned int scalar_match = _mm_movemask_ps(_mm_castsi128_ps(match));
            total = scalar_match != 0 ? total + 1 : total;
        }
        i_v += 4;
    }

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

template <typename Cpu>
vertex_similarity_result call_jaccard_default_kernel_avx2(
    const descriptor_base &desc,
    vertex_similarity_input<undirected_adjacency_array_graph<>> &input) {
    const auto &my_graph = input.get_graph();
    const auto &g = oneapi::dal::preview::detail::get_impl(my_graph);
    auto g_edge_offsets = g->_edge_offsets.data();
    auto g_vertex_neighbors = g->_vertex_neighbors.data();
    auto g_degrees = g->_degrees.data();
    const std::int32_t row_begin = static_cast<std::int32_t>(desc.get_row_range_begin());
    const auto row_end = static_cast<std::int32_t>(desc.get_row_range_end());
    const auto column_begin = static_cast<std::int32_t>(desc.get_column_range_begin());
    const auto column_end = static_cast<std::int32_t>(desc.get_column_range_end());
    const auto number_elements_in_block = (row_end - row_begin) * (column_end - column_begin);
    const size_t max_block_size =
        compute_max_block_size(row_begin, row_end, column_begin, column_end);
    void *result_ptr = input.get_caching_builder()(max_block_size);
    int *first_vertices = reinterpret_cast<int *>(result_ptr);
    int *second_vertices = first_vertices + number_elements_in_block;
    float *jaccard = reinterpret_cast<float *>(first_vertices + 2 * number_elements_in_block);
    std::int64_t nnz = 0;
    for (std::int32_t i = row_begin; i < row_end; ++i) {
        const auto i_neighbor_size = g_degrees[i];
        const auto i_neigbhors = g_vertex_neighbors + g_edge_offsets[i];
        const auto diagonal = min(i, column_end);
        for (std::int32_t j = column_begin; j < diagonal; j++) {
            const auto j_neighbor_size = g_degrees[j];
            const auto j_neigbhors = g_vertex_neighbors + g_edge_offsets[j];
            if (!(i_neigbhors[0] > j_neigbhors[j_neighbor_size - 1]) &&
                !(j_neigbhors[0] > i_neigbhors[i_neighbor_size - 1])) {
                auto intersection_value =
                    intersection(i_neigbhors, j_neigbhors, i_neighbor_size, j_neighbor_size);
                if (intersection_value) {
                    jaccard[nnz] = float(intersection_value) /
                                   float(i_neighbor_size + j_neighbor_size - intersection_value);
                    first_vertices[nnz] = i;
                    second_vertices[nnz] = j;
                    nnz++;
                }
            }
        }

        if (diagonal >= column_begin && diagonal < column_end) {
            jaccard[nnz] = 1.0;
            first_vertices[nnz] = i;
            second_vertices[nnz] = diagonal;
            nnz++;
        }

        for (std::int32_t j = max(column_begin, diagonal + 1); j < column_end; j++) {
            const auto j_neighbor_size = g_degrees[j];
            const auto j_neigbhors = g_vertex_neighbors + g_edge_offsets[j];
            if (!(i_neigbhors[0] > j_neigbhors[j_neighbor_size - 1]) &&
                !(j_neigbhors[0] > i_neigbhors[i_neighbor_size - 1])) {
                auto intersection_value =
                    intersection(i_neigbhors, j_neigbhors, i_neighbor_size, j_neighbor_size);
                if (intersection_value) {
                    jaccard[nnz] = float(intersection_value) /
                                   float(i_neighbor_size + j_neighbor_size - intersection_value);
                    first_vertices[nnz] = i;
                    second_vertices[nnz] = j;
                    nnz++;
                }
            }
        }
    }
    vertex_similarity_result res(homogen_table::wrap(first_vertices, 2, number_elements_in_block),
                                 homogen_table::wrap(jaccard, 1, number_elements_in_block),
                                 nnz);
    return res;
}
} // namespace detail
} // namespace jaccard
} // namespace oneapi::dal::preview
