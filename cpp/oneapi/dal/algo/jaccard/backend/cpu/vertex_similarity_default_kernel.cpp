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
#include "oneapi/dal/algo/jaccard/common.hpp"
#include "oneapi/dal/algo/jaccard/vertex_similarity_types.hpp"
#include "oneapi/dal/data/graph_service.hpp"

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

namespace oneapi::dal::preview {
namespace jaccard {
namespace detail {

template <class NodeID_t>
size_t intersection(NodeID_t *neigh_u, NodeID_t *neigh_v, NodeID_t n_u, NodeID_t n_v) {
    size_t total = 0;
    NodeID_t i_u = 0, i_v = 0;

#if defined AVX512
    while (i_u < (n_u / 16) * 16 && i_v < (n_v / 16) * 16) { // not in last n%16 elements
        __m512i v_u   = _mm512_loadu_si512((void *)(neigh_u + i_u)); // load 16 neighbors of u
        __m512i v_v   = _mm512_loadu_si512((void *)(neigh_v + i_v)); // load 16 neighbors of v
        NodeID_t maxu = neigh_u[i_u + 15]; // assumes neighbor list is ordered
        NodeID_t minu = neigh_u[i_u];
        NodeID_t maxv = neigh_v[i_v + 15];
        NodeID_t minv = neigh_v[i_v];
        if (minu > maxv) {
            if (minu > neigh_v[n_v - 1]) {
                return total;
            }
            i_v += 16;
            continue;
        }
        if (minv > maxu) {
            if (minv > neigh_u[n_u - 1]) {
                return total;
            }
            i_u += 16;
            continue;
        }
        if (maxu >= maxv)
            i_v += 16;
        if (maxu <= maxv)
            i_u += 16;
        __mmask16 match = _mm512_cmpeq_epi32_mask(v_u, v_v);
        if (_mm512_mask2int(match) != 0xffff) { // shortcut case where all neighbors match
            __m512i circ1  = _mm512_set_epi32(0,
                                             15,
                                             14,
                                             13,
                                             12,
                                             11,
                                             10,
                                             9,
                                             8,
                                             7,
                                             6,
                                             5,
                                             4,
                                             3,
                                             2,
                                             1); // all possible circular shifts for 16 elements
            __m512i circ2  = _mm512_set_epi32(1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2);
            __m512i circ3  = _mm512_set_epi32(2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3);
            __m512i circ4  = _mm512_set_epi32(3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4);
            __m512i circ5  = _mm512_set_epi32(4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5);
            __m512i circ6  = _mm512_set_epi32(5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6);
            __m512i circ7  = _mm512_set_epi32(6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7);
            __m512i circ8  = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);
            __m512i circ9  = _mm512_set_epi32(8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9);
            __m512i circ10 = _mm512_set_epi32(9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10);
            __m512i circ11 = _mm512_set_epi32(10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11);
            __m512i circ12 = _mm512_set_epi32(11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12);
            __m512i circ13 = _mm512_set_epi32(12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13);
            __m512i circ14 = _mm512_set_epi32(13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14);
            __m512i circ15 = _mm512_set_epi32(14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15);
            __m512i v_v1   = _mm512_permutexvar_epi32(circ1, v_v);
            __m512i v_v2   = _mm512_permutexvar_epi32(circ2, v_v);
            __m512i v_v3   = _mm512_permutexvar_epi32(circ3, v_v);
            __m512i v_v4   = _mm512_permutexvar_epi32(circ4, v_v);
            __m512i v_v5   = _mm512_permutexvar_epi32(circ5, v_v);
            __m512i v_v6   = _mm512_permutexvar_epi32(circ6, v_v);
            __m512i v_v7   = _mm512_permutexvar_epi32(circ7, v_v);
            __m512i v_v8   = _mm512_permutexvar_epi32(circ8, v_v);
            __m512i v_v9   = _mm512_permutexvar_epi32(circ9, v_v);
            __m512i v_v10  = _mm512_permutexvar_epi32(circ10, v_v);
            __m512i v_v11  = _mm512_permutexvar_epi32(circ11, v_v);
            __m512i v_v12  = _mm512_permutexvar_epi32(circ12, v_v);
            __m512i v_v13  = _mm512_permutexvar_epi32(circ13, v_v);
            __m512i v_v14  = _mm512_permutexvar_epi32(circ14, v_v);
            __m512i v_v15  = _mm512_permutexvar_epi32(circ15, v_v);
            __mmask16 tmp_match1  = _mm512_cmpeq_epi32_mask(v_u, v_v1); // find matches
            __mmask16 tmp_match2  = _mm512_cmpeq_epi32_mask(v_u, v_v2);
            __mmask16 tmp_match3  = _mm512_cmpeq_epi32_mask(v_u, v_v3);
            __mmask16 tmp_match4  = _mm512_cmpeq_epi32_mask(v_u, v_v4);
            __mmask16 tmp_match5  = _mm512_cmpeq_epi32_mask(v_u, v_v5);
            __mmask16 tmp_match6  = _mm512_cmpeq_epi32_mask(v_u, v_v6);
            __mmask16 tmp_match7  = _mm512_cmpeq_epi32_mask(v_u, v_v7);
            __mmask16 tmp_match8  = _mm512_cmpeq_epi32_mask(v_u, v_v8);
            __mmask16 tmp_match9  = _mm512_cmpeq_epi32_mask(v_u, v_v9);
            __mmask16 tmp_match10 = _mm512_cmpeq_epi32_mask(v_u, v_v10);
            __mmask16 tmp_match11 = _mm512_cmpeq_epi32_mask(v_u, v_v11);
            __mmask16 tmp_match12 = _mm512_cmpeq_epi32_mask(v_u, v_v12);
            __mmask16 tmp_match13 = _mm512_cmpeq_epi32_mask(v_u, v_v13);
            __mmask16 tmp_match14 = _mm512_cmpeq_epi32_mask(v_u, v_v14);
            __mmask16 tmp_match15 = _mm512_cmpeq_epi32_mask(v_u, v_v15);
            match                 = _mm512_kor(
                _mm512_kor(
                    _mm512_kor(_mm512_kor(match, tmp_match1), _mm512_kor(tmp_match2, tmp_match3)),
                    _mm512_kor(_mm512_kor(tmp_match4, tmp_match5),
                               _mm512_kor(tmp_match6, tmp_match7))),
                _mm512_kor(_mm512_kor(_mm512_kor(tmp_match8, tmp_match9),
                                      _mm512_kor(tmp_match10, tmp_match11)),
                           _mm512_kor(_mm512_kor(tmp_match12, tmp_match13),
                                      _mm512_kor(tmp_match14,
                                                 tmp_match15)))); // combine all matches
        }
        total += _popcnt32(_mm512_mask2int(match)); // count number of matches
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
        NodeID_t maxu = neigh_u[i_u + 7]; // assumes neighbor list is ordered
        NodeID_t minu = neigh_u[i_u];
        NodeID_t maxv = neigh_v[i_v + 7];
        NodeID_t minv = neigh_v[i_v];
        __m256i v_u   = _mm256_loadu_epi32((void *)(neigh_u + i_u)); // load 8 neighbors of u
        __m256i v_v   = _mm256_loadu_epi32((void *)(neigh_v + i_v)); // load 8 neighbors of v

        if (minu > maxv) {
            if (minu > neigh_v[n_v - 1]) {
                return total;
            }
            i_v += 8;
            continue;
        }
        if (minv > maxu) {
            if (minv > neigh_u[n_u - 1]) {
                return total;
            }
            i_u += 8;
            continue;
        }

        if (maxu >= maxv)
            i_v += 8;
        if (maxu <= maxv)
            i_u += 8;

        __mmask8 match = _mm256_cmpeq_epi32_mask(v_u, v_v);
        if (_cvtmask8_u32(match) != 0xff) { // shortcut case where all neighbors match
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
                           _kor_mask8(tmp_match6,
                                      tmp_match7))); // combine all matches
        }
        total += _popcnt32(_cvtmask8_u32(match)); // count number of matches
    }
    if (i_u <= (n_u - 8) && i_v < n_v) {
        __m256i v_u = _mm256_loadu_epi32((void *)(neigh_u + i_u));
        while (neigh_v[i_v] <= neigh_u[i_u + 7] && i_v < n_v) {
            __m256i tmp_v_v = _mm256_set1_epi32(neigh_v[i_v]);
            __mmask8 match  = _mm256_cmpeq_epi32_mask(v_u, tmp_v_v);
            if (_cvtmask8_u32(match))
                total++;
            i_v++;
        }
        i_u += 8;
    }
    if (i_v <= (n_v - 8) && i_u < n_u) {
        __m256i v_v = _mm256_loadu_epi32((void *)(neigh_v + i_v));
        while (neigh_u[i_u] <= neigh_v[i_v + 7] && i_u < n_u) {
            __m256i tmp_v_u = _mm256_set1_epi32(neigh_u[i_u]);
            __mmask8 match  = _mm256_cmpeq_epi32_mask(v_v, tmp_v_u);
            if (_cvtmask8_u32(match))
                total++;
            i_u++;
        }
        i_v += 8;
    }

    while (i_u <= (n_u - 4) && i_v <= (n_v - 4)) { // not in last n%8 elements

        NodeID_t maxu = neigh_u[i_u + 3]; // assumes neighbor list is ordered
        NodeID_t minu = neigh_u[i_u];
        NodeID_t maxv = neigh_v[i_v + 3];
        NodeID_t minv = neigh_v[i_v];
        __m128i v_u   = _mm_load_epi32((void *)(neigh_u + i_u)); // load 8 neighbors of u
        __m128i v_v   = _mm_load_epi32((void *)(neigh_v + i_v)); // load 8 neighbors of v

        if (minu > maxv) {
            if (minu > neigh_v[n_v - 1]) {
                return total;
            }
            i_v += 4;
            continue;
        }
        if (minv > maxu) {
            if (minv > neigh_u[n_u - 1]) {
                return total;
            }
            i_u += 4;
            continue;
        }

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
        total += _popcnt32(_cvtmask8_u32(match)); // count number of matches
    }
    if (i_u <= (n_u - 4) && i_v < n_v) {
        __m128i v_u = _mm_loadu_epi32((void *)(neigh_u + i_u));
        while (neigh_v[i_v] <= neigh_u[i_u + 3] && i_v < n_v) {
            __m128i tmp_v_v = _mm_set1_epi32(neigh_v[i_v]);
            __mmask8 match  = _mm_cmpeq_epi32_mask(v_u, tmp_v_v);
            if (_cvtmask8_u32(match))
                total++;
            i_v++;
        }
        i_u += 4;
    }
    if (i_v <= (n_v - 4) && i_u < n_u) {
        __m128i v_v = _mm_loadu_epi32((void *)(neigh_v + i_v));
        while (neigh_u[i_u] <= neigh_v[i_v + 3] && i_u < n_u) {
            __m128i tmp_v_u = _mm_set1_epi32(neigh_u[i_u]);
            __mmask8 match  = _mm_cmpeq_epi32_mask(v_v, tmp_v_u);
            if (_cvtmask8_u32(match))
                total++;
            i_u++;
        }
        i_v += 4;
    }

#endif

#if defined AVX2
    while (i_u <= (n_u - 8) && i_v <= (n_v - 8)) { // not in last n%8 elements
        // counter++;
        NodeID_t maxu = neigh_u[i_u + 7]; // assumes neighbor list is ordered
        NodeID_t minu = neigh_u[i_u];
        NodeID_t maxv = neigh_v[i_v + 7];
        NodeID_t minv = neigh_v[i_v];
        if (minu > maxv) {
            if (minu > neigh_v[n_v - 1]) {
                return total;
            }
            i_v += 8;
            continue;
        }
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
        //_mm256_cvtsi256_si32     _mm_loadu_si128

        //_mm256_extract_epi32
        if (maxu >= maxv)
            i_v += 8;
        if (maxu <= maxv)
            i_u += 8;

        __m256i match             = _mm256_cmpeq_epi32(v_u, v_v);
        unsigned int scalar_match = _mm256_movemask_ps(reinterpret_cast<__m256>(match));

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

            unsigned int scalar_match1 = _mm256_movemask_ps(reinterpret_cast<__m256>(tmp_match1));
            unsigned int scalar_match2 = _mm256_movemask_ps(reinterpret_cast<__m256>(tmp_match2));
            unsigned int scalar_match3 = _mm256_movemask_ps(reinterpret_cast<__m256>(tmp_match3));
            unsigned int scalar_match4 = _mm256_movemask_ps(reinterpret_cast<__m256>(tmp_match4));
            unsigned int scalar_match5 = _mm256_movemask_ps(reinterpret_cast<__m256>(tmp_match5));
            unsigned int scalar_match6 = _mm256_movemask_ps(reinterpret_cast<__m256>(tmp_match6));
            unsigned int scalar_match7 = _mm256_movemask_ps(reinterpret_cast<__m256>(tmp_match7));
            unsigned int final_match   = scalar_match | scalar_match1 | scalar_match2 |
                                       scalar_match3 | scalar_match4 | scalar_match5 |
                                       scalar_match6 | scalar_match7;

            while (final_match) {
                total += final_match & 1;
                final_match >>= 1;
            }
        }
        else {
            total += 8; // count number of matches
        }
    }

    while (i_u <= (n_u - 8) && i_v < n_v) {
        __m256i v_u = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(neigh_u + i_u));
        while (neigh_v[i_v] <= neigh_u[i_u + 7] && i_v < n_v) {
            __m256i tmp_v_v = _mm256_set1_epi32(neigh_v[i_v]);

            __m256i match             = _mm256_cmpeq_epi32(v_u, tmp_v_v);
            unsigned int scalar_match = _mm256_movemask_ps(reinterpret_cast<__m256>(match));
            if (scalar_match != 0)
                total++;
            i_v++;
        }
        i_u += 8;
    }
    while (i_v <= (n_v - 8) && i_u < n_u) {
        __m256i v_v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(neigh_v + i_v)); // load 8 neighbors of v
        while (neigh_u[i_u] <= neigh_v[i_v + 7] && i_u < n_u) {
            __m256i tmp_v_u           = _mm256_set1_epi32(neigh_u[i_u]);
            __m256i match             = _mm256_cmpeq_epi32(v_v, tmp_v_u);
            unsigned int scalar_match = _mm256_movemask_ps(reinterpret_cast<__m256>(match));
            if (scalar_match != 0)
                total++;
            i_u++;
        }
        i_v += 8;
    }

    while (i_u <= (n_u - 4) && i_v <= (n_v - 4)) { // not in last n%8 elements

        NodeID_t maxu = neigh_u[i_u + 3]; // assumes neighbor list is ordered
        NodeID_t minu = neigh_u[i_u];
        NodeID_t maxv = neigh_v[i_v + 3];
        NodeID_t minv = neigh_v[i_v];
        __m128i v_u   = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(neigh_u + i_u)); // load 8 neighbors of u
        __m128i v_v = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(neigh_v + i_v)); // load 8 neighbors of v

        if (minu > maxv) {
            if (minu > neigh_v[n_v - 1]) {
                return total;
            }
            i_v += 4;
            continue;
        }
        if (minv > maxu) {
            if (minv > neigh_u[n_u - 1]) {
                return total;
            }
            i_u += 4;
            continue;
        }
        if (maxu >= maxv) {
            i_v += 4;
        }
        if (maxu <= maxv) {
            i_u += 4;
        }

        __m128i match             = _mm_cmpeq_epi32(v_u, v_v);
        unsigned int scalar_match = _mm_movemask_ps(reinterpret_cast<__m128>(match));

        if (scalar_match != 155) { // shortcut case where all neighbors match
            __m128i v_v1 = _mm_shuffle_epi32(v_v, _MM_SHUFFLE(0, 3, 2, 1));
            __m128i v_v2 = _mm_shuffle_epi32(v_v, _MM_SHUFFLE(1, 0, 3, 2));
            __m128i v_v3 = _mm_shuffle_epi32(v_v, _MM_SHUFFLE(2, 1, 0, 3));

            __m128i tmp_match1 = _mm_cmpeq_epi32(v_u, v_v1); // find matches
            __m128i tmp_match2 = _mm_cmpeq_epi32(v_u, v_v2);
            __m128i tmp_match3 = _mm_cmpeq_epi32(v_u, v_v3);

            unsigned int scalar_match1 = _mm_movemask_ps(reinterpret_cast<__m128>(tmp_match1));
            unsigned int scalar_match2 = _mm_movemask_ps(reinterpret_cast<__m128>(tmp_match2));
            unsigned int scalar_match3 = _mm_movemask_ps(reinterpret_cast<__m128>(tmp_match3));

            unsigned int final_match = scalar_match | scalar_match1 | scalar_match2 | scalar_match3;

            while (final_match) {
                total += final_match & 1;
                final_match >>= 1;
            }
        }
        else {
            total += 4; // count number of matches
        }
    }
    if (i_u <= (n_u - 4) && i_v < n_v) {
        __m128i v_u = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(neigh_u + i_u)); // load 8 neighbors of u
        while (neigh_v[i_v] <= neigh_u[i_u + 3] && i_v < n_v) {
            __m128i tmp_v_v           = _mm_set1_epi32(neigh_v[i_v]);
            __m128i match             = _mm_cmpeq_epi32(v_u, tmp_v_v);
            unsigned int scalar_match = _mm_movemask_ps(reinterpret_cast<__m128>(match));

            if (scalar_match != 0)
                total++;
            i_v++;
        }
        i_u += 4;
    }
    if (i_v <= (n_v - 4) && i_u < n_u) {
        __m128i v_v = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(neigh_v + i_v)); // load 8 neighbors of v
        while (neigh_u[i_u] <= neigh_v[i_v + 3] && i_u < n_u) {
            __m128i tmp_v_u = _mm_set1_epi32(neigh_u[i_u]);

            __m128i match             = _mm_cmpeq_epi32(v_v, tmp_v_u);
            unsigned int scalar_match = _mm_movemask_ps(reinterpret_cast<__m128>(match));

            if (scalar_match != 0)
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

template size_t intersection<int32_t>(int32_t *neigh_u, int32_t *neigh_v, int32_t n_u, int32_t n_v);
template size_t intersection<int64_t>(int64_t *neigh_u, int64_t *neigh_v, int64_t n_u, int64_t n_v);
template size_t intersection<uint32_t>(uint32_t *neigh_u,
                                       uint32_t *neigh_v,
                                       uint32_t n_u,
                                       uint32_t n_v);
template size_t intersection<uint64_t>(uint64_t *neigh_u,
                                       uint64_t *neigh_v,
                                       uint64_t n_u,
                                       uint64_t n_v);

template <typename Graph>
similarity_result call_jaccard_block_kernel(const descriptor_base &desc,
                                            const similarity_input<Graph> &input) {
    std::cout << "Jaccard block kernel started" << std::endl;
    const auto my_graph = input.get_graph();
    std::cout << get_vertex_count(my_graph) << std::endl;
    std::cout << get_edge_count(my_graph) << std::endl;
    auto node_id = 0;
    std::cout << "degree of " << node_id << ": " << get_vertex_degree(my_graph, node_id)
              << std::endl;
    for (unsigned int j = 0; j < get_vertex_count(my_graph); ++j) {
        std::cout << "neighbors of " << j << ": ";
        auto neigh = get_vertex_neighbors(my_graph, j);
        for (auto i = neigh.first; i != neigh.second; ++i)
            std::cout << *i << " ";
        std::cout << std::endl;
    }
    const auto row_begin                = desc.get_row_begin();
    const auto row_end                  = desc.get_row_end();
    const auto column_begin             = desc.get_column_begin();
    const auto column_end               = desc.get_column_end();
    const auto number_elements_in_block = (row_end - row_begin) * (column_end - column_begin);
    array<float> jaccard                = array<float>::empty(number_elements_in_block);
    array<std::pair<std::uint32_t, std::uint32_t>> vertex_pairs =
        array<std::pair<std::uint32_t, std::uint32_t>>::empty(number_elements_in_block);
    size_t nnz = 0;
    for (auto i = row_begin; i < row_end; ++i) {
        const auto i_neighbor_size = get_vertex_degree(my_graph, i);
        const auto i_neigbhors     = get_vertex_neighbors(my_graph, i).first;
        for (auto j = column_begin; j < column_end; ++j) {
            if (j == i)
                continue;
            const auto j_neighbor_size = get_vertex_degree(my_graph, j);
            const auto j_neigbhors     = get_vertex_neighbors(my_graph, j).first;
            size_t intersection_value  = 0;
            size_t i_u = 0, i_v = 0;
            while (i_u < i_neighbor_size && i_v < j_neighbor_size) {
                if (i_neigbhors[i_u] == j_neigbhors[i_v])
                    intersection_value++, i_u++, i_v++;
                else if (i_neigbhors[i_u] < j_neigbhors[i_v])
                    i_u++;
                else if (i_neigbhors[i_u] > j_neigbhors[i_v])
                    i_v++;
            }
            if (intersection_value == 0)
                continue;
            const auto union_size = i_neighbor_size + j_neighbor_size - intersection_value;
            if (union_size == 0)
                continue;
            jaccard[nnz]      = float(intersection_value) / float(union_size);
            vertex_pairs[nnz] = std::make_pair(i, j);
            nnz++;
        }
    }
    jaccard.reset(nnz);
    vertex_pairs.reset(nnz);

    similarity_result res(homogen_table_builder{}.build(), homogen_table_builder{}.build());

    std::cout << "Jaccard block kernel ended" << std::endl;
    return res;
}

template similarity_result call_jaccard_block_kernel<undirected_adjacency_array<> &>(
    const descriptor_base &desc,
    const similarity_input<undirected_adjacency_array<> &> &input);

} // namespace detail
} // namespace jaccard
} // namespace oneapi::dal::preview
