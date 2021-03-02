/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "oneapi/dal/algo/jaccard/common.hpp"
#include "oneapi/dal/algo/jaccard/vertex_similarity_types.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::preview::jaccard::detail {

inline std::int64_t get_number_elements_in_block(const std::int32_t &row_range_begin,
                                                 const std::int32_t &row_range_end,
                                                 const std::int32_t &column_range_begin,
                                                 const std::int32_t &column_range_end) {
    ONEDAL_ASSERT(row_range_end >= row_range_begin, "Negative interval found");
    const std::int64_t row_count = row_range_end - row_range_begin;
    ONEDAL_ASSERT(column_range_end >= column_range_begin, "Negative interval found");
    const std::int64_t column_count = column_range_end - column_range_begin;
    // compute the number of the vertex pairs in the block of the graph
    const std::int64_t vertex_pairs_count = row_count * column_count;
    ONEDAL_ASSERT(vertex_pairs_count / row_count == column_count,
                  "Overflow found in multiplication of two values");
    return vertex_pairs_count;
}

template <typename Float, typename Index>
inline std::int64_t get_max_block_size(const std::int64_t &vertex_pairs_count) {
    const std::int64_t vertex_pair_element_count = 2; // 2 elements in the vertex pair
    const std::int64_t jaccard_coeff_element_count = 1; // 1 Jaccard coeff for the vertex pair

    const std::int64_t vertex_pair_size =
        vertex_pair_element_count * sizeof(Index); // size in bytes
    const std::int64_t jaccard_coeff_size =
        jaccard_coeff_element_count * sizeof(Float); // size in bytes
    const std::int64_t element_result_size = vertex_pair_size + jaccard_coeff_size;

    const std::int64_t block_result_size = element_result_size * vertex_pairs_count;
    ONEDAL_ASSERT(block_result_size / vertex_pairs_count == element_result_size,
                  "Overflow found in multiplication of two values");
    return block_result_size;
}

template <typename Index>
inline Index min(const Index &a, const Index &b) {
    return (a >= b) ? b : a;
}

template <typename Index>
inline Index max(const Index &a, const Index &b) {
    return (a <= b) ? b : a;
}

template <typename Index>
inline std::int64_t intersection(const Index *neigh_u, const Index *neigh_v, Index n_u, Index n_v);

template <typename Index>
vertex_similarity_result call_jaccard_default_kernel_general(
    const descriptor_base &desc,
    const dal::preview::detail::topology<Index> &data,
    void *result_ptr) {
    const auto g_edge_offsets = data._rows_vertex.get_data();
    const auto g_vertex_neighbors = data._cols.get_data();
    const auto g_degrees = data._degrees.get_data();
    const auto row_begin = dal::detail::integral_cast<Index>(desc.get_row_range_begin());
    const auto row_end = dal::detail::integral_cast<Index>(desc.get_row_range_end());
    const auto column_begin = dal::detail::integral_cast<Index>(desc.get_column_range_begin());
    const auto column_end = dal::detail::integral_cast<Index>(desc.get_column_range_end());
    const auto number_elements_in_block =
        get_number_elements_in_block(row_begin, row_end, column_begin, column_end);
    Index *first_vertices = reinterpret_cast<Index *>(result_ptr);
    Index *second_vertices = first_vertices + number_elements_in_block;
    float *jaccard = reinterpret_cast<float *>(second_vertices + number_elements_in_block);
    std::int64_t nnz = 0;
    for (Index i = row_begin; i < row_end; ++i) {
        const auto i_neighbor_size = g_degrees[i];
        const auto i_neigbhors = g_vertex_neighbors + g_edge_offsets[i];
        const auto diagonal = min(i, column_end);
        for (Index j = column_begin; j < diagonal; j++) {
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
                    // Safe incrementing of nnz
                    //max nnz = (2^(31)  * 2^(31))=(2^62) < 2^63 = max size of std::int64_t
                    nnz++;
                    ONEDAL_ASSERT(nnz >= 0, "Overflow found in sum of two values");
                }
            }
        }

        if (diagonal >= column_begin && diagonal < column_end) {
            jaccard[nnz] = 1.0;
            first_vertices[nnz] = i;
            second_vertices[nnz] = diagonal;
            nnz++;
        }

        for (Index j = max(column_begin, diagonal + 1); j < column_end; j++) {
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
                    ONEDAL_ASSERT(nnz >= 0, "Overflow found in sum of two values");
                }
            }
        }
    }
    vertex_similarity_result res(
        homogen_table::wrap(first_vertices, number_elements_in_block, 2, data_layout::column_major),
        homogen_table::wrap(jaccard, number_elements_in_block, 1, data_layout::column_major),
        nnz);
    return res;
}

template <typename Index>
inline std::int64_t intersection(const Index *neigh_u, const Index *neigh_v, Index n_u, Index n_v) {
    std::int64_t total = 0;
    Index i_u = 0, i_v = 0;
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

} // namespace oneapi::dal::preview::jaccard::detail
