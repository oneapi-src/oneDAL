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

#include <memory>

#include "oneapi/dal/algo/jaccard/common.hpp"
#include "oneapi/dal/algo/jaccard/vertex_similarity_types.hpp"
#include "oneapi/dal/algo/jaccard/detail/service.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/primitives/intersection/intersection.hpp"
#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/table/homogen.hpp"

namespace oneapi::dal::preview::jaccard::backend {

template <typename Cpu>
vertex_similarity_result<task::all_vertex_pairs> jaccard(
    const detail::descriptor_base<task::all_vertex_pairs> &desc,
    const dal::preview::detail::topology<std::int32_t> &t,
    void *result_ptr) {
    const auto row_begin = dal::detail::integral_cast<std::int32_t>(desc.get_row_range_begin());
    const auto row_end = dal::detail::integral_cast<std::int32_t>(desc.get_row_range_end());
    const auto column_begin =
        dal::detail::integral_cast<std::int32_t>(desc.get_column_range_begin());
    const auto column_end = dal::detail::integral_cast<std::int32_t>(desc.get_column_range_end());
    const auto number_elements_in_block =
        detail::compute_number_elements_in_block(row_begin, row_end, column_begin, column_end);
    int *first_vertices = reinterpret_cast<int *>(result_ptr);
    int *second_vertices = first_vertices + number_elements_in_block;
    float *jaccard = reinterpret_cast<float *>(second_vertices + number_elements_in_block);
    std::int64_t nnz = 0;
    for (std::int32_t i = row_begin; i < row_end; ++i) {
        const std::int32_t i_neighbor_size = t.get_vertex_degree(i);
        const auto i_neigbhors = t.get_vertex_neighbors_begin(i);
        const auto diagonal = detail::min(i, column_end);
        for (std::int32_t j = column_begin; j < diagonal; j++) {
            const std::int32_t j_neighbor_size = t.get_vertex_degree(j);
            const auto j_neigbhors = t.get_vertex_neighbors_begin(j);
            if (!(i_neigbhors[0] > j_neigbhors[j_neighbor_size - 1]) &&
                !(j_neigbhors[0] > i_neigbhors[i_neighbor_size - 1])) {
                auto intersection_value =
                    preview::backend::intersection<Cpu>(t.get_vertex_neighbors_begin(i),
                                                        t.get_vertex_neighbors_begin(j),
                                                        i_neighbor_size,
                                                        j_neighbor_size);
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

        if (diagonal >= column_begin && diagonal < column_end) {
            jaccard[nnz] = 1.0;
            first_vertices[nnz] = i;
            second_vertices[nnz] = diagonal;
            nnz++;
            ONEDAL_ASSERT(nnz >= 0, "Overflow found in sum of two values");
        }

        for (std::int32_t j = detail::max(column_begin, diagonal + 1); j < column_end; j++) {
            const std::int32_t j_neighbor_size = t.get_vertex_degree(j);
            const auto j_neigbhors = t.get_vertex_neighbors_begin(j);
            if (!(i_neigbhors[0] > j_neigbhors[j_neighbor_size - 1]) &&
                !(j_neigbhors[0] > i_neigbhors[i_neighbor_size - 1])) {
                auto intersection_value =
                    preview::backend::intersection<Cpu>(t.get_vertex_neighbors_begin(i),
                                                        t.get_vertex_neighbors_begin(j),
                                                        i_neighbor_size,
                                                        j_neighbor_size);
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

#ifdef __ARM_ARCH
template <>
vertex_similarity_result<task::all_vertex_pairs> jaccard<dal::backend::cpu_dispatch_sve>(
    const detail::descriptor_base<task::all_vertex_pairs> &desc,
    const dal::preview::detail::topology<std::int32_t> &t,
    void *result_ptr);
#else
template <>
vertex_similarity_result<task::all_vertex_pairs> jaccard<dal::backend::cpu_dispatch_avx512>(
    const detail::descriptor_base<task::all_vertex_pairs> &desc,
    const dal::preview::detail::topology<std::int32_t> &t,
    void *result_ptr);

#endif
} // namespace oneapi::dal::preview::jaccard::backend
