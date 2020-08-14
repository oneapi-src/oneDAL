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

#include "oneapi/dal/algo/jaccard/backend/cpu/vertex_similarity_default_kernel.hpp"
#include "oneapi/dal/algo/jaccard/common.hpp"
#include "oneapi/dal/algo/jaccard/vertex_similarity_types.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/data/detail/graph_service_functions_impl.hpp"
#include "oneapi/dal/detail/policy.hpp"

namespace oneapi::dal::preview {
namespace jaccard {
namespace detail {

template <class VertexType> //__declspec(noinline)
size_t intersection(VertexType *neigh_u, VertexType *neigh_v, VertexType n_u, VertexType n_v) {
    size_t total   = 0;
    VertexType i_u = 0, i_v = 0;
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
template size_t intersection<uint32_t>(uint32_t *neigh_u,
                                       uint32_t *neigh_v,
                                       uint32_t n_u,
                                       uint32_t n_v);

DAAL_FORCEINLINE int64_t min(int64_t a, int64_t b) {
    if (a >= b) {
        return b;
    }
    else {
        return a;
    }
}

template <typename Graph, typename Cpu>
vertex_similarity_result call_jaccard_default_kernel(const descriptor_base &desc,
                                                     vertex_similarity_input<Graph> &input) {
    auto my_graph                       = input.get_graph();
    auto g                              = oneapi::dal::preview::detail::get_impl(my_graph);
    auto g_edge_offsets                 = g->_edge_offsets.data();
    auto g_vertex_neighbors             = g->_vertex_neighbors.data();
    auto g_degrees                      = g->_degrees.data();
    const int32_t row_begin             = static_cast<int32_t>(desc.get_row_range_begin());
    const auto row_end                  = static_cast<int32_t>(desc.get_row_range_end());
    const auto column_begin             = static_cast<int32_t>(desc.get_column_range_begin());
    const auto column_end               = static_cast<int32_t>(desc.get_column_range_end());
    const auto number_elements_in_block = (row_end - row_begin) * (column_end - column_begin);
    int *first_vertices                 = reinterpret_cast<int *>(input.get_result_ptr());
    int *second_vertices                = first_vertices + number_elements_in_block;
    float *jaccard = reinterpret_cast<float *>(first_vertices + 2 * number_elements_in_block);
    int64_t nnz    = 0;
    for (int32_t i = row_begin; i < row_end; ++i) {
        const auto i_neighbor_size = g_degrees[i];
        const auto i_neigbhors     = g_vertex_neighbors + g_edge_offsets[i];
        const auto diagonal        = min(i, column_end);
        for (int32_t j = column_begin; j < diagonal; j++) {
            const auto j_neighbor_size = g_degrees[j];
            const auto j_neigbhors     = g_vertex_neighbors + g_edge_offsets[j];
            if (!(i_neigbhors[0] > j_neigbhors[j_neighbor_size - 1]) &&
                !(j_neigbhors[0] > i_neigbhors[i_neighbor_size - 1])) {
                auto intersection_value =
                    intersection(i_neigbhors, j_neigbhors, i_neighbor_size, j_neighbor_size);
                if (intersection_value) {
                    jaccard[nnz] = float(intersection_value) /
                                   float(i_neighbor_size + j_neighbor_size - intersection_value);
                    first_vertices[nnz]  = i;
                    second_vertices[nnz] = j;
                    nnz++;
                }
            }
        }

        auto tmp_idx = column_begin;
        if (diagonal >= column_begin) {
            jaccard[nnz]         = 1.0;
            first_vertices[nnz]  = i;
            second_vertices[nnz] = diagonal;
            nnz++;
            tmp_idx = diagonal + 1;
        }

        for (int32_t j = tmp_idx; j < column_end; j++) {
            const auto j_neighbor_size = g_degrees[j];
            const auto j_neigbhors     = g_vertex_neighbors + g_edge_offsets[j];
            if (!(i_neigbhors[0] > j_neigbhors[j_neighbor_size - 1]) &&
                !(j_neigbhors[0] > i_neigbhors[i_neighbor_size - 1])) {
                auto intersection_value =
                    intersection(i_neigbhors, j_neigbhors, i_neighbor_size, j_neighbor_size);
                if (intersection_value) {
                    jaccard[nnz] = float(intersection_value) /
                                   float(i_neighbor_size + j_neighbor_size - intersection_value);
                    first_vertices[nnz]  = i;
                    second_vertices[nnz] = j;
                    nnz++;
                }
            }
        }
    }
    vertex_similarity_result res(
        homogen_table_builder{}
            .reset(array(first_vertices, 2 * number_elements_in_block, empty_delete<const int>()),
                   2,
                   number_elements_in_block)
            .build(),
        homogen_table_builder{}
            .reset(array(jaccard, number_elements_in_block, empty_delete<const float>()),
                   1,
                   number_elements_in_block)
            .build(),
        nnz);
    return res;
}

#define INSTANTIATE(cpu)                                                  \
    template vertex_similarity_result                                     \
    call_jaccard_default_kernel<undirected_adjacency_array_graph<>, cpu>( \
        const descriptor_base &desc,                                      \
        vertex_similarity_input<undirected_adjacency_array_graph<>> &input);

INSTANTIATE(__CPU_TAG__)
} // namespace detail
} // namespace jaccard
} // namespace oneapi::dal::preview
