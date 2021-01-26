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

#include "oneapi/dal/io/detail/load_graph.hpp"

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/graph/detail/container.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/graph/common.hpp"
#include "oneapi/dal/common.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/io/detail/load_graph_service.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/io/graph_csv_data_source.hpp"
#include "oneapi/dal/io/load_graph_descriptor.hpp"
#include "services/daal_atomic_int.h"
#include "services/daal_memory.h"

namespace oneapi::dal::preview::load_graph::detail {

template std::int64_t get_vertex_count_from_edge_list(const edge_list<std::int32_t> &edges);

template void collect_degrees_from_edge_list(
    const edge_list<std::int32_t> &edges,
    typename daal::services::Atomic<std::int32_t> *&degrees_cv);

template std::int32_t compute_prefix_sum_atomic(
    typename daal::services::Atomic<std::int32_t> *const &degrees,
    std::int64_t degrees_count,
    typename daal::services::Atomic<std::int32_t> *&edge_offsets_atomic);

template std::int32_t compute_prefix_sum(std::int32_t *const &degrees,
                                         std::int64_t degrees_count,
                                         std::int32_t *&edge_offsets);

template void fill_from_atomics(std::int32_t *&arr,
                                typename daal::services::Atomic<std::int32_t> *const &atomic_arr,
                                std::int64_t elements_count);

template void fill_unfiltered_neighs(
    const edge_list<std::int32_t> &edges,
    typename daal::services::Atomic<std::int32_t> *&rows_vec_atomic,
    std::int32_t *&unfiltered_neighs);

template void fill_filtered_neighs(const std::int32_t *unfiltered_offsets,
                                   const std::int32_t *unfiltered_neighs,
                                   const std::int32_t *filtered_degrees,
                                   const std::int32_t *filtered_offsets,
                                   std::int32_t *filtered_neighs,
                                   std::int64_t vertex_count);

template <>
void filter_neighbors_and_fill_new_degrees(std::int32_t *unfiltered_neighs,
                                           std::int32_t *&unfiltered_offsets,
                                           std::int32_t *&new_degrees,
                                           std::int64_t vertex_count) {
    //removing self-loops,  multiple edges from graph, and make neighbors in CSR sorted
    dal::detail::threader_for(vertex_count, vertex_count, [&](std::int32_t u) {
        auto start_p = unfiltered_neighs + unfiltered_offsets[u];
        auto end_p = unfiltered_neighs + unfiltered_offsets[u + 1];
        dal::detail::parallel_sort(start_p, end_p);
        auto neighs_u_new_end = std::unique(start_p, end_p);
        neighs_u_new_end = std::remove(start_p, neighs_u_new_end, u);
        new_degrees[u] = (std::int32_t)std::distance(start_p, neighs_u_new_end);
    });
}

} // namespace oneapi::dal::preview::load_graph::detail
