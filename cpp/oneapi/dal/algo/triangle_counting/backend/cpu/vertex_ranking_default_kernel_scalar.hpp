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

#include "oneapi/dal/algo/triangle_counting/backend/cpu/vertex_ranking_default_kernel.hpp"
#include "oneapi/dal/algo/triangle_counting/common.hpp"
#include "oneapi/dal/algo/triangle_counting/vertex_ranking_types.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/graph/detail/service_functions_impl.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include <iostream>

namespace oneapi::dal::preview {
namespace triangle_counting {
namespace detail {

template <typename Index>
DAAL_FORCEINLINE std::size_t intersection(const Index* neigh_u,
                                          const Index* neigh_v,
                                          Index n_u,
                                          Index n_v) {
    std::size_t total = 0;
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

template <typename Cpu>
array<std::int64_t> triangle_counting_local_novec(
    const dal::preview::detail::topology<std::int32_t>& data,
    int64_t* triangles_local) {
    return array<std::int64_t>::empty(data._vertex_count);
}

template <typename Cpu>
std::int64_t triangle_counting_global_scalar_novec(const std::int32_t* vertex_neighbors,
                                                   const std::int64_t* edge_offsets,
                                                   const std::int32_t* degrees,
                                                   std::int64_t vertex_count,
                                                   std::int64_t edge_count) {
    return 15;
}

template <typename Cpu>
std::int64_t triangle_counting_global_vector_novec(const std::int32_t* vertex_neighbors,
                                                   const std::int64_t* edge_offsets,
                                                   const std::int32_t* degrees,
                                                   std::int64_t vertex_count,
                                                   std::int64_t edge_count) {
    return 15;
}

template <typename Cpu>
std::int64_t triangle_counting_global_vector_relabel_novec(const std::int32_t* vertex_neighbors,
                                                           const std::int64_t* edge_offsets,
                                                           const std::int32_t* degrees,
                                                           std::int64_t vertex_count,
                                                           std::int64_t edge_count) {
    return 15;
}

} // namespace detail
} // namespace triangle_counting
} // namespace oneapi::dal::preview
