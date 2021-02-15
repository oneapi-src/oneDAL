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

#include "oneapi/dal/algo/triangle_counting/common.hpp"
#include "oneapi/dal/algo/triangle_counting/vertex_ranking_types.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include <iostream>

namespace oneapi::dal::preview::triangle_counting::detail {

template <typename Index>
inline std::int64_t intersection(const Index *neigh_u, const Index *neigh_v, Index n_u, Index n_v);

template <typename Index>
vertex_ranking_result<task::local> call_triangle_counting_default_kernel_general(
    const detail::descriptor_base<task::local> &desc,
    const dal::preview::detail::topology<Index> &data) {
    std::cout << "default kernel local tc" << std::endl;

    vertex_ranking_result<task::local> res;
    return res;
}

template <typename Index>
vertex_ranking_result<task::global> call_triangle_counting_default_kernel_general(
    const detail::descriptor_base<task::global> &desc,
    const dal::preview::detail::topology<Index> &data) {
    std::cout << "default kernel global tc" << std::endl;

    vertex_ranking_result<task::global> res;
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

} // namespace oneapi::dal::preview::triangle_counting::detail
