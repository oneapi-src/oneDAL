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

#include "oneapi/dal/algo/subgraph_isomorphism/common.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/graph_matching_types.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

template <typename Index>
graph_matching_result call_subgraph_isomorphism_default_kernel_general(
    const descriptor_base &desc,
    const dal::preview::detail::topology<std::int32_t> &t_data,
    const dal::preview::detail::topology<std::int32_t> &p_data,
    void *result_ptr) {
    std::cout << "KERNEL avx512" << std::endl;
    graph_matching_result res;
    return res;
}

} // namespace oneapi::dal::preview::subgraph_isomorphism::detail
