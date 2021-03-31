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

#include "oneapi/dal/algo/subgraph_isomorphism/detail/graph_matching_default_kernel.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/detail/si.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/common.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/graph_matching_types.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/graph/detail/service_functions_impl.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/detail/matching.hpp"

using namespace oneapi::dal::preview::subgraph_isomorphism::detail;

namespace oneapi::dal::preview {
namespace subgraph_isomorphism {
namespace detail {

template <typename Index>
graph_matching_result call_subgraph_isomorphism_default_kernel_scalar(
    const descriptor_base &desc,
    const dal::preview::detail::topology<std::int32_t> &t_data,
    const dal::preview::detail::topology<std::int32_t> &p_data) {
    graph pattern(p_data, graph_storage_scheme::bit);
    graph target(t_data, graph_storage_scheme::auto_detect);

    std::uint64_t control_flags =
        flow_switch_ids::multi_thread_mode; // flow_switch_ids::default_single_thread_mode
    solution results =
        subgraph_isomorphism::detail::subgraph_isomorphism(pattern, target, control_flags);

    return graph_matching_result(results.export_as_table(), results.get_solution_count());
}
} // namespace detail
} // namespace subgraph_isomorphism
} // namespace oneapi::dal::preview
