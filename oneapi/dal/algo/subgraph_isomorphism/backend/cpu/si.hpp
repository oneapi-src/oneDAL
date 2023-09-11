/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/graph.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/solution.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/sorter.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/matching.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/common.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/graph_matching_types.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

template <typename Cpu>
oneapi::dal::homogen_table si(const graph<Cpu>& pattern,
                              const graph<Cpu>& target,
                              kind isomorphism_kind,
                              std::int64_t max_match_count,
                              byte_alloc_iface_t* alloc_ptr) {
    inner_alloc local_allocator(alloc_ptr);

    sorter<Cpu> sorter_graph(&target, local_allocator);
    std::int64_t pattern_vetrex_count = pattern.get_vertex_count();
    auto pattern_vertex_probability =
        local_allocator.make_shared_memory<float>(pattern_vetrex_count);
    auto sorted_pattern_vertex =
        local_allocator.make_shared_memory<std::int64_t>(pattern_vetrex_count);
    std::int64_t* sorted_pattern_vertex_array = sorted_pattern_vertex.get();

    sorter_graph.get_pattern_vertex_probability(pattern, pattern_vertex_probability.get());
    sorter_graph.sorting_pattern_vertices(pattern,
                                          pattern_vertex_probability.get(),
                                          sorted_pattern_vertex_array);

    auto predecessor = local_allocator.make_shared_memory<std::int64_t>(pattern_vetrex_count);
    auto direction = local_allocator.make_shared_memory<edge_direction>(pattern_vetrex_count);
    auto cconditions =
        local_allocator.make_shared_memory<sconsistent_conditions<Cpu>>(pattern_vetrex_count - 1);
    sconsistent_conditions<Cpu>* const cconditions_array = cconditions.get();
    for (std::int64_t i = 0; i < (pattern_vetrex_count - 1); i++) {
        new (cconditions_array + i) sconsistent_conditions<Cpu>(i + 1, local_allocator);
    }

    sorter_graph.create_sorted_pattern_tree(pattern,
                                            sorted_pattern_vertex_array,
                                            predecessor.get(),
                                            direction.get(),
                                            cconditions.get(),
                                            true);

    engine_bundle<Cpu> harness(&pattern,
                               &target,
                               sorted_pattern_vertex_array,
                               predecessor.get(),
                               direction.get(),
                               cconditions.get(),
                               pattern_vertex_probability.get(),
                               isomorphism_kind,
                               local_allocator);
    const solution<Cpu> results = harness.run(max_match_count);

    for (std::int64_t i = 0; i < (pattern_vetrex_count - 1); i++) {
        cconditions_array[i].~sconsistent_conditions();
    }
    cconditions = nullptr;

    return results.export_as_table(sorted_pattern_vertex_array, max_match_count);
}

template <typename Cpu>
subgraph_isomorphism::graph_matching_result<task::compute> si_call_kernel(
    const kind& si_kind,
    std::int64_t max_match_count,
    byte_alloc_iface_t* alloc_ptr,
    const dal::preview::detail::topology<std::int32_t>& t_data,
    const dal::preview::detail::topology<std::int32_t>& p_data,
    std::int64_t* vv_t,
    std::int64_t* vv_p) {
    graph<Cpu> pattern(p_data, graph_storage_scheme::bit, alloc_ptr);
    graph<Cpu> target(t_data, graph_storage_scheme::auto_detect, alloc_ptr);

    if (vv_t != nullptr) {
        target.set_vertex_attribute(t_data._vertex_count, vv_t);
    }
    if (vv_p != nullptr) {
        pattern.set_vertex_attribute(p_data._vertex_count, vv_p);
    }

    const oneapi::dal::homogen_table results =
        si<Cpu>(pattern, target, si_kind, max_match_count, alloc_ptr);

    const auto solution_count = results.get_row_count();
    return graph_matching_result<task::compute>().set_vertex_match(results).set_match_count(
        (max_match_count == 0) ? solution_count : std::min(solution_count, max_match_count));
}

} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
