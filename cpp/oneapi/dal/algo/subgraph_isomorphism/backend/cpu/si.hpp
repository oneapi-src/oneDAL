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
using namespace oneapi::dal::preview::subgraph_isomorphism::backend;

template <typename Cpu>
solution si(const graph<Cpu>& pattern,
            const graph<Cpu>& target,
            kind isomorphism_kind,
            detail::byte_alloc_iface* alloc_ptr) {
    inner_alloc local_allocator(alloc_ptr);
    solution sol(local_allocator);
    sorter<Cpu> sorter_graph(&target, local_allocator);
    std::int64_t pattern_vetrex_count = pattern.get_vertex_count();
    auto pattern_vertex_probability =
        local_allocator.make_shared_memory<float>(pattern_vetrex_count);

    sorter_graph.get_pattern_vertex_probability(pattern, pattern_vertex_probability.get());
    auto sorted_pattern_vertex =
        local_allocator.make_shared_memory<std::int64_t>(pattern_vetrex_count);
    sorter_graph.sorting_pattern_vertices(pattern,
                                          pattern_vertex_probability.get(),
                                          sorted_pattern_vertex.get());

    auto predecessor = local_allocator.make_shared_memory<std::int64_t>(pattern_vetrex_count);
    auto direction = local_allocator.make_shared_memory<edge_direction>(pattern_vetrex_count);
    auto cconditions =
        local_allocator.make_shared_memory<sconsistent_conditions>(pattern_vetrex_count - 1);
    auto cconditions_array = cconditions.get();
    for (std::int64_t i = 0; i < (pattern_vetrex_count - 1); i++) {
        new (cconditions_array + i)
            sconsistent_conditions(i + 1, local_allocator); // should be placement new
    }

    sorter_graph.create_sorted_pattern_tree(pattern,
                                            sorted_pattern_vertex.get(),
                                            predecessor.get(),
                                            direction.get(),
                                            cconditions.get(),
                                            true);

    engine_bundle<Cpu> harness(&pattern,
                               &target,
                               sorted_pattern_vertex.get(),
                               predecessor.get(),
                               direction.get(),
                               cconditions.get(),
                               pattern_vertex_probability.get(),
                               isomorphism_kind,
                               local_allocator);
    sol = harness.run();

    for (std::int64_t i = 0; i < (pattern_vetrex_count - 1); i++) {
        cconditions_array[i].~sconsistent_conditions();
    }
    cconditions = nullptr;

    return sol;
}

template <typename Cpu>
subgraph_isomorphism::graph_matching_result si_call_kernel(
    const kind& si_kind,
    detail::byte_alloc_iface* alloc_ptr,
    const dal::preview::detail::topology<std::int32_t>& t_data,
    const dal::preview::detail::topology<std::int32_t>& p_data,
    const std::int64_t* vv_t,
    const std::int64_t* vv_p) {
    graph<Cpu> pattern(p_data, graph_storage_scheme::bit, alloc_ptr);
    graph<Cpu> target(t_data, graph_storage_scheme::auto_detect, alloc_ptr);

    const auto t_vertex_count = t_data._vertex_count;
    const auto p_vertex_count = p_data._vertex_count;

    if (vv_t != nullptr) {
        target.load_vertex_attribute(t_vertex_count, vv_t);
    }
    if (vv_p != nullptr) {
        pattern.load_vertex_attribute(p_vertex_count, vv_p);
    }

    solution results = si<Cpu>(pattern, target, si_kind, alloc_ptr);

    return graph_matching_result(results.export_as_table(), results.get_solution_count());
}

} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
