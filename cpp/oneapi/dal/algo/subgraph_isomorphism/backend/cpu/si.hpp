#pragma once

#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/graph.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/solution.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/sorter.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/matching.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/common.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

solution si(const graph& pattern,
            const graph& target,
            kind isomorphism_kind,
            const std::uint64_t control_flags,
            byte_alloc_iface* alloc_ptr) {
    inner_alloc local_allocator(alloc_ptr);
    solution sol(local_allocator);
    sorter sorter_graph(&target, local_allocator);
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

    sorter_graph.~sorter();

    engine_bundle harness(&pattern,
                          &target,
                          sorted_pattern_vertex.get(),
                          predecessor.get(),
                          direction.get(),
                          cconditions.get(),
                          pattern_vertex_probability.get(),
                          control_flags,
                          isomorphism_kind,
                          local_allocator);
    sol = harness.run();

    for (std::int64_t i = 0; i < (pattern_vetrex_count - 1); i++) {
        cconditions_array[i].~sconsistent_conditions();
    }
    cconditions = nullptr;

    return sol;
}

} // namespace oneapi::dal::preview::subgraph_isomorphism::detail
