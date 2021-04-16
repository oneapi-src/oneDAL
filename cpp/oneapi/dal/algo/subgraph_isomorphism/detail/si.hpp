#pragma once

#include "oneapi/dal/algo/subgraph_isomorphism/detail/graph.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/detail/solution.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/detail/sorter.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/detail/matching.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/common.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

template <typename T>
std::shared_ptr<T> make_shared_malloc(std::uint64_t elements_count) {
    T* ptr = static_cast<T*>(_mm_malloc(sizeof(T) * elements_count, 64));
    return std::shared_ptr<T>(ptr, _mm_free);
}

solution si(const graph& pattern,
            const graph& target,
            kind isomorphism_kind,
            const std::uint64_t control_flags = 0) {
    solution sol;
    sorter sorter_graph(&target);
    std::int64_t pattern_vetrex_count = pattern.get_vertex_count();
    auto pattern_vertex_probability = make_shared_malloc<float>(pattern_vetrex_count);

    sorter_graph.get_pattern_vertex_probability(pattern, pattern_vertex_probability.get());
    auto sorted_pattern_vertex = make_shared_malloc<std::int64_t>(pattern_vetrex_count);
    sorter_graph.sorting_pattern_vertices(pattern,
                                          pattern_vertex_probability.get(),
                                          sorted_pattern_vertex.get());

    auto predecessor = make_shared_malloc<std::int64_t>(pattern_vetrex_count);
    auto direction = make_shared_malloc<edge_direction>(pattern_vetrex_count);
    auto cconditions = make_shared_malloc<sconsistent_conditions>(pattern_vetrex_count - 1);
    auto cconditions_array = cconditions.get();
    for (std::int64_t i = 0; i < (pattern_vetrex_count - 1); i++) {
        cconditions_array[i].init(i + 1); // should be placement new
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
                          isomorphism_kind);
    sol = harness.run();

    for (std::int64_t i = 0; i < (pattern_vetrex_count - 1); i++) {
        cconditions_array[i].~sconsistent_conditions();
    }
    cconditions = nullptr;

    return sol;
}

} // namespace oneapi::dal::preview::subgraph_isomorphism::detail
