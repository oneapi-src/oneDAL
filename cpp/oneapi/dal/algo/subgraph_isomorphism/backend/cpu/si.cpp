#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/si.hpp"
#include "debug.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

solution subgraph_isomorphism(const graph& pattern,
                              const graph& target,
                              const std::uint64_t control_flags) {
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

    auto dfs_tree_search_width = make_shared_malloc<std::int64_t>(pattern_vetrex_count);
    sorter_graph.dfs_tree_search_width_evaluation(pattern,
                                                  sorted_pattern_vertex.get(),
                                                  pattern_vertex_probability.get(),
                                                  direction.get(),
                                                  cconditions.get(),
                                                  dfs_tree_search_width.get());

    sorter_graph.~sorter();

    bool use_treading = control_flags & 0x1;

    if (use_treading) {
        engine_bundle harness(&pattern,
                              &target,
                              sorted_pattern_vertex.get(),
                              predecessor.get(),
                              direction.get(),
                              cconditions.get(),
                              pattern_vertex_probability.get(),
                              control_flags);
        sol = harness.run();
    }
    else {
        matching_engine main_engine(&pattern,
                                    &target,
                                    sorted_pattern_vertex.get(),
                                    predecessor.get(),
                                    direction.get(),
                                    cconditions.get());
        sol = main_engine.run(true);
    }

    for (std::int64_t i = 0; i < (pattern_vetrex_count - 1); i++) {
        cconditions_array[i].~sconsistent_conditions();
    }
    cconditions = nullptr;

    return sol;
}

} // namespace oneapi::dal::preview::subgraph_isomorphism::backend