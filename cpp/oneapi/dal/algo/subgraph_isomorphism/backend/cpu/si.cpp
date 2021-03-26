#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/si.hpp"
#include "debug.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

solution subgraph_isomorphism(const graph& pattern,
                              const graph& target,
                              const std::uint64_t control_flags) {
    solution sol;
    sorter sorter_graph(&target);
    std::int64_t pattern_vetrex_count = pattern.get_vertex_count();
    float* pattern_vertex_probability =
        static_cast<float*>(_mm_malloc(sizeof(float) * pattern_vetrex_count, 64));

    sorter_graph.get_pattern_vertex_probability(pattern, pattern_vertex_probability);
    std::int64_t* sorted_pattern_vertex =
        static_cast<std::int64_t*>(_mm_malloc(sizeof(std::int64_t) * pattern_vetrex_count, 64));
    sorter_graph.sorting_pattern_vertices(pattern,
                                          pattern_vertex_probability,
                                          sorted_pattern_vertex);
    std::int64_t* predecessor =
        static_cast<std::int64_t*>(_mm_malloc(sizeof(std::int64_t) * pattern_vetrex_count, 64));
    edge_direction* direction =
        static_cast<edge_direction*>(_mm_malloc(sizeof(edge_direction) * pattern_vetrex_count, 64));

    sconsistent_conditions* cconditions = static_cast<sconsistent_conditions*>(
        _mm_malloc(sizeof(sconsistent_conditions) * (pattern_vetrex_count - 1), 64));
    for (std::int64_t i = 0; i < (pattern_vetrex_count - 1); i++) {
        cconditions[i].init(i + 1);
    }

    sorter_graph.create_sorted_pattern_tree(pattern,
                                            sorted_pattern_vertex,
                                            predecessor,
                                            direction,
                                            cconditions,
                                            true);

    std::int64_t* dfs_tree_search_width =
        static_cast<std::int64_t*>(_mm_malloc(sizeof(std::int64_t) * pattern_vetrex_count, 64));
    sorter_graph.dfs_tree_search_width_evaluation(pattern,
                                                  sorted_pattern_vertex,
                                                  pattern_vertex_probability,
                                                  direction,
                                                  cconditions,
                                                  dfs_tree_search_width);

    sorter_graph.~sorter();

    bool use_treading = control_flags & 0x1;

    if (use_treading) {
        engine_bundle harness(&pattern,
                              &target,
                              sorted_pattern_vertex,
                              predecessor,
                              direction,
                              cconditions,
                              pattern_vertex_probability,
                              control_flags);
        sol = harness.run();
    }
    else {
        matching_engine main_engine(&pattern,
                                    &target,
                                    sorted_pattern_vertex,
                                    predecessor,
                                    direction,
                                    cconditions);
        sol = main_engine.run(true);
    }

    _mm_free(pattern_vertex_probability);
    _mm_free(sorted_pattern_vertex);
    _mm_free(predecessor);
    _mm_free(direction);
    _mm_free(dfs_tree_search_width);

    for (std::int64_t i = 0; i < (pattern_vetrex_count - 1); i++) {
        cconditions[i].~sconsistent_conditions();
    }
    _mm_free(cconditions);
    cconditions = nullptr;

    return sol;
}

} // namespace oneapi::dal::preview::subgraph_isomorphism::backend