#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/si.hpp"

#include <xmmintrin.h>

namespace oneapi::dal::preview {
namespace subgraph_isomorphism {
namespace detail {

solution dal_experimental::subgraph_isomorphism(const dal_experimental::graph& pattern,
                                                const dal_experimental::graph& target,
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
#ifdef DEBUG_MODE
        auto t0 = Time::now();
#endif // DEBUG_MODE

        engine_bundle harness(&pattern,
                              &target,
                              sorted_pattern_vertex,
                              predecessor,
                              direction,
                              cconditions,
                              pattern_vertex_probability,
                              control_flags);
        sol = harness.run();

#ifdef DEBUG_MODE
        auto t1 = Time::now();
        fsec fs = t1 - t0;
        std::cout << "multi thread Matching time:" << fs.count() << "s\n";
#endif // DEBUG_MODE
    }
    else {
#ifdef DEBUG_MODE
        auto t0 = Time::now();
#endif // DEBUG_MODE

        matching_engine main_engine(&pattern,
                                    &target,
                                    sorted_pattern_vertex,
                                    predecessor,
                                    direction,
                                    cconditions);
        sol = main_engine.run(true);

#ifdef DEBUG_MODE
        auto t1 = Time::now();
        fsec fs = t1 - t0;
        // std::cout << "single thread Matching time:" << fs.count() << "s\n";
        //std::cout << "Total states handled: " << main_engine.engine_statistic.state_handling << std::endl;
#endif // DEBUG_MODE
    }

#ifdef DEBUG_MODE
    // std::cout << "Solutions: " << sol.get_solution_count() << std::endl;
#endif // DEBUG_MODE
    //-------------------------------------

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

} // namespace detail
} // namespace subgraph_isomorphism
} // namespace oneapi::dal::preview
} // namespace oneapi::dal::preview