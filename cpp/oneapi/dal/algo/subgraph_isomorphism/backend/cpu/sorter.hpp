#pragma once

#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/graph.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

struct sconsistent_conditions {
    std::int64_t* array;
    std::int64_t divider;
    std::int64_t length;
    void init(std::int64_t size) {
        length = size;
        array = static_cast<std::int64_t*>(_mm_malloc(sizeof(std::int64_t) * length, 64));
        divider = length;
    }
    sconsistent_conditions() : array(nullptr), length(0), divider(length) {}
    sconsistent_conditions(std::int64_t size) {
        init(size);
    }
    ~sconsistent_conditions() {
        if (array != nullptr) {
            _mm_free(array);
            array = nullptr;
        }
    }
};

class sorter {
public:
    sorter();
    sorter(const graph* ptarget);
    virtual ~sorter();

    graph_status get_pattern_vertex_probability(const graph& pattern,
                                                float* pattern_vertex_probability) const;
    graph_status sorting_pattern_vertices(const graph& pattern,
                                          const float* pattern_vertex_probability,
                                          std::int64_t* sorted_pattern_vertex) const;
    graph_status create_sorted_pattern_tree(const graph& pattern,
                                            const std::int64_t* sorted_pattern_vertex,
                                            std::int64_t* predecessor,
                                            edge_direction* direction,
                                            sconsistent_conditions* cconditions,
                                            bool predecessor_in_core_indexing = false) const;
    graph_status dfs_tree_search_width_evaluation(const graph& pattern,
                                                  const std::int64_t* sorted_pattern_vertex,
                                                  const float* pattern_vertex_probability,
                                                  const edge_direction* direction,
                                                  const sconsistent_conditions* cconditions,
                                                  std::int64_t* tree_search_width) const;

private:
    const graph* target;
    float* p_degree_probability;
    float* p_vertex_attribute_probability;

    std::int64_t degree_max_size;
    std::int64_t vertex_attribute_max_size;

    std::int64_t find_minimum_probability_index_by_mask(
        const graph& pattern,
        const float* pattern_vertex_probability,
        const std::uint8_t* pbit_mask = nullptr,
        const std::uint8_t* pbit_core_mask = nullptr) const;
    std::int64_t get_core_linked_degree(const graph& pattern,
                                        const std::int64_t vertex,
                                        const std::uint8_t* pbit_mask) const;
};
} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
