#pragma once

#include "oneapi/dal/algo/subgraph_isomorphism/detail/graph.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

struct sconsistent_conditions {
    std::int64_t* array;
    std::int64_t divider;
    std::int64_t length;
    void init(std::int64_t size) {
        length = size;
        array = allocator_.allocate<std::int64_t>(length);
        divider = length;
    }
    sconsistent_conditions(inner_alloc allocator)
            : array(nullptr),
              length(0),
              divider(length),
              allocator_(allocator) {}
    sconsistent_conditions(std::int64_t size, inner_alloc allocator) : allocator_(allocator) {
        init(size);
    }
    ~sconsistent_conditions() {
        if (array != nullptr) {
            allocator_.deallocate<std::int64_t>(array, length);
            array = nullptr;
        }
    }

private:
    inner_alloc allocator_;
};

class sorter {
public:
    sorter(inner_alloc allocator);
    sorter(const graph* ptarget, inner_alloc allocator);
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

private:
    inner_alloc allocator_;
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
} // namespace oneapi::dal::preview::subgraph_isomorphism::detail
