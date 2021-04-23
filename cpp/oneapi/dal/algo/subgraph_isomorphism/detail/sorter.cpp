
#include "oneapi/dal/algo/subgraph_isomorphism/detail/sorter.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/detail/debug.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

sorter::sorter(inner_alloc allocator) : allocator_(allocator) {
    p_degree_probability = nullptr;
    p_vertex_attribute_probability = nullptr;
    degree_max_size = 0;
    vertex_attribute_max_size = 0;
    target = nullptr;
}

sorter::~sorter() {
    target = nullptr;

    if (p_degree_probability != nullptr) {
        allocator_.deallocate<float>(p_degree_probability, degree_max_size);
        p_degree_probability = nullptr;
    }

    if (p_vertex_attribute_probability != nullptr) {
        allocator_.deallocate<float>(p_vertex_attribute_probability, vertex_attribute_max_size);
        p_vertex_attribute_probability = nullptr;
    }
}

sorter::sorter(const graph* ptarget, inner_alloc allocator) : sorter(allocator) {
    target = ptarget;
    degree_max_size = target->get_max_degree() + 1;
    vertex_attribute_max_size = target->get_max_vertex_attribute() + 1;

    std::int64_t vertex_count = target->get_vertex_count();

    p_degree_probability = allocator_.allocate<float>(degree_max_size);
    if (p_degree_probability == nullptr) {
        return;
    }
    p_vertex_attribute_probability = allocator_.allocate<float>(vertex_attribute_max_size);
    if (p_vertex_attribute_probability == nullptr) {
        return;
    }

    for (std::int64_t i = 0; i < degree_max_size; i++) {
        p_degree_probability[i] = 0.0;
    }

    for (std::int64_t i = 0; i < vertex_attribute_max_size; i++) {
        p_vertex_attribute_probability[i] = 0.0;
    }

    if (vertex_attribute_max_size == 1) {
        p_vertex_attribute_probability[0] = 1;
    }

    float delta_probability = 1.0;
    if (vertex_count > 0) {
        delta_probability /= static_cast<float>(vertex_count);
    }

    // possible parallelization
    for (std::int64_t i = 0; i < vertex_count; i++) {
        p_degree_probability[target->get_vertex_degree(i)] += delta_probability;

        if (vertex_attribute_max_size > 1) {
            p_vertex_attribute_probability[target->get_vertex_attribute(i)] += delta_probability;
        }
    }
}

graph_status sorter::get_pattern_vertex_probability(const graph& pattern,
                                                    float* pattern_vertex_probability) const {
    if (pattern_vertex_probability == nullptr) {
        return bad_arguments;
    }
    std::int64_t vertex_count = pattern.get_vertex_count();

    float current_probability = 0.0;
    std::int64_t j = 0;
    for (std::int64_t i = 0; i < vertex_count; i++) {
        pattern_vertex_probability[i] = 1.0;
        current_probability = p_degree_probability[pattern.get_vertex_degree(i)];
        for (j = pattern.get_vertex_degree(i) + 1; j < degree_max_size; j++) {
            current_probability += p_degree_probability[j];
        }
        pattern_vertex_probability[i] *= current_probability;

        current_probability = p_vertex_attribute_probability[pattern.get_vertex_attribute(i)];
        pattern_vertex_probability[i] *= current_probability;
    }
    return ok;
}

graph_status sorter::sorting_pattern_vertices(const graph& pattern,
                                              const float* pattern_vertex_probability,
                                              std::int64_t* sorted_pattern_vertex) const {
    if (pattern_vertex_probability == nullptr || sorted_pattern_vertex == nullptr) {
        return bad_arguments;
    }

    std::int64_t vertex_count = pattern.get_vertex_count();
    std::int64_t bit_array_size = bit_vector::bit_vector_size(vertex_count);
    std::int64_t sorted_vertex_iterator = 0;

    bit_vector vertex_candidates(bit_array_size, allocator_.get_byte_allocator());
    bit_vector filling_mask(bit_array_size, allocator_.get_byte_allocator());

    std::int64_t index =
        find_minimum_probability_index_by_mask(pattern, pattern_vertex_probability);
    if (index >= 0) {
        sorted_pattern_vertex[sorted_vertex_iterator] = static_cast<std::int64_t>(index);
        filling_mask.set_bit(sorted_pattern_vertex[sorted_vertex_iterator]);
        sorted_vertex_iterator++;
    }
    else {
        return static_cast<graph_status>(index);
    }

    for (sorted_vertex_iterator; sorted_vertex_iterator < vertex_count; sorted_vertex_iterator++) {
        vertex_candidates |= pattern.p_edges_bit[sorted_pattern_vertex[sorted_vertex_iterator - 1]];
        vertex_candidates.andn(filling_mask);

        index = find_minimum_probability_index_by_mask(pattern,
                                                       pattern_vertex_probability,
                                                       vertex_candidates.get_vector_pointer(),
                                                       filling_mask.get_vector_pointer());
        if (index >= 0) {
            sorted_pattern_vertex[sorted_vertex_iterator] = static_cast<std::int64_t>(index);
            filling_mask.set_bit(sorted_pattern_vertex[sorted_vertex_iterator]);
        }
        else {
            return static_cast<graph_status>(index);
        }
    }
    /* TODO add disconnected case handling by splitting into separate trees */
    return ok;
}

std::int64_t sorter::find_minimum_probability_index_by_mask(
    const graph& pattern,
    const float* pattern_vertex_probability,
    const std::uint8_t* pbit_mask,
    const std::uint8_t* pbit_core_mask) const {
    if (pattern_vertex_probability == nullptr) {
        return bad_allocation;
    }

    std::int64_t vertex_count = pattern.get_vertex_count();
    float global_minimum = 1.1;
    std::int64_t result = -1;
    std::int64_t bit_array_size = bit_vector::bit_vector_size(vertex_count);
    // find minimum for all elements
    if (pbit_mask == nullptr || bit_vector::popcount(bit_array_size, pbit_mask) == 0) {
        for (std::int64_t i = 0; i < vertex_count; i++) {
            if (pbit_core_mask == nullptr ||
                !bit_vector::test_bit(bit_array_size, pbit_core_mask, i)) {
                if (pattern_vertex_probability[i] < global_minimum) {
                    global_minimum = pattern_vertex_probability[i];
                    result = i;
                }
                else if (pattern_vertex_probability[i] == global_minimum) {
                    if (pattern.get_vertex_degree(i) > pattern.get_vertex_degree(result)) {
                        global_minimum = pattern_vertex_probability[i];
                        result = i;
                    }
                }
            }
        }
    }
    else { //find minimum for candidates
        std::int64_t vertex_iterator = 0;
        std::int64_t vertex_core_degree = 0;
        std::uint8_t current_byte = 0;
        std::uint8_t current_bit = 0;

        for (std::int64_t i = 0; i < bit_array_size; i++) {
            current_byte = pbit_mask[i];
            while (current_byte > 0) {
                current_bit = current_byte & (-current_byte);
                current_byte &= ~current_bit;
                vertex_iterator = 8 * i + bit_vector::power_of_two(current_bit);
                if (vertex_iterator >= vertex_count) {
                    return result;
                }

                if (pattern_vertex_probability[vertex_iterator] < global_minimum) {
                    global_minimum = pattern_vertex_probability[vertex_iterator];
                    result = vertex_iterator;
                    vertex_core_degree =
                        get_core_linked_degree(pattern, vertex_iterator, pbit_core_mask);
                }
                else if (pattern_vertex_probability[vertex_iterator] == global_minimum) {
                    std::int64_t current_vertex_core_degree =
                        get_core_linked_degree(pattern, vertex_iterator, pbit_core_mask);
                    if (current_vertex_core_degree > vertex_core_degree) {
                        global_minimum = pattern_vertex_probability[vertex_iterator];
                        result = vertex_iterator;
                        vertex_core_degree = current_vertex_core_degree;
                    }
                    else if (current_vertex_core_degree == vertex_core_degree &&
                             pattern.get_vertex_degree(vertex_iterator) >
                                 pattern.get_vertex_degree(result)) {
                        global_minimum = pattern_vertex_probability[vertex_iterator];
                        result = vertex_iterator;
                        vertex_core_degree = current_vertex_core_degree;
                    }
                }
            }
        }
    }
    return result;
}

graph_status sorter::create_sorted_pattern_tree(const graph& pattern,
                                                const std::int64_t* sorted_pattern_vertex,
                                                std::int64_t* predecessor,
                                                edge_direction* direction,
                                                sconsistent_conditions* cconditions,
                                                bool predecessor_in_core_indexing) const {
    if (sorted_pattern_vertex == nullptr || predecessor == nullptr || direction == nullptr ||
        cconditions == nullptr) {
        return bad_allocation;
    }

    std::int64_t vertex_count = pattern.get_vertex_count();
    if (vertex_count == 0) {
        return bad_arguments;
    }

    predecessor[sorted_pattern_vertex[0]] = null_node;
    direction[sorted_pattern_vertex[0]] = none;

    std::int64_t _p = 0;
    std::int64_t _n = 0;

    for (std::int64_t i = 1; i < vertex_count; i++) {
        predecessor[sorted_pattern_vertex[i]] = null_node;

        _p = i - 1;
        _n = 0;
        for (std::int64_t j = 0; j < i; j++) {
            edge_direction edir =
                pattern.check_edge(sorted_pattern_vertex[j], sorted_pattern_vertex[i]);
            switch (edir) {
                case none: {
                    cconditions[i - 1].array[_n] = j;
                    _n++;
                    break;
                }
                case both: {
                    cconditions[i - 1].array[_p] = j;
                    _p--;
                    break;
                }
            }

            if (edir != none && predecessor[sorted_pattern_vertex[i]] == null_node) {
                if (predecessor_in_core_indexing) {
                    predecessor[sorted_pattern_vertex[i]] = j;
                }
                else {
                    predecessor[sorted_pattern_vertex[i]] = sorted_pattern_vertex[j];
                }
                direction[sorted_pattern_vertex[i]] = edir;
            }
        }
        cconditions[i - 1].divider = _n;
    }
    return ok;
}

std::int64_t sorter::get_core_linked_degree(const graph& pattern,
                                            const std::int64_t vertex,
                                            const std::uint8_t* pbit_mask) const {
    std::int64_t vertex_count = pattern.get_vertex_count();
    std::int64_t bit_array_size = bit_vector::bit_vector_size(vertex_count);
    bit_vector vertex_candidates(bit_array_size, allocator_.get_byte_allocator());
    std::int64_t core_degree = 0;

    vertex_candidates |= pattern.p_edges_bit[vertex];
    vertex_candidates &= pbit_mask;
    core_degree = vertex_candidates.popcount();

    bit_vector::set(bit_array_size, vertex_candidates.get_vector_pointer(), pbit_mask);
    vertex_candidates &= pattern.p_edges_bit[vertex];
    core_degree += vertex_candidates.popcount();

    return core_degree;
}

} // namespace oneapi::dal::preview::subgraph_isomorphism::detail
