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

#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/inner_alloc.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/bit_vector.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/graph.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

const std::int64_t null_node = 0xffffffffffffffff; /*!< Null node value*/

template <typename Cpu>
struct sconsistent_conditions {
    std::int64_t* array;
    std::int64_t divider;
    std::int64_t length;
    void init(std::int64_t size) {
        length = size;
        array = allocator.allocate<std::int64_t>(length);
        divider = length;
    }
    sconsistent_conditions(std::int64_t size, inner_alloc alloc) : allocator(alloc) {
        init(size);
    }
    ~sconsistent_conditions() {
        if (array != nullptr) {
            allocator.deallocate(array, length);
            array = nullptr;
        }
    }

private:
    inner_alloc allocator;
};

template <typename Cpu>
class sorter {
public:
    sorter(inner_alloc alloc);
    sorter(const graph<Cpu>* ptarget, inner_alloc alloc);
    virtual ~sorter();

    void get_pattern_vertex_probability(const graph<Cpu>& pattern,
                                        float* pattern_vertex_probability) const;
    void sorting_pattern_vertices(const graph<Cpu>& pattern,
                                  const float* pattern_vertex_probability,
                                  std::int64_t* sorted_pattern_vertex) const;
    void create_sorted_pattern_tree(const graph<Cpu>& pattern,
                                    const std::int64_t* sorted_pattern_vertex,
                                    std::int64_t* predecessor,
                                    edge_direction* direction,
                                    sconsistent_conditions<Cpu>* cconditions,
                                    bool predecessor_in_core_indexing = false) const;

    inner_alloc allocator;
    float* p_degree_probability;
    float* p_vertex_attribute_probability;

    std::int64_t degree_max_size;
    std::int64_t vertex_attribute_max_size;

    std::int64_t find_minimum_probability_index_by_mask(
        const graph<Cpu>& pattern,
        const float* pattern_vertex_probability,
        const std::uint8_t* pbit_mask = nullptr,
        const std::uint8_t* pbit_core_mask = nullptr) const;
    std::int64_t get_core_linked_degree(const graph<Cpu>& pattern,
                                        const std::int64_t vertex,
                                        const std::uint8_t* pbit_mask) const;
};

template <typename Cpu>
sorter<Cpu>::sorter(const graph<Cpu>* target, inner_alloc alloc) : sorter(alloc) {
    degree_max_size = target->get_max_degree() + 1;
    vertex_attribute_max_size = target->get_max_vertex_attribute() + 1;

    std::int64_t vertex_count = target->get_vertex_count();

    p_degree_probability = allocator.allocate<float>(degree_max_size);
    if (p_degree_probability == nullptr) {
        return;
    }
    p_vertex_attribute_probability = allocator.allocate<float>(vertex_attribute_max_size);
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

    for (std::int64_t i = 0; i < vertex_count; i++) {
        p_degree_probability[target->get_vertex_degree(i)] += delta_probability;

        if (vertex_attribute_max_size > 1) {
            p_vertex_attribute_probability[target->get_vertex_attribute(i)] += delta_probability;
        }
    }
}

template <typename Cpu>
sorter<Cpu>::sorter(inner_alloc alloc) : allocator(alloc) {
    p_degree_probability = nullptr;
    p_vertex_attribute_probability = nullptr;
    degree_max_size = 0;
    vertex_attribute_max_size = 0;
}

template <typename Cpu>
sorter<Cpu>::~sorter() {
    if (p_degree_probability != nullptr) {
        allocator.deallocate(p_degree_probability, degree_max_size);
        p_degree_probability = nullptr;
    }

    if (p_vertex_attribute_probability != nullptr) {
        allocator.deallocate(p_vertex_attribute_probability, vertex_attribute_max_size);
        p_vertex_attribute_probability = nullptr;
    }
}

template <typename Cpu>
void sorter<Cpu>::get_pattern_vertex_probability(const graph<Cpu>& pattern,
                                                 float* pattern_vertex_probability) const {
    ONEDAL_ASSERT(pattern_vertex_probability != nullptr);
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
}

template <typename Cpu>
void sorter<Cpu>::sorting_pattern_vertices(const graph<Cpu>& pattern,
                                           const float* pattern_vertex_probability,
                                           std::int64_t* sorted_pattern_vertex) const {
    ONEDAL_ASSERT(pattern_vertex_probability != nullptr);
    ONEDAL_ASSERT(sorted_pattern_vertex != nullptr);

    std::int64_t vertex_count = pattern.get_vertex_count();
    std::int64_t bit_array_size = bit_vector<Cpu>::bit_vector_size(vertex_count);
    std::int64_t sorted_vertex_iterator = 0;

    bit_vector<Cpu> vertex_candidates(bit_array_size, allocator.get_byte_allocator());
    bit_vector<Cpu> filling_mask(bit_array_size, allocator.get_byte_allocator());

    std::int64_t index =
        find_minimum_probability_index_by_mask(pattern, pattern_vertex_probability);
    if (index >= 0) {
        sorted_pattern_vertex[sorted_vertex_iterator] = static_cast<std::int64_t>(index);
        filling_mask.set_bit(sorted_pattern_vertex[sorted_vertex_iterator]);
        sorted_vertex_iterator++;
    }
    else {
        throw oneapi::dal::internal_error(
            dal::detail::error_messages::incorrect_index_is_returned());
    }

    for (; sorted_vertex_iterator < vertex_count; sorted_vertex_iterator++) {
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
            throw oneapi::dal::internal_error(
                dal::detail::error_messages::incorrect_index_is_returned());
        }
    }
}

template <typename Cpu>
std::int64_t sorter<Cpu>::find_minimum_probability_index_by_mask(
    const graph<Cpu>& pattern,
    const float* pattern_vertex_probability,
    const std::uint8_t* pbit_mask,
    const std::uint8_t* pbit_core_mask) const {
    ONEDAL_ASSERT(pattern_vertex_probability != nullptr);

    std::int64_t vertex_count = pattern.get_vertex_count();
    float global_minimum = 1.1;
    std::int64_t result = -1;
    std::int64_t bit_array_size = bit_vector<Cpu>::bit_vector_size(vertex_count);
    if (pbit_mask == nullptr || bit_vector<Cpu>::popcount(bit_array_size, pbit_mask) == 0) {
        for (std::int64_t i = 0; i < vertex_count; i++) {
            if (pbit_core_mask == nullptr ||
                !bit_vector<Cpu>::test_bit(bit_array_size, pbit_core_mask, i)) {
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
                vertex_iterator = 8 * i + bit_vector<Cpu>::power_of_two(current_bit);
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

template <typename Cpu>
void sorter<Cpu>::create_sorted_pattern_tree(const graph<Cpu>& pattern,
                                             const std::int64_t* sorted_pattern_vertex,
                                             std::int64_t* predecessor,
                                             edge_direction* direction,
                                             sconsistent_conditions<Cpu>* cconditions,
                                             bool predecessor_in_core_indexing) const {
    ONEDAL_ASSERT(sorted_pattern_vertex != nullptr);
    ONEDAL_ASSERT(predecessor != nullptr);
    ONEDAL_ASSERT(direction != nullptr);
    ONEDAL_ASSERT(cconditions != nullptr);

    std::int64_t vertex_count = pattern.get_vertex_count();
    ONEDAL_ASSERT(vertex_count != 0);

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
                    ONEDAL_ASSERT(_n < i);
                    ONEDAL_ASSERT(_n >= 0);
                    cconditions[i - 1].array[_n] = j;
                    _n++;
                    break;
                }
                case both: {
                    ONEDAL_ASSERT(_p < i);
                    ONEDAL_ASSERT(_p >= 0);
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
}

template <typename Cpu>
std::int64_t sorter<Cpu>::get_core_linked_degree(const graph<Cpu>& pattern,
                                                 const std::int64_t vertex,
                                                 const std::uint8_t* pbit_mask) const {
    std::int64_t vertex_count = pattern.get_vertex_count();
    std::int64_t bit_array_size = bit_vector<Cpu>::bit_vector_size(vertex_count);
    bit_vector<Cpu> vertex_candidates(bit_array_size, allocator.get_byte_allocator());
    std::int64_t core_degree = 0;

    vertex_candidates |= pattern.p_edges_bit[vertex];
    vertex_candidates &= pbit_mask;
    core_degree = vertex_candidates.popcount();

    bit_vector<Cpu>::set(bit_array_size, vertex_candidates.get_vector_pointer(), pbit_mask);
    vertex_candidates &= pattern.p_edges_bit[vertex];
    core_degree += vertex_candidates.popcount();

    return core_degree;
}

} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
