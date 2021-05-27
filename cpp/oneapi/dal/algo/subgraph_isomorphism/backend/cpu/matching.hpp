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

#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/sorter.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/solution.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/stack.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/bit_vector.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/graph.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/common.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

template <typename Cpu>
class engine_bundle;

template <typename Cpu>
class matching_engine {
public:
    matching_engine(inner_alloc allocator)
            : allocator_(allocator),
              vertex_candidates(allocator.get_byte_allocator()),
              local_stack(allocator),
              hlocal_stack(allocator),
              engine_solutions(allocator){};
    matching_engine(const graph<Cpu>* ppattern,
                    const graph<Cpu>* ptarget,
                    const std::int64_t* psorted_pattern_vertex,
                    const std::int64_t* ppredecessor,
                    const edge_direction* pdirection,
                    sconsistent_conditions const* pcconditions,
                    kind isomorphism_kind,
                    inner_alloc allocator);
    matching_engine(const matching_engine& _matching_engine,
                    stack& _local_stack,
                    inner_alloc allocator);
    virtual ~matching_engine();

    solution run(bool main_engine = false);
    void run_and_wait(bool main_engine = false);
    solution get_solution();

    std::int64_t state_exploration_bit(state* current_state, bool check_solution = true);
    std::int64_t state_exploration_list(state* current_state, bool check_solution = true);

    std::int64_t state_exploration_bit(bool check_solution = true);
    std::int64_t state_exploration_list(bool check_solution = true);

    std::int64_t first_states_generator(stack& stack);
    std::int64_t first_states_generator(dfs_stack& stack);

    void push_into_stack(state* _state);
    void push_into_stack(const std::int64_t vertex_id);
    bool match_vertex(const std::int64_t pattern_vertex, const std::int64_t target_vertex) const;
    bool check_vertex_candidate(const std::int64_t pattern_vertex,
                                const std::int64_t target_vertex);

    inner_alloc allocator_;
    const graph<Cpu>* pattern;
    const graph<Cpu>* target;
    const std::int64_t* sorted_pattern_vertex;
    const std::int64_t* predecessor;
    const edge_direction* direction;
    const sconsistent_conditions* pconsistent_conditions;

    std::int64_t solution_length;
    bit_vector<Cpu> vertex_candidates;

    std::int64_t temporary_list_size;
    std::int64_t* temporary_list;

    std::uint8_t* pstart_byte;

    stack local_stack;
    dfs_stack hlocal_stack;
    solution engine_solutions;

    kind isomorphism_kind_;

    std::int64_t extract_candidates(state* current_state, bool check_solution);
    bool check_vertex_candidate(state* current_state, bool check_solution, std::int64_t candidate);

    std::int64_t extract_candidates(bool check_solution);
    bool check_vertex_candidate(bool check_solution, std::int64_t candidate);
};

template <typename Cpu>
class engine_bundle {
public:
    stack exploration_stack;
    engine_bundle(const graph<Cpu>* ppattern,
                  const graph<Cpu>* ptarget,
                  const std::int64_t* psorted_pattern_vertex,
                  const std::int64_t* ppredecessor,
                  const edge_direction* pdirection,
                  sconsistent_conditions const* pcconditions,
                  float* ppattern_vertex_probability,
                  kind isomorphism_kind,
                  inner_alloc allocator);
    virtual ~engine_bundle();
    solution run();

    inner_alloc allocator_;
    const graph<Cpu>* pattern;
    const graph<Cpu>* target;
    const std::int64_t* sorted_pattern_vertex;
    const std::int64_t* predecessor;
    const edge_direction* direction;
    const sconsistent_conditions* pconsistent_conditions;
    const float* pattern_vertex_probability;
    kind isomorphism_kind_;

    solution bundle_solutions;

    typedef oneapi::dal::detail::tls_mem<matching_engine<Cpu>, std::allocator<double>> bundle;
    bundle matching_bundle;
    void first_states_generator(bool use_exploration_stack = true);
};

template <typename Cpu>
matching_engine<Cpu>::~matching_engine() {
    pattern = nullptr;
    target = nullptr;
    sorted_pattern_vertex = nullptr;
    predecessor = nullptr;
    direction = nullptr;
    pconsistent_conditions = nullptr;

    allocator_.deallocate<std::int64_t>(temporary_list, temporary_list_size);
    temporary_list = nullptr;
    temporary_list_size = 0;
}

template <typename Cpu>
matching_engine<Cpu>::matching_engine(const graph<Cpu>* ppattern,
                                      const graph<Cpu>* ptarget,
                                      const std::int64_t* psorted_pattern_vertex,
                                      const std::int64_t* ppredecessor,
                                      const edge_direction* pdirection,
                                      sconsistent_conditions const* pcconditions,
                                      kind isomorphism_kind,
                                      inner_alloc allocator)
        : allocator_(allocator),
          vertex_candidates(bit_vector<Cpu>::bit_vector_size(ptarget->get_vertex_count()),
                            allocator),
          local_stack(allocator),
          hlocal_stack(allocator),
          engine_solutions(ppattern->get_vertex_count(), psorted_pattern_vertex, allocator),
          isomorphism_kind_(isomorphism_kind) {
    pattern = ppattern;
    target = ptarget;
    sorted_pattern_vertex = psorted_pattern_vertex;
    predecessor = ppredecessor;
    direction = pdirection;
    pconsistent_conditions = pcconditions;

    solution_length = pattern->get_vertex_count();

    std::int64_t target_vertex_count = target->get_vertex_count();

    pstart_byte = vertex_candidates.get_vector_pointer();

    std::int64_t max_neighbours_size = target->get_max_degree();
    std::int64_t max_degree = target->get_max_degree();
    if (max_neighbours_size < max_degree) {
        max_neighbours_size = max_degree;
    }

    hlocal_stack.init(solution_length - 1, target_vertex_count);

    if (target->bit_representation) {
        temporary_list = nullptr;
    }
    else {
        temporary_list_size = max_neighbours_size;
        temporary_list = allocator_.allocate<std::int64_t>(temporary_list_size);
    }
}

template <typename Cpu>
matching_engine<Cpu>::matching_engine(const matching_engine& _matching_engine,
                                      stack& _local_stack,
                                      inner_alloc allocator)
        : matching_engine(_matching_engine.pattern,
                          _matching_engine.target,
                          _matching_engine.sorted_pattern_vertex,
                          _matching_engine.predecessor,
                          _matching_engine.direction,
                          _matching_engine.pconsistent_conditions,
                          _matching_engine.isomorphism_kind_,
                          allocator) {
    local_stack = std::move(_local_stack);
}

template <typename Cpu>
std::int64_t matching_engine<Cpu>::state_exploration_bit(state* current_state,
                                                         bool check_solution) {
    const std::int64_t i_cc = current_state->core_length - 1;
    const std::int64_t divider = pconsistent_conditions[i_cc].divider;

    if (isomorphism_kind_ != kind::non_induced) {
        ONEDAL_IVDEP
        for (std::int64_t j = 0; j < divider; j++) {
            or_equal<Cpu>(
                vertex_candidates.get_vector_pointer(),
                target->p_edges_bit[current_state->core[pconsistent_conditions[i_cc].array[j]]],
                vertex_candidates.size());
        }
    }

    ~vertex_candidates; // inversion?

    ONEDAL_IVDEP
    for (std::int64_t j = i_cc; j >= divider; j--) { // > divider - 1
        and_equal<Cpu>(
            vertex_candidates.get_vector_pointer(),
            target->p_edges_bit[current_state->core[pconsistent_conditions[i_cc].array[j]]],
            vertex_candidates.size());
    }

    for (std::int64_t i = 0; i < current_state->core_length; i++) {
        vertex_candidates.get_vector_pointer()[bit_vector<Cpu>::byte(current_state->core[i])] &=
            ~bit_vector<Cpu>::bit(current_state->core[i]);
    }
    return extract_candidates(current_state, check_solution);
}

template <typename Cpu>
std::int64_t matching_engine<Cpu>::state_exploration_bit(bool check_solution) {
    std::uint64_t current_level_index = hlocal_stack.get_current_level_index();
    std::int64_t divider = pconsistent_conditions[current_level_index].divider;

    if (isomorphism_kind_ != kind::non_induced) {
        ONEDAL_IVDEP
        for (std::int64_t j = 0; j < divider; j++) {
            or_equal<Cpu>(vertex_candidates.get_vector_pointer(),
                          target->p_edges_bit[hlocal_stack.top(
                              pconsistent_conditions[current_level_index].array[j])],
                          vertex_candidates.size());
        }
    }

    ~vertex_candidates;

    ONEDAL_IVDEP
    for (std::int64_t j = current_level_index; j >= divider; j--) { //j > divider - 1
        and_equal<Cpu>(vertex_candidates.get_vector_pointer(),
                       target->p_edges_bit[hlocal_stack.top(
                           pconsistent_conditions[current_level_index].array[j])],
                       vertex_candidates.size());
    }

    for (std::uint64_t i = 0; i <= current_level_index; i++) {
        vertex_candidates.get_vector_pointer()[bit_vector<Cpu>::byte(hlocal_stack.top(i))] &=
            ~bit_vector<Cpu>::bit(hlocal_stack.top(i));
    }

    return extract_candidates(check_solution);
}

template <typename Cpu>
std::int64_t matching_engine<Cpu>::extract_candidates(bool check_solution) {
    std::int64_t feasible_result_count = 0;
    std::int64_t size_in_dword = vertex_candidates.size() >> 3;
    std::uint64_t* ptr;
    std::int32_t popcnt;
    for (std::int64_t i = 0; i < size_in_dword; i++) {
        ptr = (std::uint64_t*)(pstart_byte + (i << 3));
        popcnt = ONEDAL_popcnt64(*ptr);
        ONEDAL_ASSERT(popcnt <= 64);
        for (std::int64_t j = 0; j < popcnt; j++) {
            std::int64_t candidate = 63 - ONEDAL_lzcnt_u64(*ptr);
            (*ptr) ^= (std::uint64_t)1 << candidate;
            candidate += (i << 6);
            feasible_result_count += check_vertex_candidate(check_solution, candidate);
        }
    }
    for (std::int64_t i = (size_in_dword << 3); i < vertex_candidates.size(); i++) {
        while (pstart_byte[i] > 0) {
            std::int64_t candidate = bit_vector<Cpu>::power_of_two(pstart_byte[i]);
            ONEDAL_ASSERT(candidate < 8);
            pstart_byte[i] ^= (1 << candidate);
            candidate += (i << 3);
            feasible_result_count += check_vertex_candidate(check_solution, candidate);
        }
    }

    hlocal_stack.update();

    return feasible_result_count;
}

template <typename Cpu>
bool matching_engine<Cpu>::check_vertex_candidate(bool check_solution, std::int64_t candidate) {
    std::uint64_t solution_length_unsigned = solution_length;
    if (match_vertex(sorted_pattern_vertex[hlocal_stack.get_current_level()], candidate)) {
        if (check_solution && hlocal_stack.get_current_level() + 1 == solution_length_unsigned) {
            std::int64_t* solution_core = allocator_.allocate<std::int64_t>(solution_length);
            if (solution_core != nullptr) {
                hlocal_stack.fill_solution(solution_core, candidate);
                engine_solutions.add(&solution_core); /* add new state into solution */
            }
        }
        else {
            hlocal_stack.push_into_next_level(candidate); /* add new state into local_stack */
        }
        return true;
    }
    return false;
}

template <typename Cpu>
std::int64_t matching_engine<Cpu>::state_exploration_list(state* current_state,
                                                          bool check_solution) {
    std::int64_t divider = pconsistent_conditions[current_state->core_length - 1].divider;

    ONEDAL_IVDEP
    for (std::int64_t j = 0; j < divider; j++) {
        or_equal<Cpu>(
            vertex_candidates.get_vector_pointer(),
            target->p_edges_list
                [current_state
                     ->core[pconsistent_conditions[current_state->core_length - 1].array[j]]],
            target
                ->p_degree[current_state->core
                               [pconsistent_conditions[current_state->core_length - 1].array[j]]]);
    }

    ~vertex_candidates;

    ONEDAL_IVDEP
    for (std::int64_t j = current_state->core_length - 1; j >= divider; j--) { // j> divider - 1
        and_equal<Cpu>(
            vertex_candidates.get_vector_pointer(),
            target->p_edges_list
                [current_state
                     ->core[pconsistent_conditions[current_state->core_length - 1].array[j]]],
            vertex_candidates.size(),
            target->p_degree[current_state->core
                                 [pconsistent_conditions[current_state->core_length - 1].array[j]]],
            temporary_list);
    }

    for (std::int64_t i = 0; i < current_state->core_length; i++) {
        vertex_candidates.get_vector_pointer()[bit_vector<Cpu>::byte(current_state->core[i])] &=
            ~bit_vector<Cpu>::bit(current_state->core[i]);
    }
    return extract_candidates(current_state, check_solution);
}

template <typename Cpu>
std::int64_t matching_engine<Cpu>::state_exploration_list(bool check_solution) {
    std::uint64_t current_level_index = hlocal_stack.get_current_level_index();
    std::int64_t divider = pconsistent_conditions[current_level_index].divider;

    ONEDAL_IVDEP
    for (std::int64_t j = 0; j < divider; j++) {
        or_equal<Cpu>(
            vertex_candidates.get_vector_pointer(),
            target->p_edges_list[hlocal_stack.top(
                pconsistent_conditions[current_level_index].array[j])],
            target
                ->p_degree[hlocal_stack.top(pconsistent_conditions[current_level_index].array[j])]);
    }

    ~vertex_candidates;

    ONEDAL_IVDEP
    for (std::int64_t j = current_level_index; j >= divider; j--) { //j > divider - 1
        and_equal<Cpu>(
            vertex_candidates.get_vector_pointer(),
            target->p_edges_list[hlocal_stack.top(
                pconsistent_conditions[current_level_index].array[j])],
            vertex_candidates.size(),
            target
                ->p_degree[hlocal_stack.top(pconsistent_conditions[current_level_index].array[j])],
            temporary_list);
    }

    for (std::uint64_t i = 0; i <= current_level_index; i++) {
        vertex_candidates.get_vector_pointer()[bit_vector<Cpu>::byte(hlocal_stack.top(i))] &=
            ~bit_vector<Cpu>::bit(hlocal_stack.top(i));
    }
    return extract_candidates(check_solution);
}

template <typename Cpu>
void matching_engine<Cpu>::push_into_stack(state* _state) {
    local_stack.push(_state);
}

template <typename Cpu>
void matching_engine<Cpu>::push_into_stack(const std::int64_t vertex_id) {
    hlocal_stack.push_into_current_level(vertex_id);
}

template <typename Cpu>
std::int64_t matching_engine<Cpu>::first_states_generator(stack& stack) {
    state null_state(allocator_);
    std::int64_t candidates_count = 0;
    std::int64_t degree = pattern->get_vertex_degree(sorted_pattern_vertex[0]);
    for (std::int64_t i = 0; i < target->get_vertex_count(); i++) {
        if (degree <= target->get_vertex_degree(i) &&
            pattern->get_vertex_attribute(sorted_pattern_vertex[0]) ==
                target->get_vertex_attribute(i)) {
            void* place = (void*)allocator_.allocate<state>(1);
            state* new_state = new (place) state(&null_state, i, allocator_);
            stack.push(new_state);
            candidates_count++;
        }
    }

    return candidates_count;
}

template <typename Cpu>
std::int64_t matching_engine<Cpu>::first_states_generator(dfs_stack& stack) {
    std::int64_t degree = pattern->get_vertex_degree(sorted_pattern_vertex[0]);
    for (std::int64_t i = 0; i < target->get_vertex_count(); i++) {
        if (degree <= target->get_vertex_degree(i) &&
            pattern->get_vertex_attribute(sorted_pattern_vertex[0]) ==
                target->get_vertex_attribute(i)) {
            stack.push_into_current_level(i);
        }
    }

    return stack.get_current_level_fill_size();
}

template <typename Cpu>
std::int64_t matching_engine<Cpu>::extract_candidates(state* current_state, bool check_solution) {
    std::int64_t feasible_result_count = 0;

    std::int64_t size_in_dword = vertex_candidates.size() >> 3;
    std::uint64_t* ptr;
    std::int64_t popcnt;
    for (std::int64_t i = 0; i < size_in_dword; i++) {
        ptr = (std::uint64_t*)(pstart_byte + (i << 3));
        popcnt = ONEDAL_popcnt64(*ptr);
        for (std::int64_t j = 0; j < popcnt; j++) {
            std::int64_t candidate = 63 - ONEDAL_lzcnt_u64(*ptr);
            (*ptr) ^= (std::uint64_t)1 << candidate;
            candidate += (i << 6);
            feasible_result_count +=
                check_vertex_candidate(current_state, check_solution, candidate);
        }
    }
    for (std::int64_t i = (size_in_dword << 3); i < vertex_candidates.size(); i++) {
        while (pstart_byte[i] > 0) {
            std::int64_t candidate = bit_vector<Cpu>::power_of_two(pstart_byte[i]);
            pstart_byte[i] ^= (1 << candidate);
            candidate += (i << 3);
            feasible_result_count +=
                check_vertex_candidate(current_state, check_solution, candidate);
        }
    }
    return feasible_result_count;
}

template <typename Cpu>
bool matching_engine<Cpu>::check_vertex_candidate(const std::int64_t pattern_vertex,
                                                  const std::int64_t target_vertex) {
    if (match_vertex(pattern_vertex, target_vertex)) {
        state null_state(allocator_);
        void* place = (void*)allocator_.allocate<state>(1);
        state* new_state = new (place) state(&null_state, target_vertex, allocator_);
        local_stack.push(new_state); /* add new state into local_stack */
        return true;
    }
    return false;
}

template <typename Cpu>
bool matching_engine<Cpu>::check_vertex_candidate(state* current_state,
                                                  bool check_solution,
                                                  std::int64_t candidate) {
    if (match_vertex(sorted_pattern_vertex[current_state->core_length], candidate)) {
        void* place = (void*)allocator_.allocate<state>(1);
        state* new_state = new (place) state(current_state, candidate, allocator_);

        if (check_solution && new_state->core_length == solution_length) {
            engine_solutions.add(new_state); /* add new state into solution */
        }
        else {
            local_stack.push(new_state); /* add new state into local_stack */
        }
        return true;
    }
    return false;
}

template <typename Cpu>
bool matching_engine<Cpu>::match_vertex(const std::int64_t pattern_vertex,
                                        const std::int64_t target_vertex) const {
    if (target_vertex >= target->get_vertex_count())
        return false;
    ONEDAL_ASSERT(pattern_vertex < pattern->get_vertex_count());
    ONEDAL_ASSERT(target_vertex < target->get_vertex_count());
    return pattern->get_vertex_degree(pattern_vertex) <= target->get_vertex_degree(target_vertex) &&
           pattern->get_vertex_attribute(pattern_vertex) ==
               target->get_vertex_attribute(target_vertex);
}

template <typename Cpu>
solution matching_engine<Cpu>::get_solution() {
    return std::move(engine_solutions);
}

template <typename Cpu>
void matching_engine<Cpu>::run_and_wait(bool main_engine) {
    if (main_engine) {
        first_states_generator(hlocal_stack);
    }
    if (target->bit_representation) { /* dense graph case */
        while (hlocal_stack.states_in_stack() > 0) {
            state_exploration_bit();
        }
    }
    else { /* sparse graph case */
        while (hlocal_stack.states_in_stack() > 0) {
            state_exploration_list();
        }
    }
    return;
}

template <typename Cpu>
solution matching_engine<Cpu>::run(bool main_engine) {
    run_and_wait(main_engine);
    return std::move(engine_solutions);
}

template <typename Cpu>
engine_bundle<Cpu>::engine_bundle(const graph<Cpu>* ppattern,
                                  const graph<Cpu>* ptarget,
                                  const std::int64_t* psorted_pattern_vertex,
                                  const std::int64_t* ppredecessor,
                                  const edge_direction* pdirection,
                                  sconsistent_conditions const* pcconditions,
                                  float* ppattern_vertex_probability,
                                  kind isomorphism_kind,
                                  inner_alloc allocator)
        : exploration_stack(allocator),
          allocator_(allocator),
          isomorphism_kind_(isomorphism_kind),
          bundle_solutions(allocator) {
    pattern = ppattern;
    target = ptarget;
    sorted_pattern_vertex = psorted_pattern_vertex;
    predecessor = ppredecessor;
    direction = pdirection;
    pconsistent_conditions = pcconditions;
    pattern_vertex_probability = ppattern_vertex_probability;

    bundle_solutions = solution(pattern->get_vertex_count(), psorted_pattern_vertex, allocator_);
}

template <typename Cpu>
engine_bundle<Cpu>::~engine_bundle() {
    pattern = nullptr;
    target = nullptr;
    sorted_pattern_vertex = nullptr;
    predecessor = nullptr;
    direction = nullptr;
    pconsistent_conditions = nullptr;
}

template <typename Cpu>
solution engine_bundle<Cpu>::run() {
    std::int64_t degree = pattern->get_vertex_degree(sorted_pattern_vertex[0]);

    std::uint64_t first_states_count =
        pattern_vertex_probability[0] * target->get_vertex_count() + 1;
    std::uint64_t max_threads_count = dal::detail::threader_get_max_threads();
    std::uint64_t possible_first_states_count_per_thread = first_states_count / max_threads_count;
    if (possible_first_states_count_per_thread < 1) {
        max_threads_count = first_states_count;
        possible_first_states_count_per_thread = 1;
    }
    else {
        possible_first_states_count_per_thread +=
            static_cast<bool>(first_states_count % max_threads_count);
    }

    const std::uint64_t array_size = max_threads_count * 2;
    auto engine_array_ptr = allocator_.make_shared_memory<matching_engine<Cpu>>(array_size);
    matching_engine<Cpu>* engine_array = engine_array_ptr.get();

    for (std::uint64_t i = 0; i < array_size; ++i) {
        new (engine_array + i) matching_engine<Cpu>(pattern,
                                                    target,
                                                    sorted_pattern_vertex,
                                                    predecessor,
                                                    direction,
                                                    pconsistent_conditions,
                                                    isomorphism_kind_,
                                                    allocator_);
    }

    state null_state(allocator_);
    std::uint64_t task_counter = 0, index = 0;
    for (std::int64_t i = 0; i < target->n; ++i) {
        if (degree <= target->get_vertex_degree(i) &&
            pattern->get_vertex_attribute(sorted_pattern_vertex[0]) ==
                target->get_vertex_attribute(i)) {
            index = task_counter % array_size;
            engine_array[index].push_into_stack(i);

            if ((engine_array[index].hlocal_stack.states_in_stack() /
                 possible_first_states_count_per_thread) > 0) {
                task_counter++;
            }
        }
    }

    dal::detail::threader_for(array_size, array_size, [&](const int index) {
        engine_array[index].run_and_wait(false);
    });

    for (std::uint64_t i = 0; i < array_size; i++) {
        bundle_solutions.add(engine_array[i].get_solution());
        engine_array[i].~matching_engine();
    }

    return std::move(bundle_solutions);
}

template <typename Cpu>
void engine_bundle<Cpu>::first_states_generator(bool use_exploration_stack) {
    if (use_exploration_stack) {
        typename bundle::ptr_t local_engine = matching_bundle.local();
        local_engine->first_states_generator(exploration_stack);
    }
    else {
        std::int64_t degree = pattern->get_vertex_degree(sorted_pattern_vertex[0]);
        dal::detail::threader_for(target->get_vertex_count(),
                                  target->get_vertex_count(),
                                  [=](const int i) {
                                      typename bundle::ptr_t local_engine = matching_bundle.local();
                                      state null_state(allocator_);
                                      if (degree <= target->get_vertex_degree(i) &&
                                          pattern->get_vertex_attribute(sorted_pattern_vertex[0]) ==
                                              target->get_vertex_attribute(i)) {
                                          void* place = (void*)allocator_.allocate<state>(1);
                                          state* new_state =
                                              new (place) state(&null_state, i, allocator_);
                                          local_engine->local_stack.push(new_state);
                                      }
                                  });
    }
}
} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
