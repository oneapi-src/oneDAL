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
                    sconsistent_conditions<Cpu> const* pcconditions,
                    kind isomorphism_kind,
                    inner_alloc allocator);
    matching_engine(const matching_engine& _matching_engine,
                    stack<Cpu>& _local_stack,
                    inner_alloc allocator);
    virtual ~matching_engine();

    void run_and_wait(global_stack<Cpu>& gstack,
                      std::int64_t& busy_engine_count,
                      std::int64_t& current_match_count,
                      std::int64_t target_match_count,
                      bool main_engine);
    solution<Cpu> get_solution();
    std::int64_t get_match_count() const;

    std::int64_t state_exploration();
    std::int64_t state_exploration_bit(bool check_solution = true);
    std::int64_t state_exploration_list(bool check_solution = true);

    std::int64_t first_states_generator(dfs_stack<Cpu>& stack);

    void push_into_stack(const std::int64_t vertex_id);
    bool match_vertex(const std::int64_t pattern_vertex, const std::int64_t target_vertex) const;

    inner_alloc allocator_;
    const graph<Cpu>* pattern;
    const graph<Cpu>* target;
    const std::int64_t* sorted_pattern_vertex;
    const std::int64_t* predecessor;
    const edge_direction* direction;
    const sconsistent_conditions<Cpu>* pconsistent_conditions;

    std::int64_t solution_length;
    bit_vector<Cpu> vertex_candidates;

    std::int64_t temporary_list_size;
    std::int64_t* temporary_list;

    std::uint8_t* pstart_byte;

    stack<Cpu> local_stack;
    dfs_stack<Cpu> hlocal_stack;
    solution<Cpu> engine_solutions;

    kind isomorphism_kind_;

    std::int64_t extract_candidates(bool check_solution);
    bool check_vertex_candidate(bool check_solution, std::int64_t candidate);
};

template <typename Cpu>
class engine_bundle {
public:
    stack<Cpu> exploration_stack;
    engine_bundle(const graph<Cpu>* ppattern,
                  const graph<Cpu>* ptarget,
                  const std::int64_t* psorted_pattern_vertex,
                  const std::int64_t* ppredecessor,
                  const edge_direction* pdirection,
                  sconsistent_conditions<Cpu> const* pcconditions,
                  float* ppattern_vertex_probability,
                  kind isomorphism_kind,
                  inner_alloc allocator);
    virtual ~engine_bundle();
    solution<Cpu> run(std::int64_t max_match_count);

    inner_alloc allocator_;
    const graph<Cpu>* pattern;
    const graph<Cpu>* target;
    const std::int64_t* sorted_pattern_vertex;
    const std::int64_t* predecessor;
    const edge_direction* direction;
    const sconsistent_conditions<Cpu>* pconsistent_conditions;
    const float* pattern_vertex_probability;
    kind isomorphism_kind_;

    solution<Cpu> bundle_solutions;

    typedef oneapi::dal::detail::tls_mem<matching_engine<Cpu>, std::allocator<double>> bundle;
    bundle matching_bundle;
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
                                      sconsistent_conditions<Cpu> const* pcconditions,
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
                                      stack<Cpu>& _local_stack,
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
        popcnt = ONEDAL_popcnt64<Cpu>(*ptr);
        ONEDAL_ASSERT(popcnt <= 64);
        for (std::int64_t j = 0; j < popcnt; j++) {
            std::int64_t candidate = 63 - ONEDAL_lzcnt_u64<Cpu>(*ptr);
            (*ptr) ^= (std::uint64_t)1 << candidate;
            candidate += (i << 6);
            feasible_result_count += check_vertex_candidate(check_solution, candidate);
        }
    }
    for (std::int64_t i = (size_in_dword << 3); i < vertex_candidates.size(); i++) {
        while (pstart_byte[i] > 0) {
            std::int64_t candidate = bit_vector<Cpu>::power_of_two(pstart_byte[i]);
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
                engine_solutions.add(&solution_core);
            }
        }
        else {
            hlocal_stack.push_into_next_level(candidate);
        }
        return true;
    }
    return false;
}

template <typename Cpu>
std::int64_t matching_engine<Cpu>::state_exploration_list(bool check_solution) {
    std::uint64_t current_level_index = hlocal_stack.get_current_level_index();
    std::int64_t divider = pconsistent_conditions[current_level_index].divider;

    if (isomorphism_kind_ != kind::non_induced) {
        ONEDAL_IVDEP
        for (std::int64_t j = 0; j < divider; j++) {
            or_equal<Cpu>(vertex_candidates.get_vector_pointer(),
                          target->p_edges_list[hlocal_stack.top(
                              pconsistent_conditions[current_level_index].array[j])],
                          target->p_degree[hlocal_stack.top(
                              pconsistent_conditions[current_level_index].array[j])]);
        }
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
void matching_engine<Cpu>::push_into_stack(const std::int64_t vertex_id) {
    hlocal_stack.push_into_current_level(vertex_id);
}

template <typename Cpu>
std::int64_t matching_engine<Cpu>::first_states_generator(dfs_stack<Cpu>& stack) {
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
solution<Cpu> matching_engine<Cpu>::get_solution() {
    return std::move(engine_solutions);
}

template <typename Cpu>
std::int64_t matching_engine<Cpu>::get_match_count() const {
    return engine_solutions.get_solution_count();
}

template <typename Cpu>
std::int64_t matching_engine<Cpu>::state_exploration() {
    if (target->bit_representation) {
        return state_exploration_bit();
    }
    else {
        return state_exploration_list();
    }
}

template <typename Cpu>
void matching_engine<Cpu>::run_and_wait(global_stack<Cpu>& gstack,
                                        std::int64_t& busy_engine_count,
                                        std::int64_t& current_match_count,
                                        std::int64_t target_match_count,
                                        bool main_engine) {
    if (main_engine) {
        first_states_generator(hlocal_stack);
    }
    bool is_busy_engine = true;
    ONEDAL_ASSERT(pattern != nullptr);
    for (;;) {
        if (target_match_count > 0 &&
            dal::detail::atomic_load(current_match_count) >= target_match_count) {
            return;
        }
        if (hlocal_stack.states_in_stack() > 0) {
            while ((hlocal_stack.states_in_stack() > 5) && gstack.push(hlocal_stack))
                ;
            ONEDAL_ASSERT(hlocal_stack.states_in_stack() > 0);
            const std::int64_t prev_match_cout = get_match_count();
            state_exploration();
            if (target_match_count > 0) {
                auto delta = get_match_count() - prev_match_cout;
                if (delta > 0) {
                    dal::detail::atomic_increment(current_match_count, delta);
                }
                if (dal::detail::atomic_load(current_match_count) >= target_match_count) {
                    return;
                }
            }
        }
        else {
            gstack.pop(hlocal_stack);
            if (hlocal_stack.empty()) {
                if (is_busy_engine) {
                    is_busy_engine = false;
                    dal::detail::atomic_decrement(busy_engine_count);
                }
                if (dal::detail::atomic_load(busy_engine_count) == 0)
                    break;
            }
            else if (!is_busy_engine) {
                is_busy_engine = true;
                dal::detail::atomic_increment(busy_engine_count);
            }
        }
    }

    return;
}

template <typename Cpu>
engine_bundle<Cpu>::engine_bundle(const graph<Cpu>* ppattern,
                                  const graph<Cpu>* ptarget,
                                  const std::int64_t* psorted_pattern_vertex,
                                  const std::int64_t* ppredecessor,
                                  const edge_direction* pdirection,
                                  sconsistent_conditions<Cpu> const* pcconditions,
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

    bundle_solutions =
        solution<Cpu>(pattern->get_vertex_count(), psorted_pattern_vertex, allocator_);
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
solution<Cpu> engine_bundle<Cpu>::run(std::int64_t max_match_count) {
    std::int64_t degree = pattern->get_vertex_degree(sorted_pattern_vertex[0]);

    std::uint64_t first_states_count =
        pattern_vertex_probability[0] * target->get_vertex_count() + 1;
    std::uint64_t max_threads_count = dal::detail::threader_get_max_threads();
    std::uint64_t possible_first_states_count_per_thread = first_states_count / max_threads_count;
    if (possible_first_states_count_per_thread < 1) {
        possible_first_states_count_per_thread = 1;
    }
    else {
        possible_first_states_count_per_thread +=
            static_cast<bool>(first_states_count % max_threads_count);
    }

    const std::uint64_t array_size =
        (max_threads_count >= 64)
            ? max_threads_count * 2 / 10
            : (max_threads_count >= 24)
                  ? max_threads_count * 4 / 10
                  : (max_threads_count >= 8) ? 4 : (max_threads_count >= 4) ? 2 : 1;
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

    state<Cpu> null_state(allocator_);
    std::uint64_t task_counter = 0, index = 0;
    for (std::int64_t i = 0; i < target->get_vertex_count(); ++i) {
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

    global_stack<Cpu> gstack(pattern->get_vertex_count(), allocator_);
    std::int64_t busy_engine_count(array_size);
    std::int64_t current_match_count(0);
    dal::detail::threader_for(array_size, array_size, [&](const int index) {
        engine_array[index].run_and_wait(gstack,
                                         busy_engine_count,
                                         current_match_count,
                                         max_match_count,
                                         false);
    });

    for (std::uint64_t i = 0; i < array_size; i++) {
        bundle_solutions.add(engine_array[i].get_solution());
        engine_array[i].~matching_engine();
    }

    return std::move(bundle_solutions);
}
} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
