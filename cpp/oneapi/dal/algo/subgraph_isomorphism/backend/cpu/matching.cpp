#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/matching.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/stack.hpp"
#include <atomic>

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

namespace dal = oneapi::dal;

matching_engine::~matching_engine() {
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

matching_engine::matching_engine(const graph* ppattern,
                                 const graph* ptarget,
                                 const std::int64_t* psorted_pattern_vertex,
                                 const std::int64_t* ppredecessor,
                                 const edge_direction* pdirection,
                                 sconsistent_conditions const* pcconditions,
                                 kind isomorphism_kind,
                                 inner_alloc allocator)
        : allocator_(allocator),
          vertex_candidates(allocator),
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

    //Need modification for adj lists case support, ~30Mb for Kron-28 (256 * 10^6 vertices)
    vertex_candidates = bit_vector(bit_vector::bit_vector_size(target_vertex_count), allocator_);

    pstart_byte = vertex_candidates.get_vector_pointer();
    candidate = 0;

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

matching_engine::matching_engine(const matching_engine& _matching_engine,
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

std::int64_t matching_engine::state_exploration_bit(state* current_state, bool check_solution) {
    const std::int64_t i_cc = current_state->core_length - 1;
    const std::int64_t divider = pconsistent_conditions[i_cc].divider;

    if (isomorphism_kind_ != kind::non_induced) {
        ONEDAL_IVDEP
        for (std::int64_t j = 0; j < divider; j++) {
            or_equal(
                vertex_candidates.get_vector_pointer(),
                target->p_edges_bit[current_state->core[pconsistent_conditions[i_cc].array[j]]],
                vertex_candidates.size());
        }
    }

    ~vertex_candidates; // inversion?

    ONEDAL_IVDEP
    for (std::int64_t j = i_cc; j >= divider; j--) { // > divider - 1
        and_equal(vertex_candidates.get_vector_pointer(),
                  target->p_edges_bit[current_state->core[pconsistent_conditions[i_cc].array[j]]],
                  vertex_candidates.size());
    }

    for (std::int64_t i = 0; i < current_state->core_length; i++) {
        vertex_candidates.get_vector_pointer()[bit_vector::byte(current_state->core[i])] &=
            ~bit_vector::bit(current_state->core[i]);
    }
    return extract_candidates(current_state, check_solution);
}

std::int64_t matching_engine::state_exploration_bit(bool check_solution) {
    std::uint64_t current_level_index = hlocal_stack.get_current_level_index();
    std::int64_t divider = pconsistent_conditions[current_level_index].divider;

    if (isomorphism_kind_ != kind::non_induced) {
        ONEDAL_IVDEP
        for (std::int64_t j = 0; j < divider; j++) {
            or_equal(vertex_candidates.get_vector_pointer(),
                     target->p_edges_bit[hlocal_stack.top(
                         pconsistent_conditions[current_level_index].array[j])],
                     vertex_candidates.size());
        }
    }

    ~vertex_candidates; // inversion ?

    ONEDAL_IVDEP
    for (std::int64_t j = current_level_index; j >= divider; j--) { //j > divider - 1
        and_equal(vertex_candidates.get_vector_pointer(),
                  target->p_edges_bit[hlocal_stack.top(
                      pconsistent_conditions[current_level_index].array[j])],
                  vertex_candidates.size());
    }

    for (std::uint64_t i = 0; i <= current_level_index; i++) {
        vertex_candidates.get_vector_pointer()[bit_vector::byte(hlocal_stack.top(i))] &=
            ~bit_vector::bit(hlocal_stack.top(i));
    }
    return extract_candidates(check_solution);
}

std::int64_t matching_engine::extract_candidates(bool check_solution) {
    std::int64_t feasible_result_count = 0;

    std::int64_t size_in_dword = vertex_candidates.size() >> 3;
    std::uint64_t* ptr;
    std::int32_t popcnt;
    for (std::int64_t i = 0; i < size_in_dword; i++) {
        ptr = (std::uint64_t*)(pstart_byte + (i << 3));
        popcnt = ONEDAL_popcnt64(*ptr);
        for (std::int64_t j = 0; j < popcnt; j++) {
            candidate = 63 - ONEDAL_lzcnt_u64(*ptr);
            (*ptr) ^= (std::uint64_t)1 << candidate;
            candidate += (i << 6);
            feasible_result_count += check_vertex_candidate(check_solution);
        }
    }
    for (std::int64_t i = (size_in_dword << 3); i < vertex_candidates.size(); i++) {
        while (pstart_byte[i] > 0) {
            candidate = bit_vector::power_of_two(pstart_byte[i]);
            pstart_byte[i] ^= (1 << candidate);
            candidate += (i << 3);
            feasible_result_count += check_vertex_candidate(check_solution);
        }
    }

    hlocal_stack.update();

    return feasible_result_count;
}

bool matching_engine::check_vertex_candidate(bool check_solution) {
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

std::int64_t matching_engine::state_exploration_list(state* current_state, bool check_solution) {
    std::int64_t divider = pconsistent_conditions[current_state->core_length - 1].divider;

    ONEDAL_IVDEP
    for (std::int64_t j = 0; j < divider; j++) {
        or_equal(vertex_candidates.get_vector_pointer(),
                 target->p_edges_list
                     [current_state
                          ->core[pconsistent_conditions[current_state->core_length - 1].array[j]]],
                 target->p_degree
                     [current_state
                          ->core[pconsistent_conditions[current_state->core_length - 1].array[j]]]);
    }

    ~vertex_candidates;

    ONEDAL_IVDEP
    for (std::int64_t j = current_state->core_length - 1; j >= divider; j--) { // j> divider - 1
        and_equal(
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
        vertex_candidates.get_vector_pointer()[bit_vector::byte(current_state->core[i])] &=
            ~bit_vector::bit(current_state->core[i]);
    }
    return extract_candidates(current_state, check_solution);
}

std::int64_t matching_engine::state_exploration_list(bool check_solution) {
    std::uint64_t current_level_index = hlocal_stack.get_current_level_index();
    std::int64_t divider = pconsistent_conditions[current_level_index].divider;

    ONEDAL_IVDEP
    for (std::int64_t j = 0; j < divider; j++) {
        or_equal(
            vertex_candidates.get_vector_pointer(),
            target->p_edges_list[hlocal_stack.top(
                pconsistent_conditions[current_level_index].array[j])],
            target
                ->p_degree[hlocal_stack.top(pconsistent_conditions[current_level_index].array[j])]);
    }

    ~vertex_candidates;

    ONEDAL_IVDEP
    for (std::int64_t j = current_level_index; j >= divider; j--) { //j > divider - 1
        and_equal(
            vertex_candidates.get_vector_pointer(),
            target->p_edges_list[hlocal_stack.top(
                pconsistent_conditions[current_level_index].array[j])],
            vertex_candidates.size(),
            target
                ->p_degree[hlocal_stack.top(pconsistent_conditions[current_level_index].array[j])],
            temporary_list);
    }

    for (std::uint64_t i = 0; i <= current_level_index; i++) {
        vertex_candidates.get_vector_pointer()[bit_vector::byte(hlocal_stack.top(i))] &=
            ~bit_vector::bit(hlocal_stack.top(i));
    }
    return extract_candidates(check_solution);
}

void matching_engine::push_into_stack(state* _state) {
    local_stack.push(_state);
}

void matching_engine::push_into_stack(const std::int64_t vertex_id) {
    hlocal_stack.push_into_current_level(vertex_id);
}

std::int64_t matching_engine::first_states_generator(stack& stack) {
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

std::int64_t matching_engine::first_states_generator(dfs_stack& stack) {
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

std::int64_t matching_engine::extract_candidates(state* current_state, bool check_solution) {
    std::int64_t feasible_result_count = 0;

    std::int64_t size_in_dword = vertex_candidates.size() >> 3;
    std::uint64_t* ptr;
    std::int64_t popcnt;
    for (std::int64_t i = 0; i < size_in_dword; i++) {
        ptr = (std::uint64_t*)(pstart_byte + (i << 3));
        popcnt = ONEDAL_popcnt64(*ptr);
        for (std::int64_t j = 0; j < popcnt; j++) {
            candidate = 63 - ONEDAL_lzcnt_u64(*ptr);
            (*ptr) ^= (std::uint64_t)1 << candidate;
            candidate += (i << 6);
            feasible_result_count += check_vertex_candidate(current_state, check_solution);
        }
    }
    for (std::int64_t i = (size_in_dword << 3); i < vertex_candidates.size(); i++) {
        while (pstart_byte[i] > 0) {
            candidate = bit_vector::power_of_two(pstart_byte[i]);
            pstart_byte[i] ^= (1 << candidate);
            candidate += (i << 3);
            feasible_result_count += check_vertex_candidate(current_state, check_solution);
        }
    }
    return feasible_result_count;
}

bool matching_engine::check_vertex_candidate(const std::int64_t pattern_vertex,
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

bool matching_engine::check_vertex_candidate(state* current_state, bool check_solution) {
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

bool matching_engine::match_vertex(const std::int64_t pattern_vertex,
                                   const std::int64_t target_vertex) const {
    return pattern->get_vertex_degree(pattern_vertex) <= target->get_vertex_degree(target_vertex) &&
           pattern->get_vertex_attribute(pattern_vertex) ==
               target->get_vertex_attribute(target_vertex);
}

solution matching_engine::get_solution() {
    return std::move(engine_solutions);
}

void matching_engine::run_and_wait(global_stack& gstack,
                                   std::atomic<std::uint64_t>& busy_engine_count,
                                   bool main_engine) {
    if (main_engine) {
        first_states_generator(hlocal_stack);
    }
    bool is_busy_engine = true;
    ONEDAL_ASSERT(pattern != nullptr);
    if (target->bit_representation) { /* dense graph case */
        for (;;) {
            if (hlocal_stack.states_in_stack() > 0) {
                // while ((hlocal_stack.states_in_stack() > pattern->get_vertex_count()) &&
                //        gstack.push(hlocal_stack))
                //     ;
                while ((hlocal_stack.states_in_stack() > 5) && gstack.push(hlocal_stack))
                    ;
                ONEDAL_ASSERT(hlocal_stack.states_in_stack() > 0);
                state_exploration_bit();
            }
            else {
                gstack.pop(hlocal_stack);
                if (hlocal_stack.empty()) {
                    if (is_busy_engine) {
                        is_busy_engine = false;
                        --busy_engine_count;
                    }
                    if (busy_engine_count.load() == 0)
                        break;
                }
                else if (!is_busy_engine) {
                    is_busy_engine = true;
                    ++busy_engine_count;
                }
            }
        }
    }
    // while (hlocal_stack.states_in_stack() > 0) {
    //     state_exploration_bit();
    // }
    else { /* sparse graph case */
        for (;;) {
            if (hlocal_stack.states_in_stack() > 0) {
                // while ((hlocal_stack.states_in_stack() > pattern->get_vertex_count()) &&
                //        gstack.push(hlocal_stack))
                //     ;
                while ((hlocal_stack.states_in_stack() > 5) && gstack.push(hlocal_stack))
                    ;
                ONEDAL_ASSERT(hlocal_stack.states_in_stack() > 0);
                state_exploration_list();
            }
            else {
                gstack.pop(hlocal_stack);
                if (hlocal_stack.empty()) {
                    if (is_busy_engine) {
                        is_busy_engine = false;
                        --busy_engine_count;
                    }
                    if (busy_engine_count.load() == 0)
                        break;
                }
                else if (!is_busy_engine) {
                    is_busy_engine = true;
                    ++busy_engine_count;
                }
            }
        }
        // while (hlocal_stack.states_in_stack() > 0) {
        //     state_exploration_list();
        // }
    }
    return;
}

// solution matching_engine::run(bool main_engine) {
//     run_and_wait(main_engine);
//     return std::move(engine_solutions);
// }

engine_bundle::engine_bundle(const graph* ppattern,
                             const graph* ptarget,
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

engine_bundle::~engine_bundle() {
    pattern = nullptr;
    target = nullptr;
    sorted_pattern_vertex = nullptr;
    predecessor = nullptr;
    direction = nullptr;
    pconsistent_conditions = nullptr;
}

solution engine_bundle::run() {
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

    // const std::uint64_t array_size = max_threads_count * 2;
    const std::uint64_t array_size = max_threads_count;
    auto engine_array_ptr = allocator_.make_shared_memory<matching_engine>(array_size);
    matching_engine* engine_array = engine_array_ptr.get();

    for (std::uint64_t i = 0; i < array_size; ++i) {
        new (engine_array + i) matching_engine(pattern,
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

    global_stack gstack;
    std::atomic<std::uint64_t> busy_engine_count(array_size);
    dal::detail::threader_for(array_size, array_size, [&](const int index) {
        engine_array[index].run_and_wait(gstack, busy_engine_count, false);
    });

    for (std::uint64_t i = 0; i < array_size; i++) {
        bundle_solutions.add(engine_array[i].get_solution());
        engine_array[i].~matching_engine();
    }

    return std::move(bundle_solutions);
}

void engine_bundle::first_states_generator(bool use_exploration_stack) {
    if (use_exploration_stack) {
        bundle::ptr_t local_engine = matching_bundle.local();
        local_engine->first_states_generator(exploration_stack);
    }
    else {
        std::int64_t degree = pattern->get_vertex_degree(sorted_pattern_vertex[0]);
        dal::detail::threader_for(target->get_vertex_count(),
                                  target->get_vertex_count(),
                                  [=](const int i) {
                                      bundle::ptr_t local_engine = matching_bundle.local();
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
} // namespace oneapi::dal::preview::subgraph_isomorphism::detail
