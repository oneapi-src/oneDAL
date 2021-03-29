#include "oneapi/dal/algo/subgraph_isomorphism/detail/matching.hpp"
#include "oneapi/dal/detail/threading.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

namespace dal = oneapi::dal;

matching_engine::~matching_engine() {
    pattern = nullptr;
    target = nullptr;
    sorted_pattern_vertex = nullptr;
    predecessor = nullptr;
    direction = nullptr;
    pconsistent_conditions = nullptr;

    _mm_free(temporary_list);
    temporary_list = nullptr;
    temporary_list_size = 0;
}

matching_engine::matching_engine(const graph* ppattern,
                                 const graph* ptarget,
                                 const std::int64_t* psorted_pattern_vertex,
                                 const std::int64_t* ppredecessor,
                                 const edge_direction* pdirection,
                                 sconsistent_conditions const* pcconditions) {
    pattern = ppattern;
    target = ptarget;
    sorted_pattern_vertex = psorted_pattern_vertex;
    predecessor = ppredecessor;
    direction = pdirection;
    pconsistent_conditions = pcconditions;

    solution_length = pattern->get_vertex_count();
    engine_solutions = solution(solution_length, psorted_pattern_vertex); //optimize initialization

    std::int64_t target_vertex_count = target->get_vertex_count();

    //Need modification for adj lists case support, ~30Mb for Kron-28 (256 * 10^6 vertices)
    vertex_candidates = bit_vector(bit_vector::bit_vector_size(target_vertex_count));

    pstart_byte = vertex_candidates.get_vector_pointer();
    candidate = 0;

    std::int64_t max_neighbours_size = target->get_max_degree();
    std::int64_t max_degree = target->get_max_degree();
    if (max_neighbours_size < max_degree) {
        max_neighbours_size = max_degree;
    }

#ifdef EXPERIMENTAL
    hlocal_stack.init(solution_length - 1, target_vertex_count);
#endif // EXPERIMENTAL

    if (target->bit_representation) {
        temporary_list = nullptr;
    }
    else {
        temporary_list_size = max_neighbours_size;
        temporary_list =
            static_cast<std::int64_t*>(_mm_malloc(sizeof(std::int64_t) * temporary_list_size, 64));
    }
}

matching_engine::matching_engine(const matching_engine& _matching_engine, stack& _local_stack)
        : matching_engine(_matching_engine.pattern,
                          _matching_engine.target,
                          _matching_engine.sorted_pattern_vertex,
                          _matching_engine.predecessor,
                          _matching_engine.direction,
                          _matching_engine.pconsistent_conditions) {
    local_stack = std::move(_local_stack);
}

std::int64_t matching_engine::state_exploration_bit(state* current_state, bool check_solution) {
    const std::int64_t i_cc = current_state->core_length - 1;
    const std::int64_t divider = pconsistent_conditions[i_cc].divider;

#pragma ivdep
    for (std::int64_t j = 0; j < divider; j++) {
        or_equal(vertex_candidates.get_vector_pointer(),
                 target->p_edges_bit[current_state->core[pconsistent_conditions[i_cc].array[j]]],
                 vertex_candidates.size());
    }

    ~vertex_candidates; // inversion?

#pragma ivdep
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

#pragma ivdep
    for (std::int64_t j = 0; j < divider; j++) {
        or_equal(vertex_candidates.get_vector_pointer(),
                 target->p_edges_bit[hlocal_stack.top(
                     pconsistent_conditions[current_level_index].array[j])],
                 vertex_candidates.size());
    }

    ~vertex_candidates; // inversion ?

#pragma ivdep
    for (std::int64_t j = current_level_index; j >= divider; j--) { //j > divider - 1
        and_equal(vertex_candidates.get_vector_pointer(),
                  target->p_edges_bit[hlocal_stack.top(
                      pconsistent_conditions[current_level_index].array[j])],
                  vertex_candidates.size());
    }

    for (std::int64_t i = 0; i <= current_level_index; i++) {
        vertex_candidates.get_vector_pointer()[bit_vector::byte(hlocal_stack.top(i))] &=
            ~bit_vector::bit(hlocal_stack.top(i));
    }
    return extract_candidates(check_solution);
}

std::int64_t matching_engine::extract_candidates(bool check_solution) {
    std::int64_t feasible_result_count = 0;

    std::uint64_t size_in_dword = vertex_candidates.size() >> 3;
    std::uint64_t* ptr;
    std::uint64_t popcnt;
    for (std::int64_t i = 0; i < size_in_dword; i++) {
        ptr = (std::uint64_t*)(pstart_byte + (i << 3));
        popcnt = _popcnt64(*ptr);
        for (std::int64_t j = 0; j < popcnt; j++) {
            candidate = 63 - _lzcnt_u64(*ptr);
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
    if (match_vertex(sorted_pattern_vertex[hlocal_stack.get_current_level()], candidate)) {
        if (check_solution && hlocal_stack.get_current_level() + 1 == solution_length) {
            std::int64_t* solution_core =
                static_cast<std::int64_t*>(_mm_malloc(sizeof(std::int64_t) * solution_length, 64));
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

#pragma ivdep
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

#pragma ivdep
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

#pragma ivdep
    for (std::int64_t j = 0; j < divider; j++) {
        or_equal(
            vertex_candidates.get_vector_pointer(),
            target->p_edges_list[hlocal_stack.top(
                pconsistent_conditions[current_level_index].array[j])],
            target
                ->p_degree[hlocal_stack.top(pconsistent_conditions[current_level_index].array[j])]);
    }

    ~vertex_candidates;

#pragma ivdep
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

    for (std::int64_t i = 0; i <= current_level_index; i++) {
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
    state null_state;
    std::int64_t candidates_count = 0;
    std::int64_t degree = pattern->get_vertex_degree(sorted_pattern_vertex[0]);
    for (std::int64_t i = 0; i < target->get_vertex_count(); i++) {
        if (degree <= target->get_vertex_degree(i) &&
            pattern->get_vertex_attribute(sorted_pattern_vertex[0]) ==
                target->get_vertex_attribute(i)) {
            void* place = _mm_malloc(sizeof(state), 64);
            state* new_state = new (place) state(&null_state, i);
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

    std::uint64_t size_in_dword = vertex_candidates.size() >> 3;
    std::uint64_t* ptr;
    std::uint64_t popcnt;
    for (std::int64_t i = 0; i < size_in_dword; i++) {
        ptr = (std::uint64_t*)(pstart_byte + (i << 3));
        popcnt = _popcnt64(*ptr);
        for (std::int64_t j = 0; j < popcnt; j++) {
            candidate = 63 - _lzcnt_u64(*ptr);
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
        state null_state;
        void* place = _mm_malloc(sizeof(state), 64);
        state* new_state = new (place) state(&null_state, target_vertex);
        local_stack.push(new_state); /* add new state into local_stack */
        return true;
    }
    return false;
}

bool matching_engine::check_vertex_candidate(state* current_state, bool check_solution) {
    if (match_vertex(sorted_pattern_vertex[current_state->core_length], candidate)) {
        void* place = _mm_malloc(sizeof(state), 64);
        state* new_state = new (place) state(current_state, candidate);

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

void matching_engine::run_and_wait(bool main_engine) {
#ifdef EXPERIMENTAL
    if (main_engine) {
        first_states_generator(hlocal_stack);
    }
    Lo if (target->bit_representation) { /* dense graph case */
        while (hlocal_stack.states_in_stack() > 0) {
            state_exploration_bit();
        }
    }
    else { /* sparse graph case */
        while (hlocal_stack.states_in_stack() > 0) {
            state_exploration_list();
        }
    }

#else
    if (main_engine) {
        first_states_generator(local_stack);
    }
    state* current_state = nullptr;

    if (target->bit_representation) { /* dense graph case */
        while (local_stack.size() > 0) {
            current_state = local_stack.pop();
            state_exploration_bit(current_state);
            /* for large sparse graph cases and internal threading */
            current_state->~state();
            _mm_free(current_state);
        }
    }
    else { /* sparse graph case */
        while (local_stack.size() > 0) {
            current_state = local_stack.pop();
            state_exploration_list(current_state);
            /* for large sparse graph cases and internal threading */
            current_state->~state();
            _mm_free(current_state);
        }
    }
#endif // EXPERIMENTAL
    return;
}

solution matching_engine::run(bool main_engine) {
    run_and_wait(main_engine);
    return std::move(engine_solutions);
}

engine_bundle::engine_bundle(const graph* ppattern,
                             const graph* ptarget,
                             const std::int64_t* psorted_pattern_vertex,
                             const std::int64_t* ppredecessor,
                             const edge_direction* pdirection,
                             sconsistent_conditions const* pcconditions,
                             float* ppattern_vertex_probability,
                             const std::uint64_t _control_flags) {
    pattern = ppattern;
    target = ptarget;
    sorted_pattern_vertex = psorted_pattern_vertex;
    predecessor = ppredecessor;
    direction = pdirection;
    pconsistent_conditions = pcconditions;
    pattern_vertex_probability = ppattern_vertex_probability;
    control_flags = _control_flags;

    bundle_solutions = solution(pattern->get_vertex_count(), psorted_pattern_vertex);
}

engine_bundle::~engine_bundle() {
    pattern = nullptr;
    target = nullptr;
    sorted_pattern_vertex = nullptr;
    predecessor = nullptr;
    direction = nullptr;
    pconsistent_conditions = nullptr;
}

solution engine_bundle::run_hybrid() {
    matching_bundle = bundle(pattern,
                             target,
                             sorted_pattern_vertex,
                             predecessor,
                             direction,
                             pconsistent_conditions);

    first_states_generator(true);
    std::int64_t steps = pattern->n; // pattern->n;
    for (std::int64_t level = 1; level < steps; level++) {
        for (bundle::iterator i = matching_bundle.begin(); i != matching_bundle.end(); ++i) {
            exploration_stack.add(i->local_stack);
        }

        // treading code
        dal::detail::threader_for(exploration_stack.size(),
                                  exploration_stack.size(),
                                  [=](const int i) {
                                      bundle::reference local_engine = matching_bundle.local();
                                      local_engine.state_exploration_bit(exploration_stack[i],
                                                                         false);
                                      exploration_stack.clear_state(i);
                                  });
        exploration_stack.clear(false);
    }

    if (steps != pattern->n) {
        // task_group tg; TBB
        for (bundle::iterator i = matching_bundle.begin(); i != matching_bundle.end(); ++i) {
            // tg.run([=] {
            bundle_solutions.add(i->run(false));
            // });
        }
        // tg.wait();
    }
    else {
        for (bundle::iterator i = matching_bundle.begin(); i != matching_bundle.end(); ++i) {
            std::int64_t solution_size = i->local_stack.size();
            for (std::int64_t j = 0; j < solution_size; j++) {
                bundle_solutions.add(i->local_stack.pop());
            }
        }
    }

    matching_bundle.clear();
    return std::move(bundle_solutions);
}

solution engine_bundle::run_dfs() {
    std::int64_t degree = pattern->get_vertex_degree(sorted_pattern_vertex[0]);

    std::uint64_t first_states_count = pattern_vertex_probability[0] * target->get_vertex_count();
    int max_threads_count = dal::detail::threader_get_max_threads();
    std::uint64_t possible_first_states_count_per_thread = first_states_count / max_threads_count;
    if (possible_first_states_count_per_thread < 1) {
        max_threads_count = first_states_count;
        possible_first_states_count_per_thread = 1;
    }
    else {
        possible_first_states_count_per_thread +=
            static_cast<bool>(first_states_count % max_threads_count);
    }

    matching_engine* engine_array =
        static_cast<matching_engine*>(operator new[](sizeof(matching_engine) * max_threads_count));

    for (int i = 0; i < max_threads_count; ++i) {
        new (engine_array + i) matching_engine(pattern,
                                               target,
                                               sorted_pattern_vertex,
                                               predecessor,
                                               direction,
                                               pconsistent_conditions);
    }

    state null_state;
    // task_group tg;
    std::uint64_t task_counter = 0;
    for (std::int64_t i = 0; i < target->n; ++i) {
        if (degree <= target->get_vertex_degree(i) &&
            pattern->get_vertex_attribute(sorted_pattern_vertex[0]) ==
                target->get_vertex_attribute(i)) {
#ifdef EXPERIMENTAL
            engine_array[task_counter].push_into_stack(i);

            if ((engine_array[task_counter].hlocal_stack.states_in_stack() /
                 possible_first_states_count_per_thread) > 0 ||
                i == target->n - 1) {
                // tg.run([=] {
                engine_array[task_counter].run_and_wait(false);
                // });
                task_counter++;
            }
#else
            void* place = _mm_malloc(sizeof(state), 64);
            state* new_state = new (place) state(&null_state, i);
            engine_array[task_counter].push_into_stack(new_state);

            if ((engine_array[task_counter].local_stack.size() /
                 possible_first_states_count_per_thread) > 0 ||
                i == target->n - 1) {
                // tg.run([=] {
                engine_array[task_counter].run_and_wait(false);
                // });
                task_counter++;
            }
#endif // EXPERIMENTAL
        }
    }

    // tg.wait();

    for (int i = 0; i < max_threads_count; i++) {
        bundle_solutions.add(engine_array[i].get_solution());
        engine_array[i].~matching_engine();
    }
    operator delete[](engine_array);
    engine_array = nullptr;

    return std::move(bundle_solutions);
}

solution engine_bundle::run() {
    if (control_flags & flow_switch_ids::use_hybrid_search) {
        return run_hybrid();
    }

    return run_dfs();
}

void engine_bundle::first_states_generator(bool use_exploration_stack) {
    if (use_exploration_stack) {
        bundle::reference local_engine = matching_bundle.local();
        local_engine.first_states_generator(exploration_stack);
    }
    else {
        std::int64_t degree = pattern->get_vertex_degree(sorted_pattern_vertex[0]);
        dal::detail::threader_for(target->get_vertex_count(),
                                  target->get_vertex_count(),
                                  [=](const int i) {
                                      bundle::reference local_engine = matching_bundle.local();
                                      state null_state;
                                      if (degree <= target->get_vertex_degree(i) &&
                                          pattern->get_vertex_attribute(sorted_pattern_vertex[0]) ==
                                              target->get_vertex_attribute(i)) {
                                          void* place = _mm_malloc(sizeof(state), 64);
                                          state* new_state = new (place) state(&null_state, i);
                                          local_engine.local_stack.push(new_state);
                                      }
                                  });
    }
}
} // namespace oneapi::dal::preview::subgraph_isomorphism::detail