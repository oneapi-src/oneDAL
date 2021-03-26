#pragma once
#include "tbb/enumerable_thread_specific.h"

#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/sorter.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/solution.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/stack.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

enum flow_switch_ids {
    default_single_thread_mode = 0x0,
    multi_thread_mode = 0x1,
    use_hybrid_search = 0x2 /* dfs search - default value */
};

class matching_engine {
public:
    matching_engine(){};
    matching_engine(const graph* ppattern,
                    const graph* ptarget,
                    const std::int64_t* psorted_pattern_vertex,
                    const std::int64_t* ppredecessor,
                    const edge_direction* pdirection,
                    sconsistent_conditions const* pcconditions);
    matching_engine(const matching_engine& _matching_engine, stack& _local_stack);
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

private:
    const graph* pattern;
    const graph* target;
    const std::int64_t* sorted_pattern_vertex;
    const std::int64_t* predecessor;
    const edge_direction* direction;
    const sconsistent_conditions* pconsistent_conditions;

    std::int64_t solution_length;
    bit_vector vertex_candidates;

    std::int64_t temporary_list_size;
    std::int64_t* temporary_list;

    std::uint8_t* pstart_byte;
    std::int64_t candidate;

    stack local_stack;
    dfs_stack hlocal_stack;
    solution engine_solutions;

    std::int64_t extract_candidates(state* current_state, bool check_solution);
    bool check_vertex_candidate(state* current_state, bool check_solution);

    std::int64_t extract_candidates(bool check_solution);
    bool check_vertex_candidate(bool check_solution);

    friend class engine_bundle;
};

class engine_bundle {
public:
    stack exploration_stack;
    engine_bundle(const graph* ppattern,
                  const graph* ptarget,
                  const std::int64_t* psorted_pattern_vertex,
                  const std::int64_t* ppredecessor,
                  const edge_direction* pdirection,
                  sconsistent_conditions const* pcconditions,
                  float* ppattern_vertex_probability,
                  const std::uint64_t _control_flags);
    virtual ~engine_bundle();
    solution run();

private:
    const graph* pattern;
    const graph* target;
    const std::int64_t* sorted_pattern_vertex;
    const std::int64_t* predecessor;
    const edge_direction* direction;
    const sconsistent_conditions* pconsistent_conditions;
    const float* pattern_vertex_probability;
    std::uint64_t control_flags;

    solution bundle_solutions;

    typedef tbb::enumerable_thread_specific<matching_engine> bundle;
    bundle matching_bundle;
    void first_states_generator(bool use_exploration_stack = true);

    solution run_dfs();
    solution run_hybrid();
};
} // namespace oneapi::dal::preview::subgraph_isomorphism::backend