#pragma once

#include <memory>

#include "oneapi/dal/algo/subgraph_isomorphism/detail/sorter.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/detail/solution.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/detail/stack.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/common.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

enum flow_switch_ids {
    default_single_thread_mode = 0x0,
    multi_thread_mode = 0x1,
    use_hybrid_search = 0x2 /* dfs search - default value */
};

class matching_engine {
public:
    matching_engine(inner_alloc allocator)
            : _allocator(allocator),
              vertex_candidates(allocator.get_byte_allocator()),
              local_stack(allocator),
              hlocal_stack(allocator),
              engine_solutions(allocator){};
    matching_engine(const graph* ppattern,
                    const graph* ptarget,
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

private:
    inner_alloc _allocator;
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

    kind isomorphism_kind_;

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
                  const std::uint64_t _control_flags,
                  kind isomorphism_kind,
                  inner_alloc allocator);
    virtual ~engine_bundle();
    solution run();

private:
    inner_alloc _allocator;
    const graph* pattern;
    const graph* target;
    const std::int64_t* sorted_pattern_vertex;
    const std::int64_t* predecessor;
    const edge_direction* direction;
    const sconsistent_conditions* pconsistent_conditions;
    const float* pattern_vertex_probability;
    std::uint64_t control_flags;
    kind isomorphism_kind_;

    solution bundle_solutions;

    typedef oneapi::dal::detail::tls_mem<matching_engine, std::allocator<double>> bundle;
    bundle matching_bundle;
    void first_states_generator(bool use_exploration_stack = true);
};
} // namespace oneapi::dal::preview::subgraph_isomorphism::detail
