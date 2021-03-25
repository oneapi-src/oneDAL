#pragma once

#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/graph.hpp"
#include "oneapi/dal/table/column_accessor.hpp"
#include "oneapi/dal/table/homogen.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

struct state {
    std::int64_t* core;
    std::int64_t core_length;

    state();
    state(std::int64_t length);
    state(state& parent_state, std::int64_t new_element);
    state(state* parent_state, std::int64_t new_element);
    ~state();
};

class solution {
public:
    solution();
    solution(const std::int64_t length, const std::int64_t* pattern_vertices);
    virtual ~solution();

    solution(solution&& sol);
    solution& operator=(solution&& sol);

    std::int64_t get_solution_count() const;
    graph_status add(std::int64_t** state_core);
    graph_status add(state& solution_state);
    graph_status add(state* solution_state);
    graph_status add(solution& _solution);
    graph_status add(solution&& _solution);
    oneapi::dal::homogen_table export_as_table();

private:
    std::int64_t** data;
    std::int64_t* sorted_pattern_vertices;

    std::int64_t solution_core_length;
    std::int64_t solution_count;
    std::int64_t max_solution_cout;

    graph_status increase_solutions_size();
    void delete_data();
};
} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
