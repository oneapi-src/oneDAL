#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/solution.hpp"
#include <cassert>

namespace dal = oneapi::dal;
using namespace dal_experimental;

state::state() {
    core = nullptr;
    core_length = 0;
}

state::state(std::int64_t length) {
    core_length = length;
    core = static_cast<std::int64_t*>(_mm_malloc(sizeof(std::int64_t) * core_length, 64));
}

state::state(state& parent_state, std::int64_t new_element) : state(parent_state.core_length + 1) {
    for (std::int64_t i = 0; i < parent_state.core_length; i++) {
        /* TODO replace by memset */
        core[i] = parent_state.core[i];
    }
    core[parent_state.core_length] = new_element;
}

state::state(state* parent_state, std::int64_t new_element) : state(parent_state->core_length + 1) {
    for (std::int64_t i = 0; i < parent_state->core_length; i++) {
        /* TODO replace by memset */
        core[i] = parent_state->core[i];
    }
    core[parent_state->core_length] = new_element;
}

state::~state() {
    _mm_free(core);
    core = nullptr;
    core_length = 0;
}

solution::solution() {
    solution_count = 0;
    max_solution_cout = 100;
    solution_core_length = 0;
    data = static_cast<std::int64_t**>(_mm_malloc(sizeof(std::int64_t*) * max_solution_cout, 64));
    for (std::int64_t i = 0; i < max_solution_cout; i++) {
        data[i] = nullptr;
    }

    sorted_pattern_vertices = nullptr;
}

solution::solution(const std::int64_t length, const std::int64_t* pattern_vertices) : solution() {
    solution_core_length = length;
    if (pattern_vertices != nullptr) {
        sorted_pattern_vertices =
            static_cast<std::int64_t*>(_mm_malloc(sizeof(std::int64_t) * solution_core_length, 64));
        for (std::int64_t i = 0; i < solution_core_length; i++) {
            sorted_pattern_vertices[i] = pattern_vertices[i]; // replace memset
        }
    }
}

solution::~solution() {
    delete_data();
}

void solution::delete_data() {
    if (data != nullptr) {
        for (std::int64_t i = 0; i < max_solution_cout; i++) {
            if (data[i] != nullptr) {
                _mm_free(data[i]);
                data[i] = nullptr;
            }
        }
        _mm_free(data);
        data = nullptr;
    }
    if (sorted_pattern_vertices != nullptr) {
        _mm_free(sorted_pattern_vertices);
        sorted_pattern_vertices = nullptr;
    }
}

solution::solution(solution&& sol)
        : data(sol.data),
          sorted_pattern_vertices(sol.sorted_pattern_vertices) {
    max_solution_cout = sol.max_solution_cout;
    solution_count = sol.solution_count;
    solution_core_length = sol.solution_core_length;
    sol.data = nullptr;
    sol.sorted_pattern_vertices = nullptr;
    sol.solution_count = 0;
    sol.solution_core_length = 0;
}

solution& solution::operator=(solution&& sol) {
    if (&sol == this) {
        return *this;
    }
    delete_data();
    solution_count = sol.solution_count;
    max_solution_cout = sol.max_solution_cout;
    solution_core_length = sol.solution_core_length;
    data = sol.data;
    sol.data = nullptr;
    sorted_pattern_vertices = sol.sorted_pattern_vertices;
    sol.sorted_pattern_vertices = nullptr;
    sol.solution_count = 0;
    return *this;
}

std::int64_t solution::get_solution_count() const {
    return solution_count;
}

graph_status solution::add(std::int64_t** state_core) {
    if (state_core != nullptr && *state_core != nullptr) {
        if (solution_count >= max_solution_cout) {
            graph_status increase_status = increase_solutions_size();
            if (increase_status != ok) {
                return increase_status;
            }
        }
        data[solution_count] = *state_core;
        *state_core = nullptr;
        solution_count++;
    }
    return ok;
}

graph_status solution::add(state& solution_state) {
    graph_status status = add(&solution_state.core);
    solution_state.~state();
    return status;
}

graph_status solution::add(state* solution_state) {
    if (solution_state != nullptr) {
        graph_status status = add(&solution_state->core);
        solution_state->~state();
        if (solution_state != nullptr) {
            _mm_free(solution_state);
        }
        solution_state = nullptr;
        return status;
    }
    else {
        return bad_arguments;
    }
}

graph_status solution::add(solution& _solution) {
    graph_status status = ok;
    for (std::int64_t i = 0; i < _solution.get_solution_count(); i++) {
        status = add(&_solution.data[i]);
    }

    if (_solution.get_solution_count() > 0) {
        solution_core_length = _solution.solution_core_length;
        sorted_pattern_vertices = _solution.sorted_pattern_vertices;
    }

    if (_solution.data != nullptr) {
        _mm_free(_solution.data);
        _solution.data = nullptr;
    }

    _solution.sorted_pattern_vertices = nullptr;
    _solution.solution_count = 0;
    _solution.solution_core_length = 0;

    return status;
}

graph_status solution::add(solution&& _solution) {
    return add(_solution);
}

graph_status solution::increase_solutions_size() {
    std::int64_t** tmp_data =
        static_cast<std::int64_t**>(_mm_malloc(sizeof(std::int64_t*) * 2 * max_solution_cout, 64));
    if (tmp_data == nullptr) {
        return bad_allocation;
    }
    for (std::int64_t i = 0; i < max_solution_cout; i++) {
        tmp_data[i] = data[i];
        data[i] = nullptr;
    }
    for (std::int64_t i = max_solution_cout; i < 2 * max_solution_cout; i++) {
        tmp_data[i] = nullptr;
    }
    max_solution_cout *= 2;
    if (data != nullptr) {
        _mm_free(data);
    }
    data = tmp_data;
    tmp_data = nullptr;
    return ok;
}

#ifdef DEBUG_MODE
void solution::print_solutions(bool in_sorted_view) const {
    std::cout << "--- Subgraph Isomorphism Solutions ---" << std::endl;
    std::cout << "    Solution count: " << solution_count << std::endl;
    std::cout << "    pattern size: " << solution_core_length << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "    Bijection results: Patter vertex -> Target vertex" << std::endl;

    std::int64_t sorted_index;
    for (std::int64_t i = 0; i < solution_core_length; i++) {
        if (in_sorted_view) {
            std::cout << "[" << std::setw(3) << i << "]      " << std::setw(7)
                      << sorted_pattern_vertices[i] << " | -> | ";
        }
        else {
            std::cout << "[" << std::setw(3) << i << "]      " << std::setw(7) << i << " | -> | ";
        }

        sorted_index = i;
        if (!in_sorted_view) {
            for (std::int64_t k = 0; k < solution_core_length; k++) {
                if (sorted_pattern_vertices[k] == i) {
                    sorted_index = k;
                    break;
                }
            }
        }

        for (std::int64_t j = 0; j < solution_count; j++) {
            std::cout << std::setw(3) << data[j][sorted_index] << " | ";
        }
        std::cout << std::endl;
    }
    std::cout << "--------------------------------------" << std::endl;
}

void solution::print_solutions_csv() const {
    auto begin = sorted_pattern_vertices;
    auto end = &sorted_pattern_vertices[solution_core_length];
    for (std::int64_t i = 0; i < solution_count; i++) {
        for (std::int64_t j = 0; j < solution_core_length; j++) {
            auto p_j1 = std::find(begin, end, j);
            assert(p_j1 != end && "Index not found");
            std::cout << data[i][p_j1 - sorted_pattern_vertices];
            if (solution_core_length > 0 && j != (solution_core_length - 1)) {
                std::cout << ",";
            }
        }
        std::cout << std::endl;
    }
}
#endif // DEBUG_MODE

oneapi::dal::homogen_table solution::export_as_table() {
    if (solution_count == 0)
        return dal::homogen_table();

    auto begin = sorted_pattern_vertices;
    auto end = &sorted_pattern_vertices[solution_core_length];

    auto arr_solution = dal::array<int>::empty(solution_core_length * solution_count);
    int* arr = arr_solution.get_mutable_data();

    for (std::int64_t i = 0; i < solution_count; i++) {
        for (std::int64_t j = 0; j < solution_core_length; j++) {
            auto p_j1 = std::find(begin, end, j);
            assert(p_j1 != end && "Index not found");
            arr[i * solution_core_length + j] = data[i][p_j1 - begin];
        }
    }
    return dal::detail::homogen_table_builder{}
        .reset(arr_solution, solution_count, solution_core_length)
        .build();
}