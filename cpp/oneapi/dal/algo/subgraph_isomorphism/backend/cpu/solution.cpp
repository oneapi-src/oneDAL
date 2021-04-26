#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/solution.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/common.hpp"
#include <iostream>

namespace dal = oneapi::dal;
namespace oneapi::dal::preview::subgraph_isomorphism::detail {

state::state(inner_alloc a) : allocator_(a) {
    core = nullptr;
    core_length = 0;
}

state::state(std::int64_t length, inner_alloc a) : allocator_(a) {
    core_length = length;
    core = allocator_.allocate<std::int64_t>(core_length);
}

state::state(state& parent_state, std::int64_t new_element, inner_alloc a)
        : state(parent_state.core_length + 1, a) {
    for (std::int64_t i = 0; i < parent_state.core_length; i++) {
        /* TODO replace by memset */
        core[i] = parent_state.core[i];
    }
    core[parent_state.core_length] = new_element;
}

state::state(state* parent_state, std::int64_t new_element, inner_alloc a)
        : state(parent_state->core_length + 1, a) {
    for (std::int64_t i = 0; i < parent_state->core_length; i++) {
        /* TODO replace by memset */
        core[i] = parent_state->core[i];
    }
    core[parent_state->core_length] = new_element;
}

state::~state() {
    allocator_.deallocate<std::int64_t>(core, core_length);
    core = nullptr;
    core_length = 0;
}

solution::solution(inner_alloc a) : allocator_(a) {
    solution_count = 0;
    max_solution_cout = 100;
    solution_core_length = 0;
    data = allocator_.allocate<std::int64_t*>(max_solution_cout);
    for (std::int64_t i = 0; i < max_solution_cout; i++) {
        data[i] = nullptr;
    }

    sorted_pattern_vertices = nullptr;
}

solution::solution(const std::int64_t length, const std::int64_t* pattern_vertices, inner_alloc a)
        : solution(a) {
    solution_core_length = length;
    if (pattern_vertices != nullptr) {
        sorted_pattern_vertices = allocator_.allocate<std::int64_t>(solution_core_length);
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
                allocator_.deallocate<std::int64_t>(data[i], 0);
                data[i] = nullptr;
            }
        }
        allocator_.deallocate<std::int64_t*>(data, max_solution_cout);
        data = nullptr;
    }
    if (sorted_pattern_vertices != nullptr) {
        allocator_.deallocate<std::int64_t>(sorted_pattern_vertices, solution_core_length);
        sorted_pattern_vertices = nullptr;
    }
}

solution::solution(solution&& sol)
        : data(sol.data),
          sorted_pattern_vertices(sol.sorted_pattern_vertices),
          allocator_(sol.allocator_) {
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
            allocator_.deallocate<state>(solution_state, 0);
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
        allocator_.deallocate<std::int64_t*>(_solution.data, 0);
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
    std::int64_t** tmp_data = allocator_.allocate<std::int64_t*>(2 * max_solution_cout);
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
    if (data != nullptr) {
        allocator_.deallocate<std::int64_t*>(data, max_solution_cout);
    }
    max_solution_cout *= 2;
    data = tmp_data;
    tmp_data = nullptr;
    return ok;
}

template <typename T>
inline T min(const T a, const T b) {
    return (a >= b) ? b : a;
}

oneapi::dal::homogen_table solution::export_as_table() {
    std::cout << "[solution::export_as_table] Function started" << std::endl;
    if (solution_count == 0)
        return dal::homogen_table();

    const auto begin = sorted_pattern_vertices;
    const auto end = &sorted_pattern_vertices[solution_core_length];

    std::cout << "[solution::export_as_table] Mapping creation started" << std::endl;
    auto mapping_array = dal::array<std::int64_t>::empty(solution_core_length);
    const auto mapping = mapping_array.get_mutable_data();
    for (std::int64_t j = 0; j < solution_core_length; ++j) {
        const auto p = std::find(begin, end, j);
        ONEDAL_ASSERT(p != end, "Index not found");
        mapping[j] = p - begin;
    }
    std::cout << "[solution::export_as_table] Mapping creation finished" << std::endl;

    auto arr_solution = dal::array<int>::empty(solution_core_length * solution_count);
    const auto arr = arr_solution.get_mutable_data();

    std::cout << "[solution::export_as_table] Data copying started" << std::endl;
    constexpr std::int64_t block_size = 64;
    const std::int64_t block_count = (solution_count - 1 + block_size) % block_size;
    dal::detail::threader_for(block_count, block_count, [&](int index) {
        const std::int64_t first = index * block_size;
        const std::int64_t last = min(first + block_size, solution_count);
        for (auto i = first; i != last; ++i) {
            for (std::int64_t j = 0; j < solution_core_length; ++j) {
                arr[i * solution_core_length + j] = data[i][mapping[j]];
            }
        }
    });
    std::cout << "[solution::export_as_table] Data copying finished" << std::endl;
    std::cout << "[solution::export_as_table] Function finished" << std::endl;

    return dal::detail::homogen_table_builder{}
        .reset(arr_solution, solution_count, solution_core_length)
        .build();
}
} // namespace oneapi::dal::preview::subgraph_isomorphism::detail
