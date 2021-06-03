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

#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/graph_status.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/inner_alloc.hpp"
#include "oneapi/dal/table/column_accessor.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/table/homogen.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

template <typename Cpu>
struct state {
    std::int64_t* core;
    std::int64_t core_length;

    state(inner_alloc a);
    state(std::int64_t length, inner_alloc a);
    state(state<Cpu>& parent_state, std::int64_t new_element, inner_alloc a);
    state(state<Cpu>* parent_state, std::int64_t new_element, inner_alloc a);
    void clear();
    ~state();

private:
    inner_alloc allocator_;
};

template <typename Cpu>
class solution {
public:
    solution(inner_alloc a);
    solution(const std::int64_t length, const std::int64_t* pattern_vertices, inner_alloc a);
    virtual ~solution();

    solution(solution<Cpu>&& sol);
    solution<Cpu>& operator=(solution<Cpu>&& sol);

    std::int64_t get_solution_count() const;
    graph_status add(std::int64_t** state_core);
    graph_status add(state<Cpu>& solution_state);
    graph_status add(state<Cpu>* solution_state);
    graph_status add(solution<Cpu>& _solution);
    graph_status add(solution<Cpu>&& _solution);
    oneapi::dal::homogen_table export_as_table();

private:
    inner_alloc allocator_;

public:
    std::int64_t** data;
    std::int64_t* sorted_pattern_vertices;

    std::int64_t solution_core_length;
    std::int64_t solution_count;
    std::int64_t max_solution_cout;

    graph_status increase_solutions_size();
    void delete_data();
};

template <typename Cpu>
state<Cpu>::state(inner_alloc a) : allocator_(a) {
    core = nullptr;
    core_length = 0;
}

template <typename Cpu>
state<Cpu>::state(std::int64_t length, inner_alloc a) : allocator_(a) {
    core_length = length;
    core = allocator_.allocate<std::int64_t>(core_length);
}

template <typename Cpu>
state<Cpu>::state(state<Cpu>& parent_state, std::int64_t new_element, inner_alloc a)
        : state(parent_state.core_length + 1, a) {
    for (std::int64_t i = 0; i < parent_state.core_length; i++) {
        /* TODO replace by memset */
        core[i] = parent_state.core[i];
    }
    core[parent_state.core_length] = new_element;
}

template <typename Cpu>
state<Cpu>::state(state<Cpu>* parent_state, std::int64_t new_element, inner_alloc a)
        : state(parent_state->core_length + 1, a) {
    for (std::int64_t i = 0; i < parent_state->core_length; i++) {
        /* TODO replace by memset */
        core[i] = parent_state->core[i];
    }
    core[parent_state->core_length] = new_element;
}

template <typename Cpu>
void state<Cpu>::clear() {
    allocator_.deallocate<std::int64_t>(core, core_length);
    core = nullptr;
    core_length = 0;
}

template <typename Cpu>
state<Cpu>::~state() {
    this->clear();
}

template <typename Cpu>
solution<Cpu>::solution(inner_alloc a) : allocator_(a) {
    solution_count = 0;
    max_solution_cout = 100;
    solution_core_length = 0;
    data = allocator_.allocate<std::int64_t*>(max_solution_cout);
    for (std::int64_t i = 0; i < max_solution_cout; i++) {
        data[i] = nullptr;
    }

    sorted_pattern_vertices = nullptr;
}

template <typename Cpu>
solution<Cpu>::solution(const std::int64_t length,
                        const std::int64_t* pattern_vertices,
                        inner_alloc a)
        : solution<Cpu>(a) {
    solution_core_length = length;
    if (pattern_vertices != nullptr) {
        sorted_pattern_vertices = allocator_.allocate<std::int64_t>(solution_core_length);
        for (std::int64_t i = 0; i < solution_core_length; i++) {
            sorted_pattern_vertices[i] = pattern_vertices[i]; // replace memset
        }
    }
}

template <typename Cpu>
solution<Cpu>::~solution() {
    delete_data();
}

template <typename Cpu>
void solution<Cpu>::delete_data() {
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

template <typename Cpu>
solution<Cpu>::solution(solution<Cpu>&& sol)
        : allocator_(sol.allocator_),
          data(sol.data),
          sorted_pattern_vertices(sol.sorted_pattern_vertices) {
    max_solution_cout = sol.max_solution_cout;
    solution_count = sol.solution_count;
    solution_core_length = sol.solution_core_length;
    sol.data = nullptr;
    sol.sorted_pattern_vertices = nullptr;
    sol.solution_count = 0;
    sol.solution_core_length = 0;
}

template <typename Cpu>
solution<Cpu>& solution<Cpu>::operator=(solution<Cpu>&& sol) {
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

template <typename Cpu>
std::int64_t solution<Cpu>::get_solution_count() const {
    return solution_count;
}

template <typename Cpu>
graph_status solution<Cpu>::add(std::int64_t** state_core) {
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

template <typename Cpu>
graph_status solution<Cpu>::add(state<Cpu>& solution_state) {
    graph_status status = add(&solution_state.core);
    solution_state.~state();
    return status;
}

template <typename Cpu>
graph_status solution<Cpu>::add(state<Cpu>* solution_state) {
    if (solution_state != nullptr) {
        graph_status status = add(&solution_state->core);
        solution_state->~state();
        if (solution_state != nullptr) {
            allocator_.deallocate<state<Cpu>>(solution_state, 0);
        }
        solution_state = nullptr;
        return status;
    }
    else {
        return bad_arguments;
    }
}

template <typename Cpu>
graph_status solution<Cpu>::add(solution<Cpu>& _solution) {
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

template <typename Cpu>
graph_status solution<Cpu>::add(solution<Cpu>&& _solution) {
    return add(_solution);
}

template <typename Cpu>
graph_status solution<Cpu>::increase_solutions_size() {
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

template <typename Cpu>
oneapi::dal::homogen_table solution<Cpu>::export_as_table() {
    if (solution_count == 0)
        return dal::homogen_table();

    const auto begin = sorted_pattern_vertices;
    const auto end = &sorted_pattern_vertices[solution_core_length];

    auto mapping_array = dal::array<std::int64_t>::empty(solution_core_length);
    const auto mapping = mapping_array.get_mutable_data();
    for (std::int64_t j = 0; j < solution_core_length; ++j) {
        const auto p = std::find(begin, end, j);
        ONEDAL_ASSERT(p != end, "Index not found");
        mapping[j] = p - begin;
    }

    auto arr_solution = dal::array<int>::empty(solution_core_length * solution_count);
    const auto arr = arr_solution.get_mutable_data();

    constexpr std::int64_t block_size = 64;
    const std::int64_t block_count = (solution_count - 1 + block_size) / block_size;
    dal::detail::threader_for(block_count, block_count, [&](int index) {
        const std::int64_t first = index * block_size;
        const std::int64_t last = min(first + block_size, solution_count);
        for (auto i = first; i != last; ++i) {
            for (std::int64_t j = 0; j < solution_core_length; ++j) {
                arr[i * solution_core_length + j] = data[i][mapping[j]];
            }
        }
    });

    return dal::detail::homogen_table_builder{}
        .reset(arr_solution, solution_count, solution_core_length)
        .build();
}
} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
