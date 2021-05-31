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
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/solution.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

template <typename Cpu>
class stack {
public:
    stack(inner_alloc allocator);
    stack(std::int64_t max_size, inner_alloc allocator);
    virtual ~stack();
    stack(stack<Cpu>&& _stack);
    stack<Cpu>& operator=(stack<Cpu>&& _stack);
    state<Cpu>* operator[](std::int64_t index);

    graph_status push(state<Cpu>* new_state);
    state<Cpu>* pop();
    std::int64_t size() const;
    void clear(bool direct = true);
    void clear_state(std::int64_t index);
    void add(stack<Cpu>& _stack);

private:
    inner_alloc allocator_;
    std::int64_t max_stack_size;
    std::int64_t stack_size;
    state<Cpu>** data;

    graph_status increase_stack_size();
    void delete_data();
};

template <typename Cpu>
class vertex_stack {
public:
    vertex_stack(inner_alloc allocator);
    vertex_stack(const std::uint64_t max_states_size, inner_alloc allocator);
    virtual ~vertex_stack();
    void clean();

    graph_status push(const std::uint64_t vertex_id);
    std::int64_t pop();
    bool delete_vertex();
    std::uint64_t size() const;
    std::uint64_t max_size() const;

    inner_alloc allocator_;
    std::uint64_t stack_size;
    std::uint64_t* stack_data;
    std::uint64_t* ptop;
    graph_status increase_stack_size();
    bool use_external_memory;
};

template <typename Cpu>
class dfs_stack {
public:
    dfs_stack(inner_alloc allocator);
    dfs_stack(const std::uint64_t levels, inner_alloc allocator);
    dfs_stack(const std::uint64_t levels,
              const std::uint64_t max_states_size,
              inner_alloc allocator);
    dfs_stack(const std::uint64_t levels,
              const std::uint64_t* max_states_size_per_level,
              inner_alloc allocator);
    virtual ~dfs_stack();
    void init(const std::uint64_t levels);
    void init(const std::uint64_t levels, const std::uint64_t max_states_size);
    void init(const std::uint64_t levels, const std::uint64_t* max_states_size_per_level);

    void push_into_current_level(const std::uint64_t vertex_id);
    void push_into_next_level(const std::uint64_t vertex_id);
    void update();

    std::uint64_t states_in_stack() const;
    std::uint64_t size(const std::uint64_t level) const;
    std::uint64_t max_level_width(const std::uint64_t level) const;

    std::uint64_t get_current_level() const;
    std::uint64_t get_current_level_index() const;
    std::uint64_t get_current_level_fill_size() const;
    std::uint64_t top(const std::uint64_t level) const;

    state<Cpu> get_current_state() const;
    void fill_solution(std::int64_t* solution_core, const std::uint64_t last_vertex_id) const;

    void delete_current_state();

protected:
    inner_alloc allocator_;
    std::uint64_t max_level_size;
    vertex_stack<Cpu>* data_by_levels;

    std::uint64_t current_level;

    void increase_core_level();
    void decrease_core_level();

private:
    void delete_data();
};

template <typename Cpu>
stack<Cpu>::stack(inner_alloc allocator) : allocator_(allocator) {
    max_stack_size = 100;
    stack_size = 0;
    data = allocator_.allocate<state<Cpu>*>(max_stack_size);
    for (std::int64_t i = 0; i < max_stack_size; i++) {
        data[i] = nullptr;
    }
}

template <typename Cpu>
stack<Cpu>::stack(std::int64_t max_size, inner_alloc allocator) : allocator_(allocator) {
    max_stack_size = max_size;
    stack_size = 0;
    data = allocator_.allocate<state<Cpu>*>(max_stack_size);
    for (std::int64_t i = 0; i < max_stack_size; i++) {
        data[i] = nullptr;
    }
}

template <typename Cpu>
void stack<Cpu>::delete_data() {
    if (data != nullptr) {
        for (std::int64_t i = 0; i < max_stack_size; i++) {
            if (data[i] != nullptr) {
                data[i]->clear();
                allocator_.deallocate<state<Cpu>>(data[i], 0);
                data[i] = nullptr;
            }
        }
        allocator_.deallocate<state<Cpu>*>(data, max_stack_size);
        data = nullptr;
    }
}

template <typename Cpu>
void stack<Cpu>::clear(bool direct) {
    for (std::int64_t i = 0; i < stack_size * direct; i++) {
        if (data[i] != nullptr) {
            data[i]->clear();
            allocator_.deallocate<state<Cpu>>(data[i], 0);
            data[i] = nullptr;
        }
    }
    stack_size = 0;
}

template <typename Cpu>
void stack<Cpu>::clear_state(std::int64_t index) {
    data[index]->clear();
    allocator_.deallocate<state<Cpu>>(data[index], 0);
    data[index] = nullptr;
}

template <typename Cpu>
void stack<Cpu>::add(stack<Cpu>& _stack) {
    std::int64_t current_size = _stack.size();
    for (std::int64_t i = 0; i < current_size; i++) {
        push(_stack.pop());
    }
    _stack.clear();
}

template <typename Cpu>
stack<Cpu>::stack(stack<Cpu>&& _stack) : allocator_(_stack.allocator_),
                                         data(_stack.data) {
    max_stack_size = _stack.max_stack_size;
    stack_size = _stack.stack_size;
    _stack.data = nullptr;
    _stack.max_stack_size = 0;
    _stack.stack_size = 0;
}

template <typename Cpu>
stack<Cpu>& stack<Cpu>::operator=(stack<Cpu>&& _stack) {
    if (&_stack == this) {
        return *this;
    }
    delete_data();
    max_stack_size = _stack.max_stack_size;
    stack_size = _stack.stack_size;
    data = _stack.data;
    _stack.data = nullptr;
    _stack.max_stack_size = 0;
    _stack.stack_size = 0;
    return *this;
}

template <typename Cpu>
stack<Cpu>::~stack() {
    if (data != nullptr) {
        for (std::int64_t i = 0; i < max_stack_size; i++) {
            if (data[i] != nullptr) {
                this->data[i]->clear();
                allocator_.deallocate<state<Cpu>>(data[i], 0);
                data[i] = nullptr;
            }
        }
        allocator_.deallocate<state<Cpu>*>(data, max_stack_size);
        data = nullptr;
    }
    stack_size = 0;
}

template <typename Cpu>
graph_status stack<Cpu>::push(state<Cpu>* new_state) {
    if (new_state != nullptr) {
        if (max_stack_size == 0 || stack_size >= max_stack_size) {
            graph_status increase_status = increase_stack_size();
            if (increase_status != ok) {
                return increase_status;
            }
        }
        data[stack_size] = new_state;
        stack_size++;
    }
    return ok;
}

template <typename Cpu>
state<Cpu>* stack<Cpu>::pop() {
    state<Cpu>* pstate = nullptr;
    if (stack_size > 0) {
        stack_size--;
        pstate = data[stack_size];
        data[stack_size] = nullptr;
    }
    return pstate;
}

template <typename Cpu>
state<Cpu>* stack<Cpu>::operator[](std::int64_t index) {
    return data[index];
}

template <typename Cpu>
std::int64_t stack<Cpu>::size() const {
    return stack_size;
}

template <typename Cpu>
graph_status stack<Cpu>::increase_stack_size() {
    const auto new_max_stack_size = (max_stack_size > 0) ? 2 * max_stack_size : 100;
    state<Cpu>** tmp_data = allocator_.allocate<state<Cpu>*>(new_max_stack_size);
    if (tmp_data == nullptr) {
        throw oneapi::dal::host_bad_alloc();
    }
    for (std::int64_t i = 0; i < max_stack_size; i++) {
        tmp_data[i] = data[i];
        data[i] = nullptr;
    }
    for (std::int64_t i = max_stack_size; i < new_max_stack_size; i++) {
        tmp_data[i] = nullptr;
    }
    allocator_.deallocate<state<Cpu>*>(data, max_stack_size);
    max_stack_size = new_max_stack_size;
    data = tmp_data;
    tmp_data = nullptr;
    return ok;
}

template <typename Cpu>
vertex_stack<Cpu>::vertex_stack(inner_alloc allocator) : allocator_(allocator) {
    stack_size = 0;
    stack_data = nullptr;
    ptop = nullptr;
    use_external_memory = false;
}

template <typename Cpu>
vertex_stack<Cpu>::vertex_stack(const std::uint64_t max_states_size, inner_alloc allocator)
        : allocator_(allocator) {
    use_external_memory = false;
    stack_size = max_states_size;
    stack_data = allocator_.allocate<std::uint64_t>(stack_size);
    ptop = stack_data;
}

template <typename Cpu>
void vertex_stack<Cpu>::clean() {
    allocator_.deallocate<std::uint64_t>(stack_data, stack_size);
    stack_data = nullptr;
    ptop = nullptr;
    stack_size = 0;
}

template <typename Cpu>
vertex_stack<Cpu>::~vertex_stack() {
    this->clean();
}

template <typename Cpu>
graph_status vertex_stack<Cpu>::push(const std::uint64_t vertex_id) {
    if (size() >= stack_size) {
        if (increase_stack_size() != ok) {
            throw dal::host_bad_alloc();
        }
    }
    *ptop = vertex_id;
    ptop++;
    return ok;
}

template <typename Cpu>
std::int64_t vertex_stack<Cpu>::pop() {
    if (ptop != nullptr && ptop != stack_data) {
        ptop--;
        return *ptop;
    }
    return -1;
}

template <typename Cpu>
bool vertex_stack<Cpu>::delete_vertex() {
    ptop -= (ptop != nullptr) && (ptop != stack_data);
    return !(ptop - stack_data);
}

template <typename Cpu>
std::uint64_t vertex_stack<Cpu>::size() const {
    return ptop - stack_data;
}

template <typename Cpu>
std::uint64_t vertex_stack<Cpu>::max_size() const {
    return stack_size;
}

template <typename Cpu>
graph_status vertex_stack<Cpu>::increase_stack_size() {
    std::uint64_t* tmp_data = allocator_.allocate<std::uint64_t>(2 * stack_size);
    if (tmp_data == nullptr) {
        return bad_allocation;
    }
    for (std::uint64_t i = 0; i < stack_size; i++) {
        tmp_data[i] = stack_data[i];
    }
    allocator_.deallocate<std::uint64_t>(stack_data, stack_size);
    stack_size *= 2;
    ptop = size() + tmp_data;
    stack_data = tmp_data;
    tmp_data = nullptr;
    return ok;
}

template <typename Cpu>
dfs_stack<Cpu>::dfs_stack(inner_alloc allocator) : allocator_(allocator) {
    max_level_size = 0;
    data_by_levels = nullptr;

    current_level = 0;
}

template <typename Cpu>
dfs_stack<Cpu>::dfs_stack(const std::uint64_t levels, inner_alloc allocator)
        : allocator_(allocator) {
    init(levels);
}

template <typename Cpu>
dfs_stack<Cpu>::dfs_stack(const std::uint64_t levels,
                          const std::uint64_t max_states_size,
                          inner_alloc allocator)
        : allocator_(allocator) {
    init(levels, max_states_size);
}

template <typename Cpu>
dfs_stack<Cpu>::dfs_stack(const std::uint64_t levels,
                          const std::uint64_t* max_states_size_per_level,
                          inner_alloc allocator)
        : allocator_(allocator) {
    init(levels, max_states_size_per_level);
}

template <typename Cpu>
void dfs_stack<Cpu>::init(const std::uint64_t levels) {
    max_level_size = levels;
    current_level = 0;
    data_by_levels = allocator_.allocate<vertex_stack<Cpu>>(max_level_size);
}

template <typename Cpu>
void dfs_stack<Cpu>::init(const std::uint64_t levels, const std::uint64_t max_states_size) {
    init(levels);

    for (std::uint64_t i = 0; i < max_level_size; ++i) {
        new (data_by_levels + i) vertex_stack<Cpu>(max_states_size, allocator_);
    }
}

template <typename Cpu>
void dfs_stack<Cpu>::init(const std::uint64_t levels,
                          const std::uint64_t* max_states_size_per_level) {
    init(levels);

    for (std::uint64_t i = 0; i < max_level_size; ++i) {
        new (data_by_levels + i) vertex_stack<Cpu>(max_states_size_per_level[i], allocator_);
    }
}

template <typename Cpu>
void dfs_stack<Cpu>::delete_data() {
    for (std::uint64_t i = 0; i < max_level_size; i++) {
        data_by_levels[i].clean();
    }
    allocator_.deallocate<vertex_stack<Cpu>>(data_by_levels, max_level_size);
    data_by_levels = nullptr;

    max_level_size = 0;
    current_level = 0;
}

template <typename Cpu>
dfs_stack<Cpu>::~dfs_stack() {
    delete_data();
}

template <typename Cpu>
std::uint64_t dfs_stack<Cpu>::get_current_level() const {
    return current_level + 1;
}

template <typename Cpu>
std::uint64_t dfs_stack<Cpu>::get_current_level_fill_size() const {
    return data_by_levels[current_level].size();
}

template <typename Cpu>
void dfs_stack<Cpu>::push_into_current_level(const std::uint64_t vertex_id) {
    data_by_levels[current_level].push(vertex_id);
}

template <typename Cpu>
std::uint64_t dfs_stack<Cpu>::get_current_level_index() const {
    return current_level;
}

template <typename Cpu>
void dfs_stack<Cpu>::push_into_next_level(const std::uint64_t vertex_id) {
    data_by_levels[current_level + 1].push(vertex_id);
}

template <typename Cpu>
void dfs_stack<Cpu>::increase_core_level() {
    current_level++;
}

template <typename Cpu>
void dfs_stack<Cpu>::decrease_core_level() {
    current_level--;
}

template <typename Cpu>
void dfs_stack<Cpu>::update() {
    std::uint64_t new_level = current_level + 1;
    if (new_level < max_level_size && size(new_level) > 0) {
        current_level++;
    }
    else {
        delete_current_state();
    }
}

template <typename Cpu>
state<Cpu> dfs_stack<Cpu>::get_current_state() const {
    state<Cpu> result(current_level + 1, allocator_);
    for (std::int64_t i = 0; i < result.core_length; i++) {
        result.core[i] = *(data_by_levels[i].ptop - 1);
    }
    return result;
}

template <typename Cpu>
void dfs_stack<Cpu>::fill_solution(std::int64_t* solution_core,
                                   const std::uint64_t last_vertex_id) const {
    for (std::uint64_t i = 0; i <= current_level; i++) {
        solution_core[i] = *(data_by_levels[i].ptop - 1);
    }
    solution_core[current_level + 1] = last_vertex_id;
}

template <typename Cpu>
std::uint64_t dfs_stack<Cpu>::top(const std::uint64_t level) const {
    return *(data_by_levels[level].ptop - 1);
}

template <typename Cpu>
void dfs_stack<Cpu>::delete_current_state() {
    while (data_by_levels[current_level].delete_vertex() && current_level > 0) {
        current_level--;
    }
}

template <typename Cpu>
std::uint64_t dfs_stack<Cpu>::states_in_stack() const {
    std::int64_t size = 0;
    for (std::uint64_t i = 0; i <= current_level; i++) {
        size += data_by_levels[i].size();
    }
    return size - current_level;
}

template <typename Cpu>
std::uint64_t dfs_stack<Cpu>::size(const std::uint64_t level) const {
    return data_by_levels[level].size();
}

template <typename Cpu>
std::uint64_t dfs_stack<Cpu>::max_level_width(const std::uint64_t level) const {
    return data_by_levels[level].max_size();
}

} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
