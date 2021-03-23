#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/stack.hpp"

#ifdef DEBUG_MODE
#include <iostream>
#endif //DEBUG_MODE

using namespace dal_experimental;

stack::stack() {
    max_stack_size = 100;
    stack_size = 0;
    data = static_cast<state**>(_mm_malloc(sizeof(state*) * max_stack_size, 64));
    for (std::int64_t i = 0; i < max_stack_size; i++) {
        data[i] = nullptr;
    }
}

stack::stack(std::int64_t max_size) {
    max_stack_size = max_size;
    stack_size = 0;
    data = static_cast<state**>(_mm_malloc(sizeof(state*) * max_stack_size, 64));
    for (std::int64_t i = 0; i < max_stack_size; i++) {
        data[i] = nullptr;
    }
}

void stack::delete_data() {
    if (data != nullptr) {
        for (std::int64_t i = 0; i < max_stack_size; i++) {
            if (data[i] != nullptr) {
                data[i]->~state();
                _mm_free(data[i]);
                data[i] = nullptr;
            }
        }
        _mm_free(data);
        data = nullptr;
    }
}

void stack::clear(bool direct) {
    for (std::int64_t i = 0; i < stack_size * direct; i++) {
        if (data[i] != nullptr) {
            data[i]->~state();
            _mm_free(data[i]);
            data[i] = nullptr;
        }
    }
    stack_size = 0;
}

void stack::clear_state(std::int64_t index) {
    data[index]->~state();
    _mm_free(data[index]);
    data[index] = nullptr;
}

void stack::add(stack& _stack) {
    std::int64_t current_size = _stack.size();
    for (std::int64_t i = 0; i < current_size; i++) {
        push(_stack.pop());
    }
    _stack.clear();
}

stack::stack(stack&& _stack) : data(_stack.data) {
    max_stack_size = _stack.max_stack_size;
    stack_size = _stack.stack_size;
    _stack.data = nullptr;
    _stack.max_stack_size = 0;
    _stack.stack_size = 0;
}

stack& stack::operator=(stack&& _stack) {
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

stack::~stack() {
    if (data != nullptr) {
        for (std::int64_t i = 0; i < max_stack_size; i++) {
            if (data[i] != nullptr) {
                data[i]->~state();
                _mm_free(data[i]);
                data[i] = nullptr;
            }
        }
        _mm_free(data);
        data = nullptr;
    }
    stack_size = 0;

#ifdef DEBUG_MODE
#ifdef DUMP
    delete_tree_search_dumper();
#endif //DUMP
#endif //DEBUG_MODE
}

#ifdef DEBUG_MODE
#ifdef DUMP

void stack::create_tree_search_dumper(std::int64_t vertex_count) {
    dumper_lenght = vertex_count;
    tree_search_dumper_array = static_cast<tree_search_dumper**>(
        _mm_malloc(sizeof(tree_search_dumper*) * dumper_lenght, 64));
    for (std::int64_t i = 0; i < dumper_lenght; i++) {
        tree_search_dumper_array[i] = nullptr;
    }
}

void stack::delete_tree_search_dumper() {
    if (tree_search_dumper_array != nullptr) {
        for (std::int64_t i = 0; i < dumper_lenght; i++) {
            if (tree_search_dumper_array[i] != nullptr) {
                delete tree_search_dumper_array[i];
                tree_search_dumper_array[i] = nullptr;
            }
        }
        dumper_lenght = 0;
    }
}

void stack::add_graph(std::int64_t root_id) {
    if (tree_search_dumper_array[root_id] == nullptr) {
        std::string str = std::to_string(root_id);
        tree_search_dumper_array[root_id] = new tree_search_dumper(str.c_str());
    }
}
#endif //DUMP
#endif //DEBUG_MODE

graph_status stack::push(state* new_state) {
    if (new_state != nullptr) {
        if (stack_size >= max_stack_size) {
            graph_status increase_status = increase_stack_size();
            if (increase_status != ok) {
                return increase_status;
            }
        }
        data[stack_size] = new_state;
        stack_size++;

#ifdef DEBUG_MODE
#ifdef DUMP
        if (new_state->core_length == 1) {
            add_graph(new_state->core[0]);
        }
        else {
            tree_search_dumper_array[new_state->core[0]]->add_pair(
                new_state->core_length,
                new_state->core[new_state->core_length - 2],
                new_state->core[new_state->core_length - 1]);
        }
#endif // DUMP
#endif //DEBUG_MODE
    }
    return ok;
}

state* stack::pop() {
    state* pstate = nullptr;
    if (stack_size > 0) {
        stack_size--;
        pstate = data[stack_size];
        data[stack_size] = nullptr;
    }
    return pstate;
}

state* stack::operator[](std::int64_t index) {
    return data[index];
}

std::int64_t stack::size() const {
    return stack_size;
}

graph_status stack::increase_stack_size() {
    state** tmp_data = static_cast<state**>(_mm_malloc(sizeof(state*) * 2 * max_stack_size, 64));
    if (tmp_data == nullptr) {
        return bad_allocation;
    }
    for (std::int64_t i = 0; i < max_stack_size; i++) {
        tmp_data[i] = data[i];
        data[i] = nullptr;
    }
    for (std::int64_t i = max_stack_size; i < 2 * max_stack_size; i++) {
        tmp_data[i] = nullptr;
    }
    max_stack_size *= 2;
    _mm_free(data);
    data = tmp_data;
    tmp_data = nullptr;
    return ok;
}

#ifdef DEBUG_MODE
void stack::print_stack() const {
    std::cout << "------------------ stack information --------------------" << std::endl;
    std::cout << "    maximum stack size: " << max_stack_size << std::endl;
    std::cout << "    stack size: " << stack_size << std::endl;
    std::cout << "        stack data:" << std::endl;

    for (std::int64_t i = 0; i < stack_size; i++) {
        std::cout << "          [" << i << "] core length:" << data[i]->core_length << std::endl;
        for (std::int64_t j = 0; j < data[i]->core_length; j++) {
            std::cout << "            core data[" << j << "]:" << data[i]->core[j] << std::endl;
        }
    }
}

#ifdef DUMP
tree_search_dumper::tree_search_dumper(const char* _graph_file_name) {
    std::string file_name(_graph_file_name);
    file_name = "dump\\" + file_name + ".gv";
    graph_out_stream = std::ofstream(file_name);
    graph_out_stream << "digraph " << _graph_file_name << " {\n";
    graph_out_stream << "l1_" << _graph_file_name << ";\n";
}

tree_search_dumper::~tree_search_dumper() {
    graph_out_stream << "}\n";
    graph_out_stream.close();
}

void tree_search_dumper::add_pair(std::int64_t node_id_prev, std::int64_t node_id_next) {
    graph_out_stream << node_id_prev << "->" << node_id_next << "\n";
}

void tree_search_dumper::add_pair(std::int64_t level,
                                  std::int64_t node_id_prev,
                                  std::int64_t node_id_next) {
    std::string node_prev = std::string("l") + std::to_string(level - 1);
    std::string node_current = std::string("l") + std::to_string(level);

    node_prev += "_" + std::to_string(node_id_prev);
    node_current += "_" + std::to_string(node_id_next);
    graph_out_stream << node_prev << "->" << node_current << "\n";
}

#endif //DUMP
#endif //DEBUG_MODE

vertex_stack::vertex_stack() {
    stack_size = 0;
    stack_data = nullptr;
    ptop = nullptr;
    use_external_memory = false;
}

vertex_stack::vertex_stack(const std::uint64_t max_states_size) {
    use_external_memory = false;
    stack_size = max_states_size;
    stack_data = static_cast<std::uint64_t*>(_mm_malloc(sizeof(std::uint64_t) * stack_size, 64));
    ptop = stack_data;
}

vertex_stack::vertex_stack(const std::uint64_t max_states_size, const std::uint64_t* pdata) {
    use_external_memory = true;
    stack_size = max_states_size;
    //stack_data = pdata;
    //ptop = stack_data;
}

vertex_stack::~vertex_stack() {
    _mm_free(stack_data);
    stack_data = nullptr;
    ptop = nullptr;
    stack_size = 0;
}

graph_status vertex_stack::push(const std::uint64_t vertex_id) {
    //if (size() >= stack_size) {
    //    if (increase_stack_size() != ok) {
    //        return bad_allocation;
    //    }
    //}
    *ptop = vertex_id;
    ptop++;
    return ok;
}

std::int64_t vertex_stack::pop() {
    if (ptop != nullptr && ptop != stack_data) {
        ptop--;
        return *ptop;
    }
    return -1;
}

bool vertex_stack::delete_vertex() {
    ptop -= (ptop != nullptr) && (ptop != stack_data);
    return !(ptop - stack_data);
}

std::uint64_t vertex_stack::size() const {
    return ptop - stack_data;
}

std::uint64_t vertex_stack::max_size() const {
    return stack_size;
}

graph_status vertex_stack::increase_stack_size() {
    std::uint64_t* tmp_data =
        static_cast<std::uint64_t*>(_mm_malloc(sizeof(std::uint64_t) * 2 * stack_size, 64));
    if (tmp_data == nullptr) {
        return bad_allocation;
    }
    for (std::int64_t i = 0; i < stack_size; i++) {
        tmp_data[i] = stack_data[i];
        //tmp_data[i + stack_size] = null_node;
    }
    stack_size *= 2;
    ptop = size() + tmp_data;
    _mm_free(stack_data);
    stack_data = tmp_data;
    tmp_data = nullptr;
    return ok;
}

dfs_stack::dfs_stack() {
    max_level_size = 0;
    data_by_levels = nullptr;

    current_level = 0;
}

dfs_stack::dfs_stack(const std::uint64_t levels) {
    init(levels);
}

dfs_stack::dfs_stack(const std::uint64_t levels, const std::uint64_t max_states_size) {
    init(levels, max_states_size);
}

dfs_stack::dfs_stack(const std::uint64_t levels, const std::uint64_t* max_states_size_per_level) {
    init(levels, max_states_size_per_level);
}

void dfs_stack::init(const std::uint64_t levels) {
    //if (data_by_levels != nullptr) {
    //    delete_data();
    //}

    max_level_size = levels;
    current_level = 0;

    data_by_levels =
        static_cast<vertex_stack*>(operator new[](sizeof(vertex_stack) * max_level_size));
}

void dfs_stack::init(const std::uint64_t levels, const std::uint64_t max_states_size) {
    init(levels);

    for (std::int64_t i = 0; i < max_level_size; ++i) {
        new (data_by_levels + i) vertex_stack(max_states_size);
    }
}

void dfs_stack::init(const std::uint64_t levels, const std::uint64_t* max_states_size_per_level) {
    init(levels);

    for (std::int64_t i = 0; i < max_level_size; ++i) {
        new (data_by_levels + i) vertex_stack(max_states_size_per_level[i]);
    }
}

void dfs_stack::delete_data() {
    for (std::int64_t i = 0; i < max_level_size; i++) {
        data_by_levels[i].~vertex_stack();
    }
    operator delete[](data_by_levels);
    data_by_levels = nullptr;

    max_level_size = 0;
    current_level = 0;
}

dfs_stack::~dfs_stack() {
    delete_data();
}

std::uint64_t dfs_stack::get_current_level() const {
    return current_level + 1;
}

std::uint64_t dfs_stack::get_current_level_fill_size() const {
    return data_by_levels[current_level].size();
}

void dfs_stack::push_into_current_level(const std::uint64_t vertex_id) {
    data_by_levels[current_level].push(vertex_id);
}

std::uint64_t dfs_stack::get_current_level_index() const {
    return current_level;
}

void dfs_stack::push_into_next_level(const std::uint64_t vertex_id) {
    //current_level++;
    data_by_levels[current_level + 1].push(vertex_id);
}

void dfs_stack::increase_core_level() {
    current_level++;
}

void dfs_stack::decrease_core_level() {
    current_level--;
}

void dfs_stack::update() {
    std::uint64_t new_level = current_level + 1;
    if (new_level < max_level_size && size(new_level) > 0) {
        current_level++;
    }
    else {
        delete_current_state();
    }
}

state dfs_stack::get_current_state() const {
    state result(current_level + 1);
    for (std::int64_t i = 0; i < result.core_length; i++) {
        result.core[i] = *(data_by_levels[i].ptop - 1);
    }
    return result;
}

void dfs_stack::fill_solution(std::int64_t* solution_core,
                              const std::uint64_t last_vertex_id) const {
    for (std::int64_t i = 0; i <= current_level; i++) {
        solution_core[i] = *(data_by_levels[i].ptop - 1);
    }
    solution_core[current_level + 1] = last_vertex_id;
}

std::uint64_t dfs_stack::top(const std::uint64_t level) const {
    //if (data_by_levels[level].size()) {
    return *(data_by_levels[level].ptop - 1);
    //}
    //return null_node;
}

void dfs_stack::delete_current_state() {
    while (data_by_levels[current_level].delete_vertex() && current_level > 0) {
        current_level--;
    }
}

std::uint64_t dfs_stack::states_in_stack() const {
    std::int64_t size = 0;
    for (std::int64_t i = 0; i <= current_level; i++) {
        size += data_by_levels[i].size();
    }
    return size - current_level;
}

std::uint64_t dfs_stack::size(const std::uint64_t level) const {
    return data_by_levels[level].size();
}

std::uint64_t dfs_stack::max_level_width(const std::uint64_t level) const {
    return data_by_levels[level].max_size();
}

virtual_dfs_stack::virtual_dfs_stack() : dfs_stack() {
    virtual_level_terminator = 0;
}

void virtual_dfs_stack::init(const std::uint64_t levels,
                             const std::uint64_t _virtual_level_terminator,
                             std::uint64_t** pstate,
                             std::uint64_t last_vlevel_size,
                             const std::uint64_t* max_states_size_per_level) {
    if (data_by_levels != nullptr) {
        delete_data();
    }

    max_level_size = levels;
    virtual_level_terminator = _virtual_level_terminator;

    current_level = virtual_level_terminator;
    data_by_levels =
        static_cast<vertex_stack*>(operator new[](sizeof(vertex_stack) * max_level_size));

    for (std::uint64_t i = 0; i < virtual_level_terminator; i++) {
        new (data_by_levels + i) vertex_stack();
        data_by_levels[i].stack_size = 1;
        data_by_levels[i].stack_data = pstate[i];
        data_by_levels[i].ptop = pstate[i] + 1; //???
    }

    new (data_by_levels + virtual_level_terminator) vertex_stack();
    data_by_levels[virtual_level_terminator].stack_size = last_vlevel_size;
    data_by_levels[virtual_level_terminator].stack_data = pstate[virtual_level_terminator];
    data_by_levels[virtual_level_terminator].ptop =
        pstate[virtual_level_terminator] + last_vlevel_size;

    for (std::uint64_t i = virtual_level_terminator + 1; i < max_level_size; i++) {
        new (data_by_levels + i) vertex_stack(max_states_size_per_level[i]);
    }
}

virtual_dfs_stack::virtual_dfs_stack(const std::uint64_t levels,
                                     const std::uint64_t _virtual_level_terminator,
                                     std::uint64_t** pstate,
                                     std::uint64_t last_vlevel_size,
                                     const std::uint64_t* max_states_size_per_level) {
    init(levels, _virtual_level_terminator, pstate, last_vlevel_size, max_states_size_per_level);
}

virtual_dfs_stack::~virtual_dfs_stack() {
    delete_data();
}

void virtual_dfs_stack::delete_data() {
    for (std::int64_t i = 0; i <= virtual_level_terminator; i++) {
        data_by_levels[i].stack_data = nullptr;
        data_by_levels[i].stack_size = 0;
        data_by_levels[i].ptop = nullptr;
    }

    for (std::int64_t i = virtual_level_terminator + 1; i < max_level_size; i++) {
        data_by_levels[i].~vertex_stack();
    }
    operator delete[](data_by_levels);
    data_by_levels = nullptr;

    virtual_level_terminator = 0;
    max_level_size = 0;
    current_level = 0;
}

bfs_stack::bfs_stack() {
    max_level_size = 0;
    level_stacks_count = nullptr;
    data_by_levels = nullptr;

    current_level = 0;
}

bfs_stack::bfs_stack(const std::uint64_t levels, const std::uint64_t* _level_max_stack_size)
        : bfs_stack() {
    init(levels, _level_max_stack_size);
}

bfs_stack::~bfs_stack() {
    delete_data();
}

void bfs_stack::delete_data() {
    if (data_by_levels != nullptr) {
        for (std::int64_t i = 0; i < max_level_size; i++) {
            if (data_by_levels[i] != nullptr) {
                for (std::int64_t j = 0; j < level_stacks_count[i]; j++) {
                    data_by_levels[i][j].~vertex_stack();
                }
                operator delete[](data_by_levels[i]);
                data_by_levels[i] = nullptr;
            }
        }
        operator delete[](data_by_levels);
        data_by_levels = nullptr;
    }

    level_max_stack_size = nullptr;

    if (level_stacks_count != nullptr) {
        delete[] level_stacks_count;
        level_stacks_count = nullptr;
    }

    max_level_size = 0;
    current_level = 0;
}

void bfs_stack::init(const std::uint64_t levels, const std::uint64_t* _level_max_stack_size) {
    if (data_by_levels != nullptr) {
        delete_data();
    }

    max_level_size = levels;
    level_max_stack_size = _level_max_stack_size;
    level_stacks_count = new std::uint64_t[max_level_size];
    level_stacks_count[0] = 1;

    data_by_levels =
        static_cast<vertex_stack**>(operator new[](sizeof(vertex_stack*) * max_level_size));
    if (data_by_levels == nullptr) {
        return;
    }

    current_level = 0;
    init_level(current_level);
}

std::uint64_t bfs_stack::get_level_width(const std::uint64_t level) const {
    std::uint64_t width = 0;
    for (std::uint64_t i = 0; i < level_stacks_count[level]; i++) {
        width += data_by_levels[level][i].size();
    }
    return width;
}

#ifdef DEBUG_MODE
float bfs_stack::get_filling_level(const std::uint64_t level) const {
    return (float)get_level_width(level) /
           (float)(level_max_stack_size[level] * level_stacks_count[level]);
}

float bfs_stack::get_filling() const {
    std::uint64_t level_width_sum = 0;
    std::uint64_t total_length = 0;
    for (std::uint64_t i = 0; i <= current_level; i++) {
        level_width_sum += get_level_width(i);
        total_length += level_max_stack_size[i] * level_stacks_count[i];
    }
    return (float)level_width_sum / (float)total_length;
}

std::uint64_t bfs_stack::states_in_stack() const {
    return get_level_width(current_level);
}
#endif // DEBUG_MODE

bool bfs_stack::init_level(const std::uint64_t level) {
    if (level >= max_level_size)
        return false;

    if (level != 0) {
        std::uint64_t prev_level_width = get_level_width(level - 1);
        level_stacks_count[level] = prev_level_width;
        data_by_levels[level] =
            static_cast<vertex_stack*>(operator new[](sizeof(vertex_stack) * prev_level_width));
        if (data_by_levels[level] != nullptr) {
            for (std::uint64_t i = 0; i < prev_level_width; i++) {
                try {
                    new (data_by_levels[level] + i) vertex_stack(level_max_stack_size[level]);
                }
                catch (std::bad_alloc&) {
                    operator delete[](data_by_levels[level]);
                    data_by_levels[level] = nullptr;
                    return false;
                }
            }
        }
        else {
            return false;
        }
    }
    else {
        level_stacks_count[0] = 1;
        try {
            data_by_levels[0] = new vertex_stack(level_max_stack_size[0]);
            //new (data_by_levels) vertex_stack(level_max_stack_size[0]);
            for (std::uint64_t i = 1; i < max_level_size; i++) {
                data_by_levels[i] = nullptr;
            }
        }
        catch (std::bad_alloc&) {
            return false;
        }
    }
    return true;
}

bool bfs_stack::create_next_level() {
    if (init_level(current_level + 1)) {
        current_level++;
        return true;
    }
    return false;
}

void bfs_stack::push(const std::uint64_t level,
                     const std::uint64_t stack_index,
                     const std::uint64_t vertex_id) {
    if (level < max_level_size && stack_index < level_stacks_count[level]) {
        data_by_levels[level][stack_index].push(vertex_id);
    }
}

std::int64_t bfs_stack::get_parent_vertex_id_by_level(const std::uint64_t child_level,
                                                      const std::uint64_t child_stack_index,
                                                      const std::uint64_t parent_level) const {
    // performance critical method, disable check correctness
    if (child_level < max_level_size && parent_level < child_level &&
        child_stack_index < level_stacks_count[child_level]) {
        std::uint64_t stack_index = child_stack_index;
        for (std::uint64_t i = child_level - 1; i > parent_level; i--) {
            stack_index = get_stack_index_by_vertex_index(i, stack_index);
        }
        //
    }
    return null_node;
}

std::uint64_t bfs_stack::get_stack_index_by_vertex_index(const std::uint64_t level,
                                                         const std::uint64_t vertex_index) const {
    std::int64_t stack_itr = -1;
    std::uint64_t vertex_counter = 0;
    while (vertex_counter <= vertex_index) {
        stack_itr++;
        vertex_counter += data_by_levels[level][stack_itr].size();
    }

    return stack_itr;
}

std::uint64_t bfs_stack::stack_offset(const std::uint64_t level,
                                      const std::uint64_t stack_index) const {
    std::uint64_t offset = 0;
    for (std::uint64_t i = 0; i < stack_index; i++) {
        offset += data_by_levels[level][i].size();
    }
    return offset;
}

bool bfs_stack::extract_virtual_dbs_stack(virtual_dfs_stack& pvdfs_stack,
                                          const std::uint64_t stack_index,
                                          std::uint64_t** _pstate) {
    //std::uint64_t* _pstate[max_level_size];

    extract_state(current_level, stack_index, _pstate);
    pvdfs_stack.init(max_level_size,
                     current_level,
                     _pstate,
                     data_by_levels[current_level][stack_index].size(),
                     level_max_stack_size);
    return false;
}

void bfs_stack::extract_state(const std::uint64_t level,
                              const std::uint64_t stack_index,
                              std::uint64_t** pstate) {
    pstate[level] = data_by_levels[level][stack_index].stack_data;
    std::uint64_t vertex_index;
    std::uint64_t current_stack_index = stack_index;
    for (std::int64_t i = level - 1; i >= 0; i--) {
        vertex_index = current_stack_index;
        current_stack_index = get_stack_index_by_vertex_index(i, current_stack_index);
        vertex_index -= stack_offset(i, current_stack_index);
        pstate[i] = data_by_levels[i][current_stack_index].stack_data + vertex_index;
    }
}