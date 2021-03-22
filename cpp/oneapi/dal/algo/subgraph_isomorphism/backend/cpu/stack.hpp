#pragma once

#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/graph.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/solution.hpp"
#include <cstdint>
#include <new>

namespace dal_experimental {
#ifdef DUMP
/* Class for dumping search tree into graphviz (dot *.gv) format */
class tree_search_dumper {
public:
    tree_search_dumper(){};
    tree_search_dumper(const char* _graph_file_name);
    virtual ~tree_search_dumper();
    //void add_root(std::int64_t node_id) { root_id = node_id; };
    void add_pair(std::int64_t node_id_prev, std::int64_t node_id_next);
    void add_pair(std::int64_t level, std::int64_t node_id_prev, std::int64_t node_id_next);

private:
    std::ofstream graph_out_stream;
};
#endif // DUMP

class stack {
public:
    stack();
    stack(std::int64_t max_size);
    virtual ~stack();
    stack(stack&& _stack);
    stack& operator=(stack&& _stack);
    state* operator[](std::int64_t index);

    graph_status push(state* new_state);
    state* pop();
    std::int64_t size() const;
    void clear(bool direct = true);
    void clear_state(std::int64_t index);
    void add(stack& _stack);

#ifdef DEBUG_MODE
    void print_stack() const;
#endif //DEBUG_MODE

#ifdef DUMP
    tree_search_dumper** tree_search_dumper_array = nullptr;
    std::int64_t dumper_lenght = 0;
    void create_tree_search_dumper(std::int64_t vertex_count);
    void add_graph(std::int64_t root_id);
    void delete_tree_search_dumper();
#endif //DUMP

private:
    std::int64_t max_stack_size;
    std::int64_t stack_size;
    state** data;

    graph_status increase_stack_size();
    void delete_data();
};

class vertex_stack {
public:
    vertex_stack();
    vertex_stack(const std::uint64_t max_states_size);
    vertex_stack(const std::uint64_t max_states_size, const std::uint64_t* pdata);
    virtual ~vertex_stack();

    graph_status push(const std::uint64_t vertex_id);
    std::int64_t pop();
    bool delete_vertex();
    std::uint64_t size() const;
    std::uint64_t max_size() const;

private:
    std::uint64_t stack_size;
    std::uint64_t* stack_data;
    std::uint64_t* ptop;
    graph_status increase_stack_size();
    bool use_external_memory;

    friend class dfs_stack;
    friend class bfs_stack;
    friend class virtual_dfs_stack;
};

class dfs_stack {
public:
    dfs_stack();
    dfs_stack(const std::uint64_t levels);
    dfs_stack(const std::uint64_t levels, const std::uint64_t max_states_size);
    dfs_stack(const std::uint64_t levels, const std::uint64_t* max_states_size_per_level);
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

    state get_current_state() const;
    void fill_solution(std::int64_t* solution_core, const std::uint64_t last_vertex_id) const;

    void delete_current_state();

protected:
    std::uint64_t max_level_size;
    vertex_stack* data_by_levels;

    std::uint64_t current_level;

    void increase_core_level();
    void decrease_core_level();

private:
    void delete_data();
};

class virtual_dfs_stack : public dfs_stack {
public:
    virtual_dfs_stack();
    virtual_dfs_stack(const std::uint64_t levels,
                      const std::uint64_t _virtual_level_terminator,
                      std::uint64_t** pstate,
                      std::uint64_t last_vlevel_size,
                      const std::uint64_t* max_states_size_per_level);
    ~virtual_dfs_stack();

private:
    std::uint64_t virtual_level_terminator;
    void init(const std::uint64_t levels,
              const std::uint64_t _virtual_level_terminator,
              std::uint64_t** pstate,
              std::uint64_t last_vlevel_size,
              const std::uint64_t* max_states_size_per_level);

    void delete_data();

    friend class bfs_stack;
};

class bfs_stack {
public:
    bfs_stack();
    bfs_stack(const std::uint64_t levels, const std::uint64_t* _level_max_stack_size);
    ~bfs_stack();

    bool create_next_level();
    std::uint64_t get_level_width(const std::uint64_t level) const;

    void extract_state(const std::uint64_t level,
                       const std::uint64_t stack_index,
                       std::uint64_t** pstate);
    bool extract_virtual_dbs_stack(virtual_dfs_stack& pvdfs_stack,
                                   const std::uint64_t stack_index,
                                   std::uint64_t** _pstate);

#ifdef DEBUG_MODE
    float get_filling_level(const std::uint64_t level) const;
    float get_filling() const;
    std::uint64_t states_in_stack() const;
#endif // DEBUG_MODE

    void push(const std::uint64_t level,
              const std::uint64_t stack_index,
              const std::uint64_t vertex_id);

private:
    std::uint64_t max_level_size;
    const std::uint64_t* level_max_stack_size;
    std::uint64_t* level_stacks_count;
    vertex_stack** data_by_levels;

    std::uint64_t current_level;

    void init(const std::uint64_t levels, const std::uint64_t* _level_max_stack_size);
    bool init_level(const std::uint64_t level);
    void delete_data();

    std::uint64_t get_parent_vertex_id_by_level(const std::uint64_t child_level,
                                                const std::uint64_t child_stack_index,
                                                const std::uint64_t parent_level) const;
    std::uint64_t get_stack_index_by_vertex_index(const std::uint64_t level,
                                                  const std::uint64_t vertex_index) const;
    std::uint64_t stack_offset(const std::uint64_t level, const std::uint64_t stack_index) const;
};

} // namespace dal_experimental
