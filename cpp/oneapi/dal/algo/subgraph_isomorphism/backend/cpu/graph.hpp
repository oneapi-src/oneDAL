#pragma once

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/gcc_adapt.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/inner_alloc.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/graph_status.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/bit_vector.hpp"
// #include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/matching.hpp"
// #include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/sorter.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

enum graph_storage_scheme { auto_detect, bit, list };

struct graph_input_data {
    std::int64_t vertex_count;
    std::int64_t* degree;
    std::int64_t* attr;
    std::int64_t** edges_attribute;

    graph_input_data(inner_alloc allocator);
    graph_input_data(const std::int64_t vertex_size, inner_alloc allocator);
    ~graph_input_data();

    inner_alloc allocator_;
};

struct graph_input_list_data : public graph_input_data {
    std::int64_t** data;

    graph_input_list_data(inner_alloc allocator);
    graph_input_list_data(const std::int64_t vertex_size, inner_alloc allocator);
    graph_input_list_data(graph_input_data* input_data, inner_alloc allocator);
    ~graph_input_list_data();
};

struct graph_input_bit_data : public graph_input_data {
    std::uint8_t** data;

    graph_input_bit_data(inner_alloc allocator);
    graph_input_bit_data(const std::int64_t vertex_size, inner_alloc allocator);
    graph_input_bit_data(graph_input_data* input_data, inner_alloc allocator);
    ~graph_input_bit_data();
};

struct graph_data {
    const graph_input_bit_data* pbit_data;
    const graph_input_list_data* plist_data;

    graph_data();
    graph_data(const graph_input_bit_data* pbit, const graph_input_list_data* plist);
    ~graph_data();
};

const std::int64_t null_node = 0xffffffffffffffff; /*!< Null node value*/
const double graph_storage_divider_by_density =
    0.015625; // 1/64 for memory capacity and ~0.005 for cpu.

template <typename Cpu>
class matching_engine;

template <typename Cpu>
class engine_bundle;

class sorter;

class graph {
public:
    graph(const dal::preview::detail::topology<std::int32_t>& t,
          graph_storage_scheme storage_scheme,
          byte_alloc_iface* byte_alloc);
    virtual ~graph();

    static double graph_density(const std::int64_t vertex_count, const std::int64_t edge_count);

    graph_status load_vertex_attribute(const std::int64_t vertex_count,
                                       const std::int64_t* pvertices_attribute);
    graph_status load_edge_attribute_lists(const std::int64_t vertex_count,
                                           std::int64_t const* const* p_edges_attribute_list);

    std::int64_t get_vertex_count() const;

    void set_edge(const std::int64_t current_vertex, const std::int64_t vertex);
    edge_direction check_edge(const std::int64_t current_vertex, const std::int64_t vertex) const;
    bool has_edge(const std::int64_t vertex_first, const std::int64_t vertex_second) const;

    bit_vector get_vertex_neighbors(const std::int64_t vertex,
                                    const edge_direction edge_type) const;

    graph_status get_edge_vertices_ids(const std::int64_t current_vertex,
                                       std::int64_t* vertices_ids) const;

    std::int64_t get_max_degree() const;
    std::int64_t get_max_vertex_attribute() const;

    std::int64_t get_min_degree() const;
    std::int64_t get_min_vertex_attribute() const;

    std::int64_t get_vertex_degree(const std::int64_t vertex) const;
    std::int64_t get_vertex_attribute(const std::int64_t vertex) const;

    std::int64_t get_edge_attribute(const std::int64_t current_vertex,
                                    const std::int64_t vertex) const;

    bool external_data;
    bool bit_representation;
    inner_alloc allocator_;
    std::int64_t n; /* number of graph vertices */
    const std::int64_t* p_degree; /* vertex data dergee arrays */

    std::uint8_t** p_edges_bit; /* bit vectors of edges */

    std::int64_t** p_edges_list; /* adj list of edges */

    const std::int64_t* p_vertex_attribute; /* vertices attribute array */
    std::int64_t const* const* p_edges_attribute; /* edges attribute list */

    void delete_bit_arrays();
    void delete_list_arrays();

    std::int64_t max_element(const std::int64_t* parray) const;
    std::int64_t min_element(const std::int64_t* parray) const;

    std::int64_t get_edge_index(const std::int64_t current_vertex, const std::int64_t vertex) const;

    void init_from_list(const graph_input_list_data* input_list_data);
    void init_from_bit(const graph_input_bit_data* input_bit_data);
};
} // namespace oneapi::dal::preview::subgraph_isomorphism::detail