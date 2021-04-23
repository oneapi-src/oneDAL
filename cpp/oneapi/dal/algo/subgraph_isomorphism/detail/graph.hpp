#pragma once

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/detail/common.hpp"

#if defined(__INTEL_COMPILER)
#define ONEAPI_RESTRICT
//restrict
#else
#define ONEAPI_RESTRICT
#endif

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

struct byte_alloc_iface {
    using byte_t = char;
    virtual byte_t* allocate(std::int64_t n) = 0;
    virtual void deallocate(byte_t* ptr, std::int64_t n) = 0;
};

struct inner_alloc {
    using byte_t = char;

    inner_alloc(byte_alloc_iface* byte_allocator) : byte_allocator_(byte_allocator) {}
    inner_alloc(const byte_alloc_iface* byte_allocator)
            : byte_allocator_(const_cast<byte_alloc_iface*>(byte_allocator)) {}

    template <typename T>
    T* allocate(std::int64_t n) {
        return reinterpret_cast<T*>(byte_allocator_->allocate(n * sizeof(T)));
    }

    template <typename T>
    void deallocate(T* ptr, std::int64_t n) {
        return byte_allocator_->deallocate(reinterpret_cast<byte_t*>(ptr), n * sizeof(T));
    }

    template <typename T>
    oneapi::dal::detail::shared<T> make_shared_memory(std::int64_t n) {
        return oneapi::dal::detail::shared<T>(allocate<T>(n), [=](T* p) {
            deallocate<T>(p, n);
        });
    }

    byte_alloc_iface* get_byte_allocator() {
        return byte_allocator_;
    }

    const byte_alloc_iface* get_byte_allocator() const {
        return byte_allocator_;
    }

private:
    byte_alloc_iface* byte_allocator_;
};

enum graph_storage_scheme { auto_detect, bit, list };

enum graph_status {
    ok = 0, /*!< No error found*/
    bad_arguments = -5, /*!< Bad argument(s) passed*/
    bad_allocation = -11, /*!< Memory allocation error*/
};

enum edge_direction {
    none = 0, /*!< No edge*/
    both = 1 /*!< Edge exist */
};

enum register_size { r8 = 1, r16 = 2, r32 = 4, r64 = 8, r128 = 16, r256 = 32, r512 = 64 };

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

void inversion(std::uint8_t* ONEAPI_RESTRICT vec, const std::int64_t size);
void or_equal(std::uint8_t* ONEAPI_RESTRICT vec,
              const std::uint8_t* ONEAPI_RESTRICT pa,
              const std::int64_t size);
void and_equal(std::uint8_t* ONEAPI_RESTRICT vec,
               const std::uint8_t* ONEAPI_RESTRICT pa,
               const std::int64_t size);

void or_equal(std::uint8_t* ONEAPI_RESTRICT vec,
              const std::int64_t* ONEAPI_RESTRICT bit_index,
              const std::int64_t list_size);
void and_equal(std::uint8_t* ONEAPI_RESTRICT vec,
               const std::int64_t* ONEAPI_RESTRICT bit_index,
               const std::int64_t bit_size,
               const std::int64_t list_size,
               std::int64_t* ONEAPI_RESTRICT tmp_array = nullptr,
               const std::int64_t tmp_size = 0);
void set(std::uint8_t* ONEAPI_RESTRICT vec, std::int64_t size, const std::uint8_t byte_val = 0x0);

class bit_vector {
public:
    // precomputed count of ones in a number from 0 to 255
    // e.g. bit_set_table[2] = 1, because of 2 is 0x00000010
    // e.g. bit_set_table[7] = 7, because of 7 is 0x00000111
    static constexpr std::uint8_t bit_set_table[256] = {
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3,
        4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4,
        4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4,
        5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5,
        4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2,
        3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5,
        5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4,
        5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6,
        4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
    };

    static constexpr std::int64_t byte(std::int64_t x) {
        return x >> 3;
    };

    static constexpr std::uint8_t bit(std::int64_t x) {
        return 1 << (x & 7);
    };

    static constexpr std::int64_t bit_vector_size(std::int64_t vertex_count) {
        return -~(vertex_count >> 3); /* (vertex_count / 8) + 1 */
    };

    static constexpr std::uint8_t power_of_two(const std::uint8_t bit_val) {
        return 31 - _lzcnt_u32(bit_val);
    };

    static void split_by_registers(const std::int64_t vector_size,
                                   std::int64_t* registers,
                                   const register_size max_size);

    bit_vector(const inner_alloc allocator);
    bit_vector(bit_vector& bvec);
    bit_vector(const std::int64_t vector_size, inner_alloc allocator);
    bit_vector(const std::int64_t vector_size, const std::uint8_t byte_val, inner_alloc allocator);
    bit_vector(const std::int64_t vector_size, std::uint8_t* pvector, inner_alloc allocator);
    virtual ~bit_vector();
    graph_status unset_bit(const std::int64_t vertex);
    graph_status set_bit(const std::int64_t vertex);
    void set(const std::uint8_t byte_val = 0x0);
    std::uint8_t* get_vector_pointer();
    std::int64_t size() const;
    std::int64_t popcount() const;

    bit_vector& operator&=(bit_vector& a);
    bit_vector& operator|=(bit_vector& a);
    bit_vector& operator^=(bit_vector& a);
    bit_vector& operator=(bit_vector& a);
    bit_vector& operator~();
    bit_vector& operator&=(const std::uint8_t* pa);
    bit_vector& operator|=(const std::uint8_t* pa);
    bit_vector& operator^=(const std::uint8_t* pa);
    bit_vector& operator=(const std::uint8_t* pa);

    bit_vector& or_equal(const std::int64_t* bit_index, const std::int64_t list_size);
    bit_vector& and_equal(const std::int64_t* bit_index,
                          const std::int64_t list_size,
                          std::int64_t* tmp_array = nullptr,
                          const std::int64_t tmp_size = 0);

    bit_vector(bit_vector&& a);
    bit_vector& operator=(bit_vector&& a);

    bit_vector& andn(const std::uint8_t* pa);
    bit_vector& andn(bit_vector& a);
    bit_vector& orn(const std::uint8_t* pa);
    bit_vector& orn(bit_vector& a);

    std::int64_t min_index() const;
    std::int64_t max_index() const;

    graph_status get_vertices_ids(std::int64_t* vertices_ids);
    bool test_bit(const std::int64_t vertex);

    static graph_status set(const std::int64_t vector_size,
                            std::uint8_t* result_vector,
                            const std::uint8_t byte_val = 0);
    static graph_status set(const std::int64_t vector_size,
                            std::uint8_t* result_vector,
                            const std::uint8_t* vector);
    static graph_status unset_bit(std::uint8_t* result_vector, const std::int64_t vertex);
    static graph_status set_bit(std::uint8_t* result_vector, const std::int64_t vertex);
    static std::int64_t get_bit_index(const std::int64_t vector_size,
                                      const std::uint8_t* vector,
                                      const std::int64_t vertex);
    static std::int64_t popcount(const std::int64_t vector_size, const std::uint8_t* vector);
    static bool test_bit(const std::int64_t vector_size,
                         const std::uint8_t* vector,
                         const std::int64_t vertex);

private:
    inner_alloc allocator_;
    std::uint8_t* vector;
    std::int64_t n;
};

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

private:
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

    friend class sorter;
    friend class matching_engine;
    friend class engine_bundle;
};
} // namespace oneapi::dal::preview::subgraph_isomorphism::detail