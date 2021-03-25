
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/graph.hpp"

//#include <string.h>
#include <xmmintrin.h>

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

void or_equal(std::uint8_t* /*restrict*/ vec,
              const std::uint8_t* /*restrict*/ pa,
              std::int64_t size) {
#pragma ivdep
    for (std::int64_t i = 0; i < size; i++) {
        vec[i] |= pa[i];
    }
}
void and_equal(std::uint8_t* /*restrict*/ vec,
               const std::uint8_t* /*restrict*/ pa,
               std::int64_t size) {
#pragma ivdep
    for (std::int64_t i = 0; i < size; i++) {
        vec[i] &= pa[i];
    }
}

void inversion(std::uint8_t* /*restrict*/ vec, std::int64_t size) {
#pragma ivdep
    for (std::int64_t i = 0; i < size; i++) {
        vec[i] = ~vec[i];
    }
}

void or_equal(std::uint8_t* /*restrict*/ vec,
              const std::int64_t* /*restrict*/ bit_index,
              const std::int64_t list_size) {
#pragma ivdep
    for (std::int64_t i = 0; i < list_size; i++) {
        vec[bit_vector::byte(bit_index[i])] |= bit_vector::bit(bit_index[i]);
    }
}

void set(std::uint8_t* /*restrict*/ vec, std::int64_t size, const std::uint8_t byte_val) {
#pragma vector always
    for (std::int64_t i = 0; i < size; i++) {
        vec[i] = byte_val;
    }
}

void and_equal(std::uint8_t* /*restrict*/ vec,
               const std::int64_t* /*restrict*/ bit_index,
               const std::int64_t bit_size,
               const std::int64_t list_size,
               std::int64_t* /*restrict*/ tmp_array,
               const std::int64_t tmp_size) {
    std::int64_t counter = 0;
#pragma ivdep
    for (std::int64_t i = 0; i < list_size; i++) {
        tmp_array[counter] = bit_index[i];
        counter += bit_vector::bit_set_table[vec[bit_vector::byte(bit_index[i])] &
                                             bit_vector::bit(bit_index[i])];
    }

    set(vec, bit_size, 0x0);

#pragma ivdep
    for (std::int64_t i = 0; i < counter; i++) {
        vec[bit_vector::byte(tmp_array[i])] |= bit_vector::bit(tmp_array[i]);
    }
}

graph::graph() {
    n = 0;
    p_degree = nullptr;
    p_edges_bit = nullptr;
    p_vertex_attribute = nullptr;
    p_edges_attribute = nullptr;
    external_data = false;
    bit_representation = true;

    p_edges_list = nullptr;
}

graph::graph(std::int64_t vertex_count) : graph() {
    n = vertex_count;
    create_bit_arrays(n);
}

graph::graph(const graph_data* pgraph_data) {
    if (pgraph_data->pbit_data != nullptr) {
        init_from_bit(pgraph_data->pbit_data);
    }
    else if (pgraph_data->plist_data != nullptr) {
        init_from_list(pgraph_data->plist_data);
    }
}

graph::graph(const graph_input_list_data* input_list_data) {
    init_from_list(input_list_data);
}

graph::graph(const graph_input_bit_data* input_bit_data) {
    init_from_bit(input_bit_data);
}

void graph::init_from_list(const graph_input_list_data* input_list_data) {
    if (input_list_data != nullptr) {
        external_data = true;
        bit_representation = false;
        n = input_list_data->vertex_count;
        p_degree = input_list_data->degree;
        p_edges_list = input_list_data->data;
        p_vertex_attribute = input_list_data->attr;
        p_edges_attribute = input_list_data->edges_attribute;
    }
}

void graph::init_from_bit(const graph_input_bit_data* input_bit_data) {
    if (input_bit_data != nullptr) {
        external_data = true;
        bit_representation = true;
        n = input_bit_data->vertex_count;
        p_degree = input_bit_data->degree;
        p_edges_bit = input_bit_data->data;
        p_vertex_attribute = input_bit_data->attr;
        p_edges_attribute = input_bit_data->edges_attribute;
    }
}

graph_data::graph_data() {
    pbit_data = nullptr;
    plist_data = nullptr;
}

graph_data::graph_data(const graph_input_bit_data* pbit, const graph_input_list_data* plist) {
    pbit_data = pbit;
    plist_data = plist;
}

graph_data::~graph_data() {
    pbit_data = nullptr;
    plist_data = nullptr;
}

graph::~graph() {
    if (!external_data) {
        if (bit_representation) {
            delete_bit_arrays();
        }
    }
    p_degree = nullptr;
    p_vertex_attribute = nullptr;
    p_edges_attribute = nullptr;
}

double graph::graph_density(const std::int64_t vertex_count, const std::int64_t edge_count) {
    return (double)(edge_count) / (double)(vertex_count * (vertex_count - 1));
}

graph_status graph::load_edge_lists(const std::int64_t vertex_count,
                                    std::int64_t const* const* ptr_edges_list,
                                    const std::int64_t* ptr_degree) {
    if (n != 0) {
        delete_bit_arrays();
    }

    n = vertex_count;
    create_bit_arrays(n);
    p_degree = ptr_degree;

    for (std::int64_t i = 0; i < n; i++) {
        for (std::int64_t j = 0; j < p_degree[i]; j++) {
            set_edge(i, ptr_edges_list[i][j]);
        }
    }
    return ok;
}

graph_status graph::load_vertex_attribute(const std::int64_t vertex_count,
                                          const std::int64_t* pvertices_attribute) {
    if (n != vertex_count || pvertices_attribute == nullptr) {
        return bad_arguments;
    }
    p_vertex_attribute = pvertices_attribute;
    return ok;
}

graph_status graph::load_edge_attribute_lists(const std::int64_t vertex_count,
                                              std::int64_t const* const* p_edges_attribute_list) {
    if (n != vertex_count || p_edges_attribute_list == nullptr) {
        return bad_arguments;
    }

    p_edges_attribute = p_edges_attribute_list;
    return ok;
}

graph_status graph::create_bit_arrays(std::int64_t n) {
    std::int64_t bit_array_size = bit_vector::bit_vector_size(n);

    if (p_edges_bit != nullptr) {
        delete_bit_arrays();
    }

    bool allocation_error_flag = false;

    p_edges_bit = static_cast<std::uint8_t**>(_mm_malloc(sizeof(std::uint8_t*) * n, 64));
    if (p_edges_bit == nullptr) {
        allocation_error_flag = true;
    }

    if (!allocation_error_flag) {
        for (std::int64_t i = 0; i < n; i++) {
            p_edges_bit[i] =
                static_cast<std::uint8_t*>(_mm_malloc(sizeof(std::uint8_t) * bit_array_size, 64));
            if (p_edges_bit[i] == nullptr) {
                allocation_error_flag = true;
            }
        }
    }

    if (allocation_error_flag) {
        delete_bit_arrays();
        return bad_allocation;
    }

    for (std::int64_t i = 0; i < n; i++) {
        for (std::int64_t j = 0; j < bit_array_size; j++) {
            p_edges_bit[i][j] = 0;
        }
    }

    return ok;
}

void graph::delete_bit_arrays() {
    if (p_edges_bit != nullptr) {
        for (std::int64_t i = 0; i < n; i++) {
            if (p_edges_bit[i] != nullptr) {
                _mm_free(p_edges_bit[i]);
                p_edges_bit[i] = nullptr;
            }
        }
        _mm_free(p_edges_bit);
        p_edges_bit = nullptr;
    }
}

void graph::set_edge(const std::int64_t current_vertex, const std::int64_t vertex) {
    p_edges_bit[current_vertex][bit_vector::byte(vertex)] |= bit_vector::bit(vertex);
}

edge_direction graph::check_edge(const std::int64_t current_vertex,
                                 const std::int64_t vertex) const {
    return static_cast<edge_direction>(
        (bool)(p_edges_bit[current_vertex][bit_vector::byte(vertex)] & bit_vector::bit(vertex)));
}

bool graph::has_edge(const std::int64_t vertex_first, const std::int64_t vertex_second) const {
    return check_edge(vertex_first, vertex_second);
}

std::int64_t graph::get_max_degree() const {
    return max_element(p_degree);
}

std::int64_t graph::get_max_vertex_attribute() const {
    return max_element(p_vertex_attribute);
}

std::int64_t graph::get_min_degree() const {
    return min_element(p_degree);
}

std::int64_t graph::get_min_vertex_attribute() const {
    return min_element(p_vertex_attribute);
}

std::int64_t graph::get_vertex_count() const {
    return n;
}

std::int64_t graph::get_vertex_degree(std::int64_t vertex) const {
    return p_degree[vertex];
}

std::int64_t graph::get_vertex_attribute(std::int64_t vertex) const {
    if (p_vertex_attribute == nullptr) {
        return 0;
    }

    return p_vertex_attribute[vertex];
}

std::int64_t graph::max_element(const std::int64_t* parray) const {
    std::int64_t result = 0;

    if (parray != nullptr) {
        for (std::int64_t i = 0; i < n; i++) {
            if (parray[i] > result) {
                result = parray[i];
            }
        }
    }
    return result;
}

std::int64_t graph::min_element(const std::int64_t* parray) const {
    std::int64_t result = 0;

    if (parray != nullptr) {
        for (std::int64_t i = 0; i < n; i++) {
            if (parray[i] < result) {
                result = parray[i];
            }
        }
    }
    return result;
}

graph_status graph::get_edge_vertices_ids(const std::int64_t current_vertex,
                                          std::int64_t* vertices_ids) const {
    if (vertices_ids == nullptr) {
        return bad_arguments;
    }

    std::int64_t bit_array_size = bit_vector::bit_vector_size(n);
    std::uint8_t current_byte = 0;
    std::uint8_t current_bit = 0;
    std::int64_t vertex_iterator = 0;
    for (std::int64_t i = 0; i < bit_array_size; i++) {
        current_byte = p_edges_bit[current_vertex][i];
        while (current_byte > 0) {
            current_bit = current_byte & (-current_byte);
            current_byte &= ~current_bit;
            vertices_ids[vertex_iterator] = 8 * i + bit_vector::power_of_two(current_bit);
            vertex_iterator++;
        }
    }
    return ok;
}

bit_vector graph::get_vertex_neighbors(const std::int64_t vertex,
                                       const edge_direction edge_type) const {
    std::int64_t bit_array_size = bit_vector::bit_vector_size(n);
    switch (edge_type) {
        case both: {
            bit_vector result(bit_array_size);
            bit_vector::set(bit_array_size, result.get_vector_pointer(), p_edges_bit[vertex]);
            return result;
        }
        default: {
            return bit_vector();
        }
    }
}

std::int64_t graph::get_edge_index(const std::int64_t current_vertex,
                                   const std::int64_t vertex) const {
    if (!check_edge(current_vertex, vertex)) {
        return bad_arguments;
    }

    return bit_vector::get_bit_index((n / 8) + 1, p_edges_bit[current_vertex], vertex);
}

std::int64_t graph::get_edge_attribute(const std::int64_t current_vertex,
                                       const std::int64_t vertex) const {
    std::int64_t attribute_index = get_edge_index(current_vertex, vertex);
    if (attribute_index < 0) {
        return attribute_index;
    }

    return p_edges_attribute[current_vertex][attribute_index];
}

graph_status bit_vector::set(const std::int64_t vector_size,
                             std::uint8_t* result_vector,
                             const std::uint8_t* vector) {
    if (result_vector == nullptr || vector == nullptr) {
        return bad_allocation;
    }

    for (std::int64_t i = 0; i < vector_size; i++) {
        result_vector[i] = vector[i];
    }

    /*
      TODO improvements  use register and sets functions
    */

    return ok;
}

graph_status bit_vector::set(const std::int64_t vector_size,
                             std::uint8_t* result_vector,
                             const std::uint8_t byte_val) {
    if (result_vector == nullptr) {
        return bad_allocation;
    }

    for (std::int64_t i = 0; i < vector_size; i++) {
        result_vector[i] = byte_val;
    }

    /*
        TODO improvements  use register and sets functions
    */

    return ok;
}

void bit_vector::set(const std::uint8_t byte_val) {
    const std::int64_t nn = n;
#pragma vector always
    for (std::int64_t i = 0; i < nn; i++) {
        vector[i] = byte_val;
    }
}

graph_status bit_vector::set_bit(std::uint8_t* result_vector, const std::int64_t vertex) {
    if (result_vector == nullptr) {
        return bad_allocation;
    }
    result_vector[byte(vertex)] |= bit(vertex);
    return ok;
}

graph_status bit_vector::unset_bit(std::uint8_t* result_vector, const std::int64_t vertex) {
    if (result_vector == nullptr) {
        return bad_allocation;
    }
    result_vector[byte(vertex)] &= ~bit(vertex);
    return ok;
}

bit_vector::bit_vector() {
    vector = nullptr;
    n = 0;
}

bit_vector::bit_vector(bit_vector& bvec) {
    n = bvec.n;
    vector = static_cast<std::uint8_t*>(_mm_malloc(sizeof(std::uint8_t) * n, 64));
    set(n, vector, bvec.vector);
}

bit_vector::bit_vector(const std::int64_t vector_size) : bit_vector() {
    n = vector_size;
    vector = static_cast<std::uint8_t*>(_mm_malloc(sizeof(std::uint8_t) * n, 64));
    set(n, vector);
}

bit_vector::bit_vector(const std::int64_t vector_size, const std::uint8_t byte_val) {
    n = vector_size;
    vector = static_cast<std::uint8_t*>(_mm_malloc(sizeof(std::uint8_t) * n, 64));
    set(n, vector, byte_val);
}

bit_vector::bit_vector(bit_vector&& a) : vector(a.vector) {
    n = a.n;
    a.n = 0;
    a.vector = nullptr;
}

bit_vector& bit_vector::operator=(bit_vector&& a) {
    if (&a == this) {
        return *this;
    }
    vector = a.vector;
    n = a.n;

    a.vector = nullptr;
    a.n = 0;
    return *this;
}

bit_vector::~bit_vector() {
    if (vector != nullptr) {
        _mm_free(vector);
        vector = nullptr;
        n = 0;
    }
}

bit_vector::bit_vector(const std::int64_t vector_size, std::uint8_t* pvector) {
    n = vector_size;
    vector = pvector;
    pvector = nullptr;
}

graph_status bit_vector::unset_bit(const std::int64_t vertex) {
    return unset_bit(vector, vertex);
}

graph_status bit_vector::set_bit(const std::int64_t vertex) {
    return set_bit(vector, vertex);
}

std::uint8_t* bit_vector::get_vector_pointer() {
    return vector;
}

std::int64_t bit_vector::size() const {
    return n;
}

bit_vector& bit_vector::operator&=(bit_vector& a) {
    if (n <= a.size()) {
        std::uint8_t* pa = a.get_vector_pointer();
        for (std::int64_t i = 0; i < n; i++) {
            vector[i] &= pa[i];
        }
    }
    return *this;
}

bit_vector& bit_vector::operator|=(bit_vector& a) {
    if (n <= a.size()) {
        std::uint8_t* pa = a.get_vector_pointer();
        for (std::int64_t i = 0; i < n; i++) {
            vector[i] |= pa[i];
        }
    }
    return *this;
}

bit_vector& bit_vector::operator^=(bit_vector& a) {
    if (n <= a.size()) {
        std::uint8_t* pa = a.get_vector_pointer();
        for (std::int64_t i = 0; i < n; i++) {
            vector[i] ^= pa[i];
        }
    }
    return *this;
}

bit_vector& bit_vector::operator=(bit_vector& a) {
    if (n <= a.size()) {
        std::uint8_t* pa = a.get_vector_pointer();
        for (std::int64_t i = 0; i < n; i++) {
            vector[i] = pa[i];
        }
    }
    return *this;
}

bit_vector& bit_vector::operator~() {
    const std::int64_t nn = n;
#pragma ivdep
    for (std::int64_t i = 0; i < nn; i++) {
        vector[i] = ~vector[i];
    }
    return *this;
}

bit_vector& bit_vector::operator&=(const std::uint8_t* /*restrict*/ pa) {
    for (std::int64_t i = 0; i < n; i++) {
        vector[i] &= pa[i];
    }
    return *this;
}

bit_vector& bit_vector::operator|=(const std::uint8_t* /*restrict*/ pa) {
    for (std::int64_t i = 0; i < n; i++) {
        vector[i] |= pa[i];
    }
    return *this;
}

bit_vector& bit_vector::operator^=(const std::uint8_t* pa) {
    for (std::int64_t i = 0; i < n; i++) {
        vector[i] ^= pa[i];
    }
    return *this;
}

bit_vector& bit_vector::operator=(const std::uint8_t* pa) {
    for (std::int64_t i = 0; i < n; i++) {
        vector[i] = pa[i];
    }
    return *this;
}

bit_vector& bit_vector::andn(const std::uint8_t* pa) {
    for (std::int64_t i = 0; i < n; i++) {
        vector[i] &= ~pa[i];
    }
    return *this;
}

bit_vector& bit_vector::andn(bit_vector& a) {
    if (n <= a.size()) {
        std::uint8_t* pa = a.get_vector_pointer();
        for (std::int64_t i = 0; i < n; i++) {
            vector[i] &= ~pa[i];
        }
    }
    return *this;
}

bit_vector& bit_vector::orn(const std::uint8_t* pa) {
    for (std::int64_t i = 0; i < n; i++) {
        vector[i] |= ~pa[i];
    }
    return *this;
}

bit_vector& bit_vector::orn(bit_vector& a) {
    if (n <= a.size()) {
        std::uint8_t* pa = a.get_vector_pointer();
        for (std::int64_t i = 0; i < n; i++) {
            vector[i] |= ~pa[i];
        }
    }
    return *this;
}

bit_vector& bit_vector::or_equal(const std::int64_t* bit_index, const std::int64_t list_size) {
    for (std::int64_t i = 0; i < list_size; i++) {
        vector[byte(bit_index[i])] |= bit(bit_index[i]);
    }
    return *this;
}

bit_vector& bit_vector::and_equal(const std::int64_t* bit_index,
                                  const std::int64_t list_size,
                                  std::int64_t* tmp_array,
                                  const std::int64_t tmp_size) {
    std::int64_t counter = 0;
    for (std::int64_t i = 0; i < list_size; i++) {
        tmp_array[counter] = bit_index[i];
        counter += bit_set_table[vector[byte(bit_index[i])] & bit(bit_index[i])];
    }

    set(0x0);
    for (std::int64_t i = 0; i < counter; i++) {
        vector[byte(tmp_array[i])] |= bit(tmp_array[i]);
    }
    return *this;
}

graph_status bit_vector::get_vertices_ids(std::int64_t* vertices_ids) {
    std::int64_t vertex_iterator = 0;
    if (vertices_ids == nullptr) {
        return bad_arguments;
    }
    std::uint8_t current_byte = 0;
    std::uint8_t current_bit = 0;
    for (std::int64_t i = 0; i < n; i++) {
        current_byte = vector[i];
        while (current_byte > 0) {
            current_bit = current_byte & (-current_byte);
            current_byte &= ~current_bit;
            vertices_ids[vertex_iterator] = 8 * i + power_of_two(current_bit);
            vertex_iterator++;
        }
    }
    return static_cast<graph_status>(vertex_iterator);
}

bool bit_vector::test_bit(const std::int64_t vertex) {
    return vector[byte(vertex)] & bit(vertex);
}

bool bit_vector::test_bit(const std::int64_t vector_size,
                          const std::uint8_t* vector,
                          const std::int64_t vertex) {
    if (vertex / 8 > vector_size) {
        return false;
    }
    return vector[byte(vertex)] & bit(vertex);
}

std::int64_t bit_vector::get_bit_index(const std::int64_t vector_size,
                                       const std::uint8_t* vector,
                                       const std::int64_t vertex) {
    if (vertex / 8 > vector_size) {
        return bad_arguments;
    }

    std::int64_t nbyte = byte(vertex);
    std::uint8_t bit = vertex & 7;
    std::int64_t result = 0;

    for (std::int64_t i = 0; i < nbyte; i++)
        result += bit_vector::bit_set_table[vector[i]];

    std::uint8_t checkbyte = vector[nbyte];
    for (std::uint8_t i = 0; i < bit; i++) {
        result += checkbyte & 1;
        checkbyte >>= 1;
    }
    return result;
}

std::int64_t bit_vector::popcount(const std::int64_t vector_size, const std::uint8_t* vector) {
    if (vector == nullptr) {
        return bad_arguments;
    }

    std::int64_t result = 0;
    for (std::int64_t i = 0; i < vector_size; i++) {
        result += bit_vector::bit_set_table[vector[i]];
    }

    return result;
}

std::int64_t bit_vector::popcount() const {
    return popcount(n, vector);
}

void bit_vector::split_by_registers(const std::int64_t vector_size,
                                    std::int64_t* registers,
                                    const register_size max_size) {
    std::int64_t size = vector_size;
    for (std::int64_t i = power_of_two(max_size); i >= 0; i--) {
        registers[i] = size >> i;
        size -= registers[i] << i;
    }
}

std::int64_t bit_vector::min_index() const {
    std::int64_t result = 0;

    for (std::int64_t i = n; i > 0; i--) {
        if (vector[i] > 0) {
            result = power_of_two(vector[i]);
            result += (i << 3);
            break;
        }
    }
    return result;
}

std::int64_t bit_vector::max_index() const {
    std::int64_t result = 1 << n;

    for (std::int64_t i = 0; i < n; i++) {
        if (vector[i] > 0) {
            result = power_of_two(vector[i]);
            result += (i << 3);
            break;
        }
    }
    return result;
}

graph_input_data::graph_input_data() {
    vertex_count = 0;
    degree = nullptr;
    attr = nullptr;
    edges_attribute = nullptr;
}

graph_input_data::graph_input_data(const std::int64_t vertex_size) {
    vertex_count = vertex_size;

    degree = static_cast<std::int64_t*>(_mm_malloc(sizeof(std::int64_t) * vertex_count, 64));
    attr = static_cast<std::int64_t*>(_mm_malloc(sizeof(std::int64_t) * vertex_count, 64));

    edges_attribute =
        static_cast<std::int64_t**>(_mm_malloc(sizeof(std::int64_t*) * vertex_count, 64));
    if (edges_attribute != nullptr) {
        for (int64_t i = 0; i < vertex_count; i++) {
            edges_attribute[i] = nullptr;
            degree[i] = 0;
            attr[i] = 1;
        }
    }
}

graph_input_data::~graph_input_data() {
    _mm_free(degree);
    _mm_free(attr);

    for (int64_t i = 0; i < vertex_count; i++) {
        if (edges_attribute[i] != nullptr) {
            _mm_free(edges_attribute[i]);
            edges_attribute[i] = nullptr;
        }
    }
    _mm_free(edges_attribute);
}

graph_input_list_data::graph_input_list_data() : graph_input_data() {
    data = nullptr;
}

graph_input_list_data::graph_input_list_data(const std::int64_t vertex_size)
        : graph_input_data(vertex_size) {
    data = static_cast<std::int64_t**>(_mm_malloc(sizeof(std::int64_t*) * vertex_count, 64));
    for (int64_t i = 0; i < vertex_count; i++) {
        data[i] = nullptr;
    }
}

graph_input_list_data::graph_input_list_data(graph_input_data* input_data) {
    vertex_count = input_data->vertex_count;
    degree = input_data->degree;
    attr = input_data->attr;
    edges_attribute = input_data->edges_attribute;

    input_data->vertex_count = 0;
    input_data->degree = nullptr;
    input_data->attr = nullptr;
    input_data->edges_attribute = nullptr;

    data = static_cast<std::int64_t**>(_mm_malloc(sizeof(std::int64_t*) * vertex_count, 64));
    for (int64_t i = 0; i < vertex_count; i++) {
        data[i] = nullptr;
    }
}

graph_input_list_data::~graph_input_list_data() {
    for (int64_t i = 0; i < vertex_count; i++) {
        if (data[i] != nullptr) {
            _mm_free(data[i]);
            data[i] = nullptr;
        }
    }
    _mm_free(data);
}

graph_input_bit_data::graph_input_bit_data() : graph_input_data() {
    data = nullptr;
}

graph_input_bit_data::graph_input_bit_data(const std::int64_t vertex_size)
        : graph_input_data(vertex_size) {
    std::int64_t bit_array_size = bit_vector::bit_vector_size(vertex_count);

    data = static_cast<std::uint8_t**>(_mm_malloc(sizeof(std::uint8_t*) * vertex_count, 64));
    for (int64_t i = 0; i < vertex_count; i++) {
        data[i] = static_cast<std::uint8_t*>(_mm_malloc(sizeof(std::uint8_t) * bit_array_size, 64));
        bit_vector::set(bit_array_size, data[i]);
    }
}

graph_input_bit_data::graph_input_bit_data(graph_input_data* input_data) {
    vertex_count = input_data->vertex_count;
    degree = input_data->degree;
    attr = input_data->attr;
    edges_attribute = input_data->edges_attribute;

    input_data->vertex_count = 0;
    input_data->degree = nullptr;
    input_data->attr = nullptr;
    input_data->edges_attribute = nullptr;

    std::int64_t bit_array_size = bit_vector::bit_vector_size(vertex_count);

    data = static_cast<std::uint8_t**>(_mm_malloc(sizeof(std::uint8_t*) * vertex_count, 64));
    for (int64_t i = 0; i < vertex_count; i++) {
        data[i] = static_cast<std::uint8_t*>(_mm_malloc(sizeof(std::uint8_t) * bit_array_size, 64));
        bit_vector::set(bit_array_size, data[i]);
    }
}

graph_input_bit_data::~graph_input_bit_data() {
    for (int64_t i = 0; i < vertex_count; i++) {
        if (data[i] != nullptr) {
            _mm_free(data[i]);
            data[i] = nullptr;
        }
    }
    _mm_free(data);
}
} // namespace oneapi::dal::preview::subgraph_isomorphism::backend