
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/graph.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/gcc_adapt.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

graph::graph(const dal::preview::detail::topology<std::int32_t>& t,
             graph_storage_scheme storage_scheme,
             byte_alloc_iface* byte_alloc)
        : allocator_(byte_alloc) {
    bool has_edges_attribute = false;
    bool use_bit_representation = false;
    graph_data graph_data_storage;
    std::int64_t vertex_count = t._vertex_count;
    if (vertex_count <= 0) {
        vertex_count = 0;
        return;
    }

    switch (storage_scheme) {
        case list: {
            use_bit_representation = false;
            break;
        }
        default: {
            use_bit_representation = true;
            break;
        }
    }

    if (use_bit_representation) { // use bit vector
        graph_data_storage.pbit_data = new graph_input_bit_data(vertex_count, allocator_);
    }
    else { // use adj list
        graph_data_storage.plist_data = new graph_input_list_data(vertex_count, allocator_);
    }

    for (std::int64_t i = 0; i < vertex_count; i++) {
        auto degree = t._degrees[i];
        if (use_bit_representation) {
            graph_data_storage.pbit_data->degree[i] = degree;
        }
        else {
            graph_data_storage.plist_data->degree[i] = degree;
            if (degree > 0) {
                graph_data_storage.plist_data->data[i] = allocator_.allocate<std::int64_t>(degree);
            }
            else {
                graph_data_storage.plist_data->data[i] = nullptr;
            }
        }
    }

    for (std::int64_t i = 0; i < vertex_count; i++) {
        auto degree = t._degrees[i];

        for (std::int64_t j = 0; j < degree; j++) {
            std::int64_t edge_attr = 0;
            std::int64_t vertex_1 = i;
            std::int64_t vertex_2 = t._cols[t._rows[i] + j];

            if (use_bit_representation) {
                bit_vector::set_bit(graph_data_storage.pbit_data->data[vertex_1], vertex_2);
                bit_vector::set_bit(graph_data_storage.pbit_data->data[vertex_2], vertex_1);
                if (edge_attr >= 0 || has_edges_attribute) {
                    if (graph_data_storage.pbit_data->edges_attribute[i] == nullptr) {
                        graph_data_storage.pbit_data->edges_attribute[i] =
                            allocator_.allocate<std::int64_t>(degree);
                        has_edges_attribute = true;
                    }
                    graph_data_storage.pbit_data->edges_attribute[i][j] = edge_attr;
                }
            }
            else {
                graph_data_storage.plist_data->data[i][j] = vertex_2;
                if (edge_attr >= 0 || has_edges_attribute) {
                    if (graph_data_storage.plist_data->edges_attribute[i] == nullptr) {
                        graph_data_storage.plist_data->edges_attribute[i] =
                            allocator_.allocate<std::int64_t>(degree);
                        has_edges_attribute = true;
                    }
                    graph_data_storage.plist_data->edges_attribute[i][j] = edge_attr;
                }
            }
        }
    }
    if (graph_data_storage.pbit_data != nullptr) {
        init_from_bit(graph_data_storage.pbit_data);
    }
    else if (graph_data_storage.plist_data != nullptr) {
        init_from_list(graph_data_storage.plist_data);
    }
    return;
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
    if (external_data) {
        if (bit_representation) {
            delete_bit_arrays();
        }
        if (!bit_representation) {
            delete_list_arrays();
        }
    }
    p_degree = nullptr;
    p_vertex_attribute = nullptr;
    p_edges_attribute = nullptr;
}

double graph::graph_density(const std::int64_t vertex_count, const std::int64_t edge_count) {
    return (double)(edge_count) / (double)(vertex_count * (vertex_count - 1));
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

void graph::delete_bit_arrays() {
    if (p_edges_bit != nullptr) {
        for (std::int64_t i = 0; i < n; i++) {
            if (p_edges_bit[i] != nullptr) {
                allocator_.deallocate<std::uint8_t>(p_edges_bit[i], 0);
                p_edges_bit[i] = nullptr;
            }
        }
        allocator_.deallocate<std::uint8_t*>(p_edges_bit, n);
        p_edges_bit = nullptr;
    }
}

void graph::delete_list_arrays() {
    if (p_edges_list != nullptr) {
        for (std::int64_t i = 0; i < n; i++) {
            if (p_edges_list[i] != nullptr) {
                allocator_.deallocate<std::int64_t>(p_edges_list[i], 0);
                p_edges_list[i] = nullptr;
            }
        }
        allocator_.deallocate<std::int64_t*>(p_edges_list, n);
        p_edges_list = nullptr;
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
            bit_vector result(bit_array_size, allocator_.get_byte_allocator());
            bit_vector::set(bit_array_size, result.get_vector_pointer(), p_edges_bit[vertex]);
            return result;
        }
        default: {
            return bit_vector(allocator_.get_byte_allocator());
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

graph_input_data::graph_input_data(inner_alloc allocator) : allocator_(allocator) {
    vertex_count = 0;
    degree = nullptr;
    attr = nullptr;
    edges_attribute = nullptr;
}

graph_input_data::graph_input_data(const std::int64_t vertex_size, inner_alloc allocator)
        : allocator_(allocator) {
    vertex_count = vertex_size;

    degree = allocator_.allocate<std::int64_t>(vertex_count);
    attr = allocator_.allocate<std::int64_t>(vertex_count);

    edges_attribute = allocator_.allocate<std::int64_t*>(vertex_count);
    if (edges_attribute != nullptr) {
        for (int64_t i = 0; i < vertex_count; i++) {
            edges_attribute[i] = nullptr;
            degree[i] = 0;
            attr[i] = 1;
        }
    }
}

graph_input_data::~graph_input_data() {
    allocator_.deallocate<std::int64_t>(degree, vertex_count);
    allocator_.deallocate<std::int64_t>(attr, vertex_count);

    for (int64_t i = 0; i < vertex_count; i++) {
        if (edges_attribute[i] != nullptr) {
            allocator_.deallocate<std::int64_t>(edges_attribute[i], 1);
            edges_attribute[i] = nullptr;
        }
    }
    allocator_.deallocate<std::int64_t*>(edges_attribute, vertex_count);
}

graph_input_list_data::graph_input_list_data(inner_alloc allocator) : graph_input_data(allocator) {
    data = nullptr;
}

graph_input_list_data::graph_input_list_data(const std::int64_t vertex_size, inner_alloc allocator)
        : graph_input_data(vertex_size, allocator) {
    data = allocator_.allocate<std::int64_t*>(vertex_count);
    for (int64_t i = 0; i < vertex_count; i++) {
        data[i] = nullptr;
    }
}

graph_input_list_data::graph_input_list_data(graph_input_data* input_data, inner_alloc allocator)
        : graph_input_data(allocator) {
    vertex_count = input_data->vertex_count;
    degree = input_data->degree;
    attr = input_data->attr;
    edges_attribute = input_data->edges_attribute;

    input_data->vertex_count = 0;
    input_data->degree = nullptr;
    input_data->attr = nullptr;
    input_data->edges_attribute = nullptr;

    data = allocator_.allocate<std::int64_t*>(vertex_count);
    for (int64_t i = 0; i < vertex_count; i++) {
        data[i] = nullptr;
    }
}

graph_input_list_data::~graph_input_list_data() {
    for (int64_t i = 0; i < vertex_count; i++) {
        if (data[i] != nullptr) {
            allocator_.deallocate<std::int64_t>(data[i], 0);
            data[i] = nullptr;
        }
    }
    allocator_.deallocate<std::int64_t*>(data, vertex_count);
}

graph_input_bit_data::graph_input_bit_data(inner_alloc allocator) : graph_input_data(allocator) {
    data = nullptr;
}

graph_input_bit_data::graph_input_bit_data(const std::int64_t vertex_size, inner_alloc allocator)
        : graph_input_data(vertex_size, allocator) {
    std::int64_t bit_array_size = bit_vector::bit_vector_size(vertex_count);

    data = allocator_.allocate<std::uint8_t*>(vertex_count);
    for (int64_t i = 0; i < vertex_count; i++) {
        data[i] = allocator_.allocate<std::uint8_t>(bit_array_size);
        bit_vector::set(bit_array_size, data[i]);
    }
}

graph_input_bit_data::graph_input_bit_data(graph_input_data* input_data, inner_alloc allocator)
        : graph_input_data(allocator) {
    vertex_count = input_data->vertex_count;
    degree = input_data->degree;
    attr = input_data->attr;
    edges_attribute = input_data->edges_attribute;

    input_data->vertex_count = 0;
    input_data->degree = nullptr;
    input_data->attr = nullptr;
    input_data->edges_attribute = nullptr;

    std::int64_t bit_array_size = bit_vector::bit_vector_size(vertex_count);

    data = allocator_.allocate<std::uint8_t*>(vertex_count);
    for (int64_t i = 0; i < vertex_count; i++) {
        data[i] = allocator_.allocate<std::uint8_t>(bit_array_size);
        bit_vector::set(bit_array_size, data[i]);
    }
}

graph_input_bit_data::~graph_input_bit_data() {
    for (int64_t i = 0; i < vertex_count; i++) {
        if (data[i] != nullptr) {
            allocator_.deallocate<std::uint8_t>(data[i], 0);
            data[i] = nullptr;
        }
    }
    allocator_.deallocate<std::uint8_t*>(data, vertex_count);
}
} // namespace oneapi::dal::preview::subgraph_isomorphism::detail