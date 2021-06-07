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

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/compiler_adapt.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/inner_alloc.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/bit_vector.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

enum graph_storage_scheme { auto_detect, bit, list };

enum edge_direction {
    none = 0, /*!< No edge*/
    both = 1 /*!< Edge exist */
};

template <typename Cpu>
struct graph_input_data {
    std::int64_t vertex_count;
    std::int64_t* degree;
    std::int64_t* attr;
    std::int64_t** edges_attribute;
    inner_alloc allocator_;

    graph_input_data(const std::int64_t vertex_size, inner_alloc allocator);
    ~graph_input_data();
};

template <typename Cpu>
struct graph_input_list_data : public graph_input_data<Cpu> {
    std::int64_t** data;
    using graph_input_data<Cpu>::allocator_;
    using graph_input_data<Cpu>::vertex_count;
    using graph_input_data<Cpu>::degree;
    using graph_input_data<Cpu>::attr;
    using graph_input_data<Cpu>::edges_attribute;

    graph_input_list_data(const std::int64_t vertex_size, inner_alloc allocator);
    ~graph_input_list_data();
};

template <typename Cpu>
struct graph_input_bit_data : public graph_input_data<Cpu> {
    std::uint8_t** data;
    using graph_input_data<Cpu>::allocator_;
    using graph_input_data<Cpu>::vertex_count;
    using graph_input_data<Cpu>::degree;
    using graph_input_data<Cpu>::attr;
    using graph_input_data<Cpu>::edges_attribute;

    graph_input_bit_data(const std::int64_t vertex_size, inner_alloc allocator);
    ~graph_input_bit_data();
};

template <typename Cpu>
struct graph_data {
    const graph_input_bit_data<Cpu>* pbit_data;
    const graph_input_list_data<Cpu>* plist_data;

    graph_data();
    ~graph_data();
};

template <typename Cpu>
class graph {
public:
    graph(const dal::preview::detail::topology<std::int32_t>& t,
          graph_storage_scheme storage_scheme,
          detail::byte_alloc_iface* byte_alloc);
    ~graph();

    static double graph_density(const std::int64_t vertex_count, const std::int64_t edge_count);

    void set_vertex_attribute(const std::int64_t vertex_count,
                              const std::int64_t* pvertices_attribute);
    void set_edge_attribute_lists(const std::int64_t vertex_count,
                                  std::int64_t const* const* p_edges_attribute_list);

    std::int64_t get_vertex_count() const;

    edge_direction check_edge(const std::int64_t current_vertex, const std::int64_t vertex) const;

    std::int64_t get_max_degree() const;
    std::int64_t get_max_vertex_attribute() const;

    std::int64_t get_vertex_degree(const std::int64_t vertex) const;
    std::int64_t get_vertex_attribute(const std::int64_t vertex) const;

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

    void init_bit_representation(const dal::preview::detail::topology<std::int32_t>& t);
    void init_list_representation(const dal::preview::detail::topology<std::int32_t>& t);

    void init_from_list(const graph_input_list_data<Cpu>* input_list_data);
    void init_from_bit(const graph_input_bit_data<Cpu>* input_bit_data);
};

template <typename Cpu>
graph<Cpu>::graph(const dal::preview::detail::topology<std::int32_t>& t,
                  graph_storage_scheme storage_scheme,
                  detail::byte_alloc_iface* byte_alloc)
        : allocator_(byte_alloc) {
    bool use_bit_representation = (storage_scheme != list);

    (use_bit_representation) ? init_bit_representation(t) : init_list_representation(t);

    return;
}

template <typename Cpu>
void graph<Cpu>::init_bit_representation(const dal::preview::detail::topology<std::int32_t>& t) {
    bool has_edges_attribute = false;
    std::int64_t vertex_count = t._vertex_count;
    graph_data<Cpu> graph_data_storage;
    graph_input_bit_data<Cpu>* const memory = allocator_.allocate<graph_input_bit_data<Cpu>>(1);
    graph_data_storage.pbit_data = new (memory) graph_input_bit_data<Cpu>(vertex_count, allocator_);
    for (std::int64_t i = 0; i < vertex_count; i++) {
        auto degree = t._degrees[i];
        graph_data_storage.pbit_data->degree[i] = degree;
    }

    const std::int64_t size_in_bit = bit_vector<Cpu>::bit_vector_size(vertex_count);
    for (std::int64_t i = 0; i < vertex_count; i++) {
        auto degree = t._degrees[i];

        for (std::int64_t j = 0; j < degree; j++) {
            std::int64_t edge_attr = 0;
            std::int64_t vertex_1 = i;
            std::int64_t vertex_2 = t._cols[t._rows[i] + j];

            bit_vector<Cpu>::set_bit(graph_data_storage.pbit_data->data[vertex_1],
                                     vertex_2,
                                     size_in_bit);
            bit_vector<Cpu>::set_bit(graph_data_storage.pbit_data->data[vertex_2],
                                     vertex_1,
                                     size_in_bit);
            if (edge_attr >= 0 || has_edges_attribute) {
                if (graph_data_storage.pbit_data->edges_attribute[i] == nullptr) {
                    graph_data_storage.pbit_data->edges_attribute[i] =
                        allocator_.allocate<std::int64_t>(degree);
                    has_edges_attribute = true;
                }
                graph_data_storage.pbit_data->edges_attribute[i][j] = edge_attr;
            }
        }
    }
    init_from_bit(graph_data_storage.pbit_data);
    return;
}

template <typename Cpu>
void graph<Cpu>::init_list_representation(const dal::preview::detail::topology<std::int32_t>& t) {
    bool has_edges_attribute = false;
    std::int64_t vertex_count = t._vertex_count;
    graph_data<Cpu> graph_data_storage;
    graph_input_list_data<Cpu>* const memory = allocator_.allocate<graph_input_list_data<Cpu>>(1);
    graph_data_storage.plist_data =
        new (memory) graph_input_list_data<Cpu>(vertex_count, allocator_);

    for (std::int64_t i = 0; i < vertex_count; i++) {
        auto degree = t._degrees[i];
        graph_data_storage.plist_data->degree[i] = degree;
        if (degree > 0) {
            graph_data_storage.plist_data->data[i] = allocator_.allocate<std::int64_t>(degree);
        }
        else {
            graph_data_storage.plist_data->data[i] = nullptr;
        }
    }

    for (std::int64_t i = 0; i < vertex_count; i++) {
        auto degree = t._degrees[i];

        for (std::int64_t j = 0; j < degree; j++) {
            std::int64_t edge_attr = 0;
            std::int64_t vertex_2 = t._cols[t._rows[i] + j];

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
    init_from_list(graph_data_storage.plist_data);
    return;
}

template <typename Cpu>
void graph<Cpu>::init_from_list(const graph_input_list_data<Cpu>* input_list_data) {
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

template <typename Cpu>
void graph<Cpu>::init_from_bit(const graph_input_bit_data<Cpu>* input_bit_data) {
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

template <typename Cpu>
graph_data<Cpu>::graph_data() : pbit_data(nullptr),
                                plist_data(nullptr) {}

template <typename Cpu>
graph_data<Cpu>::~graph_data() {}

template <typename Cpu>
graph<Cpu>::~graph() {
    if (external_data) {
        if (bit_representation) {
            delete_bit_arrays();
        }
        if (!bit_representation) {
            delete_list_arrays();
        }
    }
}

template <typename Cpu>
double graph<Cpu>::graph_density(const std::int64_t vertex_count, const std::int64_t edge_count) {
    return (double)(edge_count) / (double)(vertex_count * (vertex_count - 1));
}

template <typename Cpu>
void graph<Cpu>::set_vertex_attribute(const std::int64_t vertex_count,
                                      const std::int64_t* pvertices_attribute) {
    ONEDAL_ASSERT(n == vertex_count);
    ONEDAL_ASSERT(pvertices_attribute != nullptr);
    p_vertex_attribute = pvertices_attribute;
}

template <typename Cpu>
void graph<Cpu>::set_edge_attribute_lists(const std::int64_t vertex_count,
                                          std::int64_t const* const* p_edges_attribute_list) {
    ONEDAL_ASSERT(n == vertex_count);
    ONEDAL_ASSERT(p_edges_attribute_list != nullptr);
    p_edges_attribute = p_edges_attribute_list;
}

template <typename Cpu>
void graph<Cpu>::delete_bit_arrays() {
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

template <typename Cpu>
void graph<Cpu>::delete_list_arrays() {
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

template <typename Cpu>
edge_direction graph<Cpu>::check_edge(const std::int64_t current_vertex,
                                      const std::int64_t vertex) const {
    return static_cast<edge_direction>(
        (bool)(p_edges_bit[current_vertex][bit_vector<Cpu>::byte(vertex)] &
               bit_vector<Cpu>::bit(vertex)));
}

template <typename Cpu>
std::int64_t graph<Cpu>::get_max_degree() const {
    return max_element(p_degree);
}

template <typename Cpu>
std::int64_t graph<Cpu>::get_max_vertex_attribute() const {
    return max_element(p_vertex_attribute);
}

template <typename Cpu>
std::int64_t graph<Cpu>::get_vertex_count() const {
    return n;
}

template <typename Cpu>
std::int64_t graph<Cpu>::get_vertex_degree(std::int64_t vertex) const {
    ONEDAL_ASSERT(vertex < n);
    return p_degree[vertex];
}

template <typename Cpu>
std::int64_t graph<Cpu>::get_vertex_attribute(std::int64_t vertex) const {
    if (p_vertex_attribute == nullptr) {
        return 0;
    }

    return p_vertex_attribute[vertex];
}

template <typename Cpu>
std::int64_t graph<Cpu>::max_element(const std::int64_t* parray) const {
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

template <typename Cpu>
graph_input_data<Cpu>::graph_input_data(const std::int64_t vertex_size, inner_alloc allocator)
        : vertex_count(vertex_size),
          allocator_(allocator) {
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

template <typename Cpu>
graph_input_data<Cpu>::~graph_input_data() {
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

template <typename Cpu>
graph_input_list_data<Cpu>::graph_input_list_data(const std::int64_t vertex_size,
                                                  inner_alloc allocator)
        : graph_input_data<Cpu>(vertex_size, allocator) {
    data = allocator_.template allocate<std::int64_t*>(vertex_count);
    for (int64_t i = 0; i < vertex_count; i++) {
        data[i] = nullptr;
    }
}

template <typename Cpu>
graph_input_list_data<Cpu>::~graph_input_list_data() {
    for (int64_t i = 0; i < vertex_count; i++) {
        if (data[i] != nullptr) {
            allocator_.template deallocate<std::int64_t>(data[i], 0);
            data[i] = nullptr;
        }
    }
    allocator_.template deallocate<std::int64_t*>(data, vertex_count);
}
template <typename Cpu>
graph_input_bit_data<Cpu>::graph_input_bit_data(const std::int64_t vertex_size,
                                                inner_alloc allocator)
        : graph_input_data<Cpu>(vertex_size, allocator) {
    std::int64_t bit_array_size = bit_vector<Cpu>::bit_vector_size(vertex_count);

    data = allocator_.template allocate<std::uint8_t*>(vertex_count);
    for (int64_t i = 0; i < vertex_count; i++) {
        data[i] = allocator_.template allocate<std::uint8_t>(bit_array_size);
        bit_vector<Cpu>::set(bit_array_size, data[i]);
    }
}

template <typename Cpu>
graph_input_bit_data<Cpu>::~graph_input_bit_data() {
    for (int64_t i = 0; i < vertex_count; i++) {
        if (data[i] != nullptr) {
            allocator_.template deallocate<std::uint8_t>(data[i], 0);
            data[i] = nullptr;
        }
    }
    allocator_.template deallocate<std::uint8_t*>(data, vertex_count);
}
} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
