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

// 1/64 for memory capacity and ~0.005 for cpu.
constexpr double density_threshold = 0.015625;

enum graph_storage_scheme { auto_detect, bit, list };

enum edge_direction {
    none = 0, /*!< No edge*/
    both = 1 /*!< Edge exist */
};

template <typename Cpu>
class graph {
public:
    graph(const dal::preview::detail::topology<std::int32_t>& t,
          graph_storage_scheme storage_scheme,
          byte_alloc_iface_t* byte_alloc);
    ~graph();

    double get_graph_density();

    void set_vertex_attribute(const std::int64_t vertex_count, std::int64_t* pvertices_attribute);
    void set_edge_attribute_lists(const std::int64_t vertex_count,
                                  std::int64_t** p_edges_attribute_list);

    std::int64_t get_vertex_count() const;

    edge_direction check_edge(const std::int64_t current_vertex, const std::int64_t vertex) const;

    std::int64_t get_max_degree() const;
    std::int64_t get_max_vertex_attribute() const;

    std::int64_t get_vertex_degree(const std::int64_t vertex) const;
    std::int64_t get_vertex_attribute(const std::int64_t vertex) const;

    bool external_data;
    bool bit_representation;
    inner_alloc allocator;
    std::int64_t* p_degree; /* vertex data dergee arrays */
    std::uint8_t** p_edges_bit; /* bit vectors of edges */
    std::int64_t** p_edges_list; /* adj list of edges */
    std::int64_t* p_vertex_attribute; /* vertices attribute array */
    std::int64_t** p_edges_attribute; /* edges attribute list */

    void delete_bit_arrays();
    void delete_list_arrays();

    std::int64_t max_element(const std::int64_t* parray) const;

    void init_bit_representation(const dal::preview::detail::topology<std::int32_t>& t);
    void init_list_representation(const dal::preview::detail::topology<std::int32_t>& t);
    void allocate_arrays();

private:
    std::int64_t vertex_count_; /* number of graph vertices */
    std::int64_t edge_count_; /* number of graph edges */
};

template <typename Cpu>
void graph<Cpu>::allocate_arrays() {
    p_degree = allocator.allocate<std::int64_t>(vertex_count_);

    for (std::int64_t i = 0; i < vertex_count_; i++) {
        p_degree[i] = 0;
    }

    if (bit_representation) {
        const std::int64_t bit_array_size = bit_vector<Cpu>::bit_vector_size(vertex_count_);
        p_edges_list = nullptr;
        p_edges_bit = allocator.template allocate<std::uint8_t*>(vertex_count_);
        for (std::int64_t i = 0; i < vertex_count_; ++i) {
            p_edges_bit[i] = allocator.template allocate<std::uint8_t>(bit_array_size);
            bit_vector<Cpu>::set(bit_array_size, p_edges_bit[i]);
        }
    }
    else {
        p_edges_bit = nullptr;
        p_edges_list = allocator.template allocate<std::int64_t*>(vertex_count_);
        for (std::int64_t i = 0; i < vertex_count_; i++) {
            p_edges_list[i] = nullptr;
        }
    }
}

template <typename Cpu>
graph<Cpu>::graph(const dal::preview::detail::topology<std::int32_t>& t,
                  graph_storage_scheme storage_scheme,
                  byte_alloc_iface_t* byte_alloc)
        : external_data(true),
          bit_representation(false),
          allocator(byte_alloc),
          p_vertex_attribute(nullptr),
          p_edges_attribute(nullptr),
          vertex_count_(t.get_vertex_count()),
          edge_count_(t.get_edge_count()) {
    switch (storage_scheme) {
        case auto_detect:
            bit_representation = !(this->get_graph_density() < density_threshold);
            break;
        case bit: bit_representation = true; break;
        case list: bit_representation = false; break;
    };

    allocate_arrays();

    (bit_representation) ? init_bit_representation(t) : init_list_representation(t);

    return;
}

template <typename Cpu>
void graph<Cpu>::init_bit_representation(const dal::preview::detail::topology<std::int32_t>& t) {
    bool has_edges_attribute = false;
    for (std::int64_t i = 0; i < vertex_count_; i++) {
        auto degree = t._degrees[i];
        p_degree[i] = degree;
    }

    for (std::int64_t i = 0; i < vertex_count_; i++) {
        auto degree = t._degrees[i];

        for (std::int64_t j = 0; j < degree; j++) {
            std::int64_t edge_attr = 0;
            std::int64_t vertex_1 = i;
            std::int64_t vertex_2 = t._cols[t._rows[i] + j];

            bit_vector<Cpu>::set_bit(p_edges_bit[vertex_1], vertex_2, vertex_count_);
            bit_vector<Cpu>::set_bit(p_edges_bit[vertex_2], vertex_1, vertex_count_);
            if (edge_attr >= 0 && has_edges_attribute) {
                if (p_edges_attribute[i] == nullptr) {
                    p_edges_attribute[i] = allocator.allocate<std::int64_t>(degree);
                    has_edges_attribute = true;
                }
                p_edges_attribute[i][j] = edge_attr;
            }
        }
    }
    return;
}

template <typename Cpu>
void graph<Cpu>::init_list_representation(const dal::preview::detail::topology<std::int32_t>& t) {
    bool has_edges_attribute = false;
    for (std::int64_t i = 0; i < vertex_count_; i++) {
        auto degree = t._degrees[i];
        p_degree[i] = degree;
        if (degree > 0) {
            p_edges_list[i] = allocator.allocate<std::int64_t>(degree);
        }
        else {
            p_edges_list[i] = nullptr;
        }
    }

    for (std::int64_t i = 0; i < vertex_count_; i++) {
        auto degree = t._degrees[i];

        for (std::int64_t j = 0; j < degree; j++) {
            std::int64_t edge_attr = 0;
            std::int64_t vertex_2 = t._cols[t._rows[i] + j];

            p_edges_list[i][j] = vertex_2;
            if (edge_attr >= 0 && has_edges_attribute) {
                if (p_edges_attribute[i] == nullptr) {
                    // in case of subsequent calls of this function,
                    // p_edges_attribute can have not enough memory to store data
                    p_edges_attribute[i] = allocator.allocate<std::int64_t>(degree);
                    has_edges_attribute = true;
                }
                p_edges_attribute[i][j] = edge_attr;
            }
        }
    }
    return;
}

template <typename Cpu>
void graph<Cpu>::delete_bit_arrays() {
    if (p_edges_bit != nullptr) {
        for (std::int64_t i = 0; i < vertex_count_; i++) {
            if (p_edges_bit[i] != nullptr) {
                allocator.deallocate(p_edges_bit[i], 0);
                p_edges_bit[i] = nullptr;
            }
        }
        allocator.deallocate(p_edges_bit, vertex_count_);
        p_edges_bit = nullptr;
    }
}

template <typename Cpu>
void graph<Cpu>::delete_list_arrays() {
    if (p_edges_list != nullptr) {
        for (std::int64_t i = 0; i < vertex_count_; ++i) {
            if (p_edges_list[i] != nullptr) {
                allocator.deallocate(p_edges_list[i], 0);
                p_edges_list[i] = nullptr;
            }
        }
        allocator.deallocate(p_edges_list, vertex_count_);
        p_edges_list = nullptr;
    }
}

template <typename Cpu>
graph<Cpu>::~graph() {
    allocator.deallocate(p_degree, vertex_count_);

    if (p_edges_attribute) {
        for (std::int64_t i = 0; i < vertex_count_; i++) {
            if (p_edges_attribute[i] != nullptr) {
                allocator.deallocate(p_edges_attribute[i], 1);
                p_edges_attribute[i] = nullptr;
            }
        }
    }

    if (external_data) {
        if (bit_representation) {
            delete_bit_arrays();
        }
        else {
            delete_list_arrays();
        }
    }
}

template <typename Cpu>
double graph<Cpu>::get_graph_density() {
    return (double)(edge_count_) / (double)(vertex_count_ * (vertex_count_ - 1));
}

template <typename Cpu>
void graph<Cpu>::set_vertex_attribute(const std::int64_t vertex_count,
                                      std::int64_t* pvertices_attribute) {
    if (vertex_count == vertex_count_ && pvertices_attribute != nullptr) {
        p_vertex_attribute = pvertices_attribute;
    }
    else {
        throw oneapi::dal::internal_error(
            dal::detail::error_messages::invalid_vertex_edge_attributes());
    }
}

template <typename Cpu>
void graph<Cpu>::set_edge_attribute_lists(const std::int64_t vertex_count,
                                          std::int64_t** p_edges_attribute_list) {
    if (vertex_count == vertex_count_ && p_edges_attribute_list != nullptr) {
        p_edges_attribute = p_edges_attribute_list;
    }
    else {
        throw oneapi::dal::internal_error(
            dal::detail::error_messages::invalid_vertex_edge_attributes());
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
    return vertex_count_;
}

template <typename Cpu>
std::int64_t graph<Cpu>::get_vertex_degree(std::int64_t vertex) const {
    ONEDAL_ASSERT(vertex < vertex_count_);
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
        for (std::int64_t i = 0; i < vertex_count_; i++) {
            if (parray[i] > result) {
                result = parray[i];
            }
        }
    }
    return result;
}
} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
