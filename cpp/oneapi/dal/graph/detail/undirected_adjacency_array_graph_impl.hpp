/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/graph/detail/graph_container.hpp"
#include "oneapi/dal/graph/graph_common.hpp"

namespace oneapi::dal::preview::detail {

template <typename Index>
class topology {
public:
    topology() = default;
    virtual ~topology() = default;

    array<Index> _vertex_neighbors;
    array<Index> _degrees;
    array<Index> _edge_offsets;
    int64_t _vertex_count = 0;
    int64_t _edge_count = 0;
};

template <typename VertexValue>
class vertex_values {
public:
    vertex_values() = default;
    virtual ~vertex_values() = default;
    array<VertexValue> _vertex_value;
};

template <typename EdgeValue>
class edge_values {
public:
    edge_values() = default;
    virtual ~edge_values() = default;
    array<EdgeValue> _edge_value;
};

template <typename VertexValue = empty_value,
          typename EdgeValue = empty_value,
          typename GraphValue = empty_value,
          typename IndexType = std::int32_t,
          typename Allocator = std::allocator<char>>
class ONEDAL_EXPORT undirected_adjacency_array_graph_impl {
public:
    using allocator_type = Allocator;

    using vertex_type = IndexType;
    using vertex_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<vertex_type>;
    using vertex_allocator_traits =
        typename std::allocator_traits<Allocator>::template rebind_traits<vertex_type>;

    using vertex_set = detail::graph_container<vertex_type, vertex_allocator_type>;
    using vertex_iterator = vertex_type*;
    using const_vertex_iterator = const vertex_type*;
    using vertex_size_type = std::int64_t;

    using edge_type = IndexType;
    using edge_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<edge_type>;
    using edge_allocator_traits =
        typename std::allocator_traits<Allocator>::template rebind_traits<edge_type>;
    using edge_set = detail::graph_container<edge_type, edge_allocator_type>;
    using edge_iterator = edge_type*;
    using const_edge_iterator = const edge_type*;
    using edge_size_type = std::int64_t;

    using vertex_user_value_type = VertexValue;
    using vertex_user_value_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<vertex_user_value_type>;
    using vertex_user_value_allocator_traits =
        typename std::allocator_traits<Allocator>::template rebind_traits<vertex_user_value_type>;
    using vertex_user_value_set =
        detail::graph_container<vertex_user_value_type, vertex_user_value_allocator_type>;

    using edge_user_value_type = EdgeValue;
    using edge_user_value_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<edge_user_value_type>;
    using edge_user_value_allocator_traits =
        typename std::allocator_traits<Allocator>::template rebind_traits<edge_user_value_type>;
    using edge_user_value_set =
        detail::graph_container<edge_user_value_type, edge_user_value_allocator_type>;

    undirected_adjacency_array_graph_impl() = default;

    virtual ~undirected_adjacency_array_graph_impl() {
        auto& vertex_neighbors = _topology._vertex_neighbors;
        auto& degrees = _topology._degrees;
        auto& edge_offsets = _topology._edge_offsets;
        auto& vertex_value = _vertex_values._vertex_value;
        auto& edge_value = _edge_values._edge_value;

        if (vertex_neighbors.get_data() != nullptr && vertex_neighbors.has_mutable_data()) {
            vertex_allocator_traits::deallocate(_vertex_allocator,
                                                vertex_neighbors.get_mutable_data(),
                                                vertex_neighbors.get_count());
        }
        if (degrees.get_data() != nullptr && degrees.has_mutable_data()) {
            vertex_allocator_traits::deallocate(_vertex_allocator,
                                                degrees.get_mutable_data(),
                                                degrees.get_count());
        }
        if (edge_offsets.get_data() != nullptr && edge_offsets.has_mutable_data()) {
            edge_allocator_traits::deallocate(_edge_allocator,
                                              edge_offsets.get_mutable_data(),
                                              edge_offsets.get_count());
        }
        if (vertex_value.get_data() != nullptr && vertex_value.has_mutable_data()) {
            vertex_user_value_allocator_traits::deallocate(_vertex_user_value_allocator,
                                                           vertex_value.get_mutable_data(),
                                                           vertex_value.get_count());
        }
        if (edge_value.get_data() != nullptr && edge_value.has_mutable_data()) {
            edge_user_value_allocator_traits::deallocate(_edge_user_value_allocator,
                                                         edge_value.get_mutable_data(),
                                                         edge_value.get_count());
        }
    }

    void set_topology(vertex_size_type vertex_count,
                      edge_size_type edge_count,
                      edge_type* offsets,
                      vertex_type* neighbors,
                      vertex_type* degrees) {
        _topology._vertex_count = vertex_count;
        _topology._edge_count = edge_count;
        _topology._edge_offsets = array<edge_type>::wrap(offsets, vertex_count + 1);
        _topology._degrees = array<vertex_type>::wrap(degrees, vertex_count);
        _topology._vertex_neighbors = array<vertex_type>::wrap(neighbors, edge_count * 2);
    }

    topology<IndexType>& get_topology() {
        return _topology;
    }

    vertex_values<VertexValue>& get_vertex_values() {
        return _vertex_values;
    }

    edge_values<EdgeValue>& get_edge_values() {
        return _edge_values;
    }

    const topology<IndexType>& get_topology() const {
        return _topology;
    }

    const vertex_values<VertexValue>& get_vertex_values() const {
        return _vertex_values;
    }

    const edge_values<EdgeValue>& get_edge_values() const {
        return _edge_values;
    }

    allocator_type _allocator;
    vertex_allocator_type _vertex_allocator;
    edge_allocator_type _edge_allocator;
    vertex_user_value_allocator_type _vertex_user_value_allocator;
    edge_user_value_allocator_type _edge_user_value_allocator;

private:
    topology<IndexType> _topology;
    vertex_values<VertexValue> _vertex_values;
    edge_values<EdgeValue> _edge_values;
};

template <typename Index>
inline std::int64_t get_topology_vertex_count(const topology<Index>& _topology) {
    return _topology._vertex_count;
}

template <typename Index>
inline std::int64_t get_topology_edge_count(const topology<Index>& _topology) {
    return _topology._edge_count;
}

template <typename Index>
inline auto get_topology_vertex_degree(const topology<Index>& _topology,
                                       const Index& vertex) noexcept {
    return _topology._degrees[vertex];
}

template <typename Index>
inline auto get_topology_vertex_neighbors(const topology<Index>& _topology,
                                          const Index& vertex) noexcept {
    const Index* vertex_neighbors_begin =
        _topology._vertex_neighbors.get_data() + _topology._edge_offsets[vertex];
    const Index* vertex_neighbors_end =
        _topology._vertex_neighbors.get_data() + _topology._edge_offsets[vertex + 1];
    return std::make_pair(vertex_neighbors_begin, vertex_neighbors_end);
}

} // namespace oneapi::dal::preview::detail
