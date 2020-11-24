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

    Index* _vertex_neighbors = nullptr;
    Index* _degrees = nullptr;
    Index* _edge_offsets = nullptr;
    int64_t _vertex_count = 0;
    int64_t _edge_count = 0;
};

template <typename VertexValue>
class vertex_values {
public:
    vertex_values() = default;
    virtual ~vertex_values() = default;
    std::int64_t _vertex_value_count = 0;
    VertexValue* _vertex_value = nullptr;
};

template <typename EdgeValue>
class edge_values {
public:
    edge_values() = default;
    virtual ~edge_values() = default;
    std::int64_t _edge_value_count = 0;
    EdgeValue* _edge_value = nullptr;
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
    using vertex_size_type = typename vertex_set::size_type;

    using edge_type = IndexType;
    using edge_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<edge_type>;
    using edge_allocator_traits =
        typename std::allocator_traits<Allocator>::template rebind_traits<edge_type>;
    using edge_set = detail::graph_container<edge_type, edge_allocator_type>;
    using edge_iterator = edge_type*;
    using const_edge_iterator = const edge_type*;
    using edge_size_type = typename edge_set::size_type;

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

    undirected_adjacency_array_graph_impl() = default; /* : _topology(new topology<IndexType>()),
                     _vertex_values(new vertex_values<VertexValue>()),
                     _edge_values(new edge_values<VertexValue>()) {}*/

    virtual ~undirected_adjacency_array_graph_impl() {
        //auto &_topology = this->get_topology();
        if (_topology._vertex_neighbors != nullptr) {
            vertex_allocator_traits::deallocate(_vertex_allocator,
                                                _topology._vertex_neighbors,
                                                2 * _topology._edge_count);
        }
        if (_topology._degrees != nullptr) {
            vertex_allocator_traits::deallocate(_vertex_allocator,
                                                _topology._degrees,
                                                _topology._vertex_count);
        }
        if (_topology._edge_offsets != nullptr) {
            edge_allocator_traits::deallocate(_edge_allocator,
                                              _topology._edge_offsets,
                                              1 + _topology._vertex_count);
        }
        if (_vertex_values._vertex_value != nullptr) {
            vertex_user_value_allocator_traits::deallocate(_vertex_user_value_allocator,
                                                           _vertex_values._vertex_value,
                                                           _vertex_values._vertex_value_count);
        }
        if (_edge_values._edge_value != nullptr) {
            edge_user_value_allocator_traits::deallocate(_edge_user_value_allocator,
                                                         _edge_values._edge_value,
                                                         _edge_values._edge_value_count);
        }
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
std::int64_t get_topology_vertex_count(const topology<Index>& _topology) {
    return _topology._vertex_count;
}

template <>
std::int64_t get_topology_vertex_count(const topology<std::int32_t>& _topology);

} // namespace oneapi::dal::preview::detail
