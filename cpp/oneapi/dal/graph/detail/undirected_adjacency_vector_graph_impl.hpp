/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
#include "oneapi/dal/graph/common.hpp"
#include "oneapi/dal/graph/detail/container.hpp"

namespace oneapi::dal::preview::detail {

template <typename IndexType>
constexpr bool is_valid_index_v = dal::detail::is_one_of_v<IndexType, std::int32_t>;

template <typename IndexType>
class topology {
public:
    using vertex_type = IndexType;
    using vertex_set = container<vertex_type>;
    using vertex_iterator = vertex_type*;
    using const_vertex_iterator = const vertex_type*;
    using vertex_size_type = std::int64_t;
    using vertex_edge_type = vertex_type;
    using vertex_edge_size_type = vertex_type;

    using edge_type = std::int64_t;
    using edge_set = container<edge_type>;

    using vertex_edge_set = container<vertex_edge_type>;
    using vertex_edge_iterator = vertex_edge_type*;
    using const_vertex_edge_iterator = const vertex_edge_type*;
    using edge_size_type = std::int64_t;

    // ranges
    using vertex_edge_range = range<vertex_edge_iterator>;
    using const_vertex_edge_range = range<const_vertex_edge_iterator>;

    topology() = default;
    virtual ~topology() = default;

    vertex_set _cols;
    vertex_set _degrees;
    edge_set _rows;
    vertex_edge_set _rows_vertex;

    std::int64_t _vertex_count = 0;
    std::int64_t _edge_count = 0;
};

template <typename VertexValue>
using vertex_values = container<VertexValue>;

template <typename EdgeValue>
using edge_values = container<EdgeValue>;

template <typename VertexValue = empty_value,
          typename EdgeValue = empty_value,
          typename GraphValue = empty_value,
          typename IndexType = std::int32_t,
          typename Allocator = std::allocator<char>>
class ONEDAL_EXPORT undirected_adjacency_vector_graph_impl {
public:
    using allocator_type = Allocator;

    using topology_type = topology<IndexType>;

    // graph weight types
    using graph_user_value_type = GraphValue;
    using const_graph_user_value_type = const graph_user_value_type;

    using vertex_type = typename topology_type::vertex_type;
    using vertex_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<vertex_type>;

    using vertex_set = typename topology_type::vertex_set;
    using vertex_iterator = typename topology_type::vertex_iterator;
    using const_vertex_iterator = typename topology_type::const_vertex_iterator;
    using vertex_size_type = typename topology_type::vertex_size_type;

    using vertex_edge_type = typename topology_type::vertex_edge_type;
    using vertex_edge_size_type = typename topology_type::vertex_edge_size_type;
    using vertex_edge_set = typename topology_type::vertex_edge_set;
    using vertex_edge_iterator = typename topology_type::vertex_edge_iterator;
    using const_vertex_edge_iterator = typename topology_type::const_vertex_edge_iterator;
    using vertex_edge_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<vertex_edge_type>;

    using edge_type = typename topology_type::edge_type;
    using edge_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<edge_type>;
    using edge_set = typename topology_type::edge_set;

    using edge_size_type = typename topology_type::edge_size_type;

    using vertex_user_value_type = VertexValue;
    using vertex_user_value_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<vertex_user_value_type>;
    using vertex_user_value_set = container<vertex_user_value_type>;

    using edge_user_value_type = EdgeValue;
    using edge_user_value_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<edge_user_value_type>;
    using edge_user_value_set = container<edge_user_value_type>;

    // ranges
    using vertex_edge_range = typename topology_type::vertex_edge_range;
    using const_vertex_edge_range = typename topology_type::const_vertex_edge_range;

    undirected_adjacency_vector_graph_impl() = default;

    virtual ~undirected_adjacency_vector_graph_impl() {
        auto& cols = _topology._cols;
        auto& degrees = _topology._degrees;
        auto& rows = _topology._rows;
        auto& rows_vertex = _topology._rows_vertex;

        if (cols.has_mutable_data()) {
            oneapi::dal::preview::detail::deallocate(_vertex_allocator,
                                                     cols.get_mutable_data(),
                                                     cols.get_count());
        }
        if (degrees.has_mutable_data()) {
            oneapi::dal::preview::detail::deallocate(_vertex_allocator,
                                                     degrees.get_mutable_data(),
                                                     degrees.get_count());
        }
        if (rows.has_mutable_data()) {
            oneapi::dal::preview::detail::deallocate(_edge_allocator,
                                                     rows.get_mutable_data(),
                                                     rows.get_count());
        }
        if (rows_vertex.has_mutable_data()) {
            oneapi::dal::preview::detail::deallocate(_vertex_edge_allocator,
                                                     rows_vertex.get_mutable_data(),
                                                     rows_vertex.get_count());
        }

        if (_vertex_values.has_mutable_data()) {
            oneapi::dal::preview::detail::deallocate(_vertex_user_value_allocator,
                                                     _vertex_values.get_mutable_data(),
                                                     _vertex_values.get_count());
        }
        if (_edge_values.has_mutable_data()) {
            oneapi::dal::preview::detail::deallocate(_edge_user_value_allocator,
                                                     _edge_values.get_mutable_data(),
                                                     _edge_values.get_count());
        }
    }

    inline void set_topology(vertex_size_type vertex_count,
                             edge_size_type edge_count,
                             edge_type* offsets,
                             vertex_type* neighbors,
                             vertex_type* degrees) {
        _topology._vertex_count = vertex_count;
        _topology._edge_count = edge_count;
        _topology._rows = edge_set::wrap(offsets, vertex_count + 1);
        _topology._degrees = vertex_set::wrap(degrees, vertex_count);
        _topology._cols = vertex_set::wrap(neighbors, edge_count * 2);
    }

    inline topology<IndexType>& get_topology() {
        return _topology;
    }

    inline vertex_values<VertexValue>& get_vertex_values() {
        return _vertex_values;
    }

    inline edge_values<EdgeValue>& get_edge_values() {
        return _edge_values;
    }

    inline const topology<IndexType> get_topology() const {
        return _topology;
    }

    inline const vertex_values<VertexValue> get_vertex_values() const {
        return _vertex_values;
    }

    inline const edge_values<EdgeValue> get_edge_values() const {
        return _edge_values;
    }

    allocator_type _allocator;
    vertex_allocator_type _vertex_allocator{ _allocator };
    edge_allocator_type _edge_allocator{ _allocator };
    vertex_edge_allocator_type _vertex_edge_allocator{ _allocator };
    vertex_user_value_allocator_type _vertex_user_value_allocator{ _allocator };
    edge_user_value_allocator_type _edge_user_value_allocator{ _allocator };

private:
    topology<IndexType> _topology;
    vertex_values<VertexValue> _vertex_values;
    edge_values<EdgeValue> _edge_values;
};

template <typename IndexType>
constexpr std::int64_t get_topology_vertex_count(const topology<IndexType>& _topology) {
    return _topology._vertex_count;
}

template <typename IndexType>
constexpr std::int64_t get_topology_edge_count(const topology<IndexType>& _topology) {
    return _topology._edge_count;
}

template <typename IndexType>
constexpr auto get_topology_vertex_degree(const topology<IndexType>& _topology,
                                          const IndexType& vertex) noexcept ->
    typename topology<IndexType>::edge_size_type {
    return _topology._degrees[vertex];
}

template <typename IndexType>
constexpr auto get_topology_vertex_neighbors(const topology<IndexType>& _topology,
                                             const IndexType& vertex) noexcept ->
    typename topology<IndexType>::const_vertex_edge_range {
    const IndexType* vertex_neighbors_begin = _topology._cols.get_data() + _topology._rows[vertex];
    const IndexType* vertex_neighbors_end =
        _topology._cols.get_data() + _topology._rows[vertex + 1];
    return std::make_pair(vertex_neighbors_begin, vertex_neighbors_end);
}

} // namespace oneapi::dal::preview::detail
