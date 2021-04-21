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
class directed_topology {
public:
    using vertex_type = IndexType;
    using vertex_set = container<vertex_type>;
    using vertex_iterator = vertex_type*;
    using const_vertex_iterator = const vertex_type*;
    using vertex_size_type = std::int64_t;
    using vertex_edge_type = vertex_type;
    using vertex_edge_size_type = vertex_type;

    using vertex_outward_edge_type = vertex_type;
    using vertex_outward_edge_size_type = vertex_type;

    using edge_type = std::int64_t;
    using edge_set = container<edge_type>;

    using vertex_edge_set = container<vertex_edge_type>;
    using vertex_edge_iterator = vertex_edge_type*;
    using const_vertex_edge_iterator = const vertex_edge_type*;

    using vertex_outward_edge_set = container<vertex_outward_edge_type>;
    using vertex_outward_edge_iterator = vertex_outward_edge_type*;
    using const_vertex_outward_edge_iterator = const vertex_outward_edge_type*;

    using edge_size_type = std::int64_t;

    // ranges
    using vertex_edge_range = range<vertex_edge_iterator>;
    using const_vertex_edge_range = range<const_vertex_edge_iterator>;

    using vertex_outward_edge_range = range<vertex_outward_edge_iterator>;
    using const_vertex_outward_edge_range = range<const_vertex_outward_edge_iterator>;

    directed_topology() = default;
    virtual ~directed_topology() = default;

    ONEDAL_FORCEINLINE std::int64_t get_vertex_count() const {
        return _vertex_count;
    }

    ONEDAL_FORCEINLINE std::int64_t get_edge_count() const {
        return _edge_count;
    }

    ONEDAL_FORCEINLINE auto get_vertex_degree(const IndexType& vertex) const noexcept
        -> edge_size_type {
        return _degrees_ptr[vertex];
    }

    ONEDAL_FORCEINLINE auto get_vertex_neighbors(const IndexType& vertex) const noexcept
        -> const_vertex_edge_range {
        const IndexType* vertex_neighbors_begin = _cols.get_data() + _rows[vertex];
        const IndexType* vertex_neighbors_end = _cols.get_data() + _rows[vertex + 1];
        return std::make_pair(vertex_neighbors_begin, vertex_neighbors_end);
    }

    ONEDAL_FORCEINLINE auto get_vertex_neighbors_begin(const IndexType& vertex) const noexcept
        -> const_vertex_edge_iterator {
        return _cols_ptr + _rows[vertex];
    }

    ONEDAL_FORCEINLINE auto get_vertex_neighbors_end(const IndexType& vertex) const noexcept
        -> const_vertex_edge_iterator {
        return _cols_ptr + _rows[vertex + 1];
    }
    vertex_set _cols;
    vertex_set _degrees;
    edge_set _rows;
    vertex_edge_set _rows_vertex;

    const vertex_type* _cols_ptr;
    const vertex_type* _degrees_ptr;
    const edge_type* _rows_ptr;

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
class ONEDAL_EXPORT directed_adjacency_vector_graph_impl {
public:
    using allocator_type = Allocator;

    using directed_topology_type = directed_topology<IndexType>;

    // graph weight types
    using graph_user_value_type = GraphValue;
    using const_graph_user_value_type = const graph_user_value_type;

    using vertex_type = typename directed_topology_type::vertex_type;
    using vertex_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<vertex_type>;

    using vertex_set = typename directed_topology_type::vertex_set;
    using vertex_iterator = typename directed_topology_type::vertex_iterator;
    using const_vertex_iterator = typename directed_topology_type::const_vertex_iterator;
    using vertex_size_type = typename directed_topology_type::vertex_size_type;

    using vertex_edge_type = typename directed_topology_type::vertex_edge_type;
    using vertex_edge_size_type = typename directed_topology_type::vertex_edge_size_type;
    using vertex_edge_set = typename directed_topology_type::vertex_edge_set;
    using vertex_edge_iterator = typename directed_topology_type::vertex_edge_iterator;
    using const_vertex_edge_iterator = typename directed_topology_type::const_vertex_edge_iterator;
    using vertex_edge_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<vertex_edge_type>;

    // vertex edge types
    using vertex_outward_edge_type = typename directed_topology_type::vertex_outward_edge_type;
    using vertex_outward_edge_size_type =
        typename directed_topology_type::vertex_outward_edge_size_type;
    using vertex_outward_edge_set = typename directed_topology_type::vertex_outward_edge_set;
    using vertex_outward_edge_iterator =
        typename directed_topology_type::vertex_outward_edge_iterator;
    using const_vertex_outward_edge_iterator =
        typename directed_topology_type::const_vertex_outward_edge_iterator;
    using vertex_outward_edge_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<vertex_outward_edge_type>;

    using edge_type = typename directed_topology_type::edge_type;
    using edge_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<edge_type>;
    using edge_set = typename directed_topology_type::edge_set;

    using edge_size_type = typename directed_topology_type::edge_size_type;

    using vertex_user_value_type = VertexValue;
    using vertex_user_value_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<vertex_user_value_type>;
    using vertex_user_value_set = container<vertex_user_value_type>;

    using edge_user_value_type = EdgeValue;
    using edge_user_value_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<edge_user_value_type>;
    using edge_user_value_set = container<edge_user_value_type>;

    // ranges
    using vertex_edge_range = typename directed_topology_type::vertex_edge_range;
    using const_vertex_edge_range = typename directed_topology_type::const_vertex_edge_range;

    // ranges
    using vertex_outward_edge_range = typename directed_topology_type::vertex_outward_edge_range;
    using const_vertex_outward_edge_range =
        typename directed_topology_type::const_vertex_outward_edge_range;

    directed_adjacency_vector_graph_impl() = default;

    virtual ~directed_adjacency_vector_graph_impl() {
        auto& cols = _directed_topology._cols;
        auto& degrees = _directed_topology._degrees;
        auto& rows = _directed_topology._rows;
        auto& rows_vertex = _directed_topology._rows_vertex;

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
        _directed_topology._vertex_count = vertex_count;
        _directed_topology._edge_count = edge_count;
        _directed_topology._rows = edge_set::wrap(offsets, vertex_count + 1);
        _directed_topology._degrees = vertex_set::wrap(degrees, vertex_count);
        _directed_topology._cols = vertex_set::wrap(neighbors, edge_count);
        _directed_topology._rows_ptr = _directed_topology._rows.get_data();
        _directed_topology._cols_ptr = _directed_topology._cols.get_data();
        _directed_topology._degrees_ptr = _directed_topology._degrees.get_data();
    }

    inline void set_topology(vertex_size_type vertex_count,
                             edge_size_type edge_count,
                             const edge_type* offsets,
                             const vertex_type* neighbors,
                             const vertex_type* degrees) {
        _directed_topology._vertex_count = vertex_count;
        _directed_topology._edge_count = edge_count;
        _directed_topology._rows = edge_set::wrap(offsets, vertex_count + 1);
        _directed_topology._degrees = vertex_set::wrap(degrees, vertex_count);
        _directed_topology._cols = vertex_set::wrap(neighbors, edge_count);
        _directed_topology._rows_ptr = _directed_topology._rows.get_data();
        _directed_topology._cols_ptr = _directed_topology._cols.get_data();
        _directed_topology._degrees_ptr = _directed_topology._degrees.get_data();
    }

    inline directed_topology<IndexType>& get_topology() {
        return _directed_topology;
    }

    inline vertex_values<VertexValue>& get_vertex_values() {
        return _vertex_values;
    }

    inline edge_values<EdgeValue>& get_edge_values() {
        return _edge_values;
    }

    inline const directed_topology<IndexType> get_topology() const {
        return _directed_topology;
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
    directed_topology<IndexType> _directed_topology;
    vertex_values<VertexValue> _vertex_values;
    edge_values<EdgeValue> _edge_values;
};

} // namespace oneapi::dal::preview::detail
