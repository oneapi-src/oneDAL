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


template<typename Index>
class topology {
public:
    topology() = default;
    virtual ~topology() = default;

    Index* _vertex_neighbors;
    Index* _degrees;
    Index* _edge_offsets;
    int64_t _vertex_count;
    int64_t _edge_count;
};

template<typename VertexValue>
class vertex_values {
public:
    vertex_values() = default;
    virtual ~vertex_values() = default;

    VertexValue* _vertex_value;
};

template<typename EdgeValue>
class edge_values {
public:
    edge_values() = default;
    virtual ~edge_values() = default;

    EdgeValue* _edge_value;
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
    using vertex_iterator =  vertex_type*;
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
    using vertex_user_value_set =
        detail::graph_container<vertex_user_value_type, vertex_user_value_allocator_type>;

    using edge_user_value_type = EdgeValue;
    using edge_user_value_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<edge_user_value_type>;
    using edge_user_value_set =
        detail::graph_container<edge_user_value_type, edge_user_value_allocator_type>;

    undirected_adjacency_array_graph_impl() = default;
    ~undirected_adjacency_array_graph_impl() {
        if (_topology._vertex_neighbors != nullptr) {
            vertex_allocator_traits::deallocate(_vertex_allocator, _topology._vertex_neighbors,  2 * _topology._edge_count);
        }
        if (_topology._degrees != nullptr) {
            vertex_allocator_traits::deallocate(_vertex_allocator, _topology._degrees,  _topology._vertex_count);
        }
        if (_topology._edge_offsets != nullptr) {
            edge_allocator_traits::deallocate(_edge_allocator, _topology._edge_offsets,  1 + _topology._vertex_count);
        }
    }

    vertex_size_type _vertex_count;
    edge_size_type _edge_count;

    topology<IndexType> _topology;
    vertex_values<VertexValue> _vertex_values;
    edge_values<VertexValue> _edge_values;
    //vertex_user_value_set _vertex_value;
    //edge_user_value_set _edge_value;

    allocator_type _allocator;
    vertex_allocator_type _vertex_allocator;
    edge_allocator_type _edge_allocator;
};



} // namespace oneapi::dal::preview::detail
