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

#include <cstdint>
#include <memory>

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/data/array.hpp"
#include "oneapi/dal/data/detail/graph_container.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::preview::detail {

template <typename VertexValue = empty_value,
          typename EdgeValue   = empty_value,
          typename GraphValue  = empty_value,
          typename IndexType   = std::int64_t,
          typename Allocator   = std::allocator<empty_value>>
class undirected_adjacency_array_graph_impl {
public:
    using allocator_type = Allocator;

    using vertex_type = IndexType;
    using vertex_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<vertex_type>;
    using vertex_set       = array<vertex_type>;
    using vertex_size_type = std::int64_t;

    using edge_type = IndexType;
    using edge_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<edge_type>;
    using edge_set       = array<edge_type>;
    using edge_size_type = std::int64_t;

    using vertex_user_value_type = VertexValue;
    using vertex_user_value_set  = detail::graph_container<vertex_user_value_type, allocator_type>;

    using edge_user_value_type = EdgeValue;
    using edge_value_set       = detail::graph_container<edge_user_value_type, allocator_type>;

    undirected_adjacency_array_graph_impl() = default;

    vertex_size_type _vertex_count;
    edge_size_type _edge_count;

    vertex_set _vertex_neighbors;
    vertex_set _degrees;
    edge_set _edge_offsets;

    vertex_user_value_set _vertex_value;
    edge_value_set _edge_value;
};
} // namespace oneapi::dal::preview::detail
